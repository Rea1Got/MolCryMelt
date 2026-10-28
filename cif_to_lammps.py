#!/usr/bin/env python
"""
CIF (молекулярный кристалл) -> LAMMPS (data + input) с силовым полем OpenFF.

Что делает:
  1. Читает CIF через ASE (симметрия раскрывается сама).
  2. Находит молекулы по межатомным расстояниям (ковалентные радиусы) с учётом
     периодических границ и «собирает» каждую молекулу целиком.
  3. Определяет порядки связей / заряды / SMILES автоматически (RDKit
     rdDetermineBonds) -> SMILES не нужно хардкодить.
     Если автоопределение не сработало, можно подсказать --smiles (необязательно).
  4. Строит суперъячейку так, чтобы в результате было ровно N молекул.
  5. Параметризует через OpenFF Interchange и пишет LAMMPS .data / .in.
  6. (--liquid) Строит «жидкую» ячейку для метода склейки фаз: те же молекулы,
     случайно расположенные и ориентированные; сечение (векторы a, b) совпадает
     с кристаллической суперъячейкой, вектор c направлен строго вдоль z (без скосов).

Файлы пишутся в <output>/<имя_cif>/<имя_cif>_n<N>[_liquid].{data,in,pdb}

Примеры:
  python cif_to_lammps.py structures/128707.cif -n 64
  python cif_to_lammps.py structures/128707.cif -n 64 --liquid --seed 1
  python cif_to_lammps.py structures/128707.cif --supercell 1 5 3 -o runs
"""

import argparse
import itertools
import re
import sys
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
from ase.geometry import cell_to_cellpar
from ase.io import read as ase_read
from ase.neighborlist import natural_cutoffs, neighbor_list
from openff.interchange import Interchange
from openff.toolkit import ForceField, Molecule, Topology
from openff.units import unit
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="CIF молекулярного кристалла -> LAMMPS (OpenFF Interchange)."
    )
    p.add_argument("cif", help="Путь к .cif файлу.")
    p.add_argument(
        "-n",
        "--n-molecules",
        type=int,
        default=None,
        help="Сколько молекул должно быть в итоговой системе "
        "(должно быть кратно числу молекул в элементарной ячейке Z).",
    )
    p.add_argument(
        "--supercell",
        type=int,
        nargs=3,
        metavar=("NA", "NB", "NC"),
        default=None,
        help="Явно задать суперъячейку (вместо -n).",
    )
    p.add_argument(
        "-o",
        "--output",
        default="cells_openff",
        help="Корневая папка вывода. Файлы пишутся в "
        "<output>/<имя_cif>/<имя_cif>_n<N>.{data,in,pdb} (по умолчанию: cells_openff).",
    )
    p.add_argument(
        "-ff",
        "--forcefield",
        default="openff-2.3.0.offxml",
        help="Силовое поле OpenFF (по умолчанию: openff-2.3.0.offxml).",
    )
    p.add_argument(
        "--bond-scale",
        type=float,
        default=1.2,
        help="Множитель ковалентных радиусов для поиска связей (по умолчанию 1.2).",
    )
    p.add_argument(
        "--charge",
        type=int,
        default=None,
        help="Полный заряд каждой молекулы. По умолчанию перебираются 0, +1, -1, +2, -2.",
    )
    p.add_argument(
        "--smiles",
        action="append",
        default=[],
        help="(Необязательно) SMILES-шаблон, если автоопределение связей не "
        "справилось. Можно указать несколько раз.",
    )
    p.add_argument(
        "--cutoff",
        type=float,
        default=9.0,
        help="Радиус обрезки vdW (Å) — только для предупреждения о маленькой ячейке.",
    )
    p.add_argument(
        "--write-pdb",
        action="store_true",
        help="Дополнительно записать PDB (residue = молекула) для просмотра.",
    )
    p.add_argument(
        "--no-lammps",
        action="store_true",
        help="Только построить систему (и PDB), без параметризации.",
    )

    g = p.add_argument_group("жидкая ячейка (для склейки фаз)")
    g.add_argument(
        "--liquid",
        action="store_true",
        help="Дополнительно построить жидкую ячейку с тем же числом молекул.",
    )
    g.add_argument(
        "--liquid-z-factor",
        type=float,
        default=1.0,
        help="Множитель высоты жидкой ячейки вдоль z относительно кристалла "
        "(1.0 — тот же размер; 1.2–1.3 облегчает упаковку). По умолчанию 1.0.",
    )
    g.add_argument(
        "--min-dist",
        type=float,
        default=2.0,
        help="Начальный минимальный контакт между атомами разных молекул, Å "
        "(при неудаче порог автоматически снижается). По умолчанию 2.0.",
    )
    g.add_argument(
        "--seed", type=int, default=0, help="Seed генератора случайных чисел."
    )

    args = p.parse_args()
    if args.n_molecules is None and args.supercell is None:
        p.error("нужно указать -n/--n-molecules или --supercell")
    return args


# ----------------------------------------------------------------------------
# Поиск молекул в элементарной ячейке
# ----------------------------------------------------------------------------
def find_molecules(atoms, scale):
    """
    Возвращает список словарей {idx, symbols, coords, bonds} — по одному на
    молекулу элементарной ячейки. Координаты «собраны» (молекула целая,
    может выходить за пределы ячейки).
    """
    cell = np.array(atoms.cell)
    pos = atoms.get_positions()
    n = len(atoms)

    # проверка на совпадающие / слишком близкие атомы
    i_c, j_c, d_c = neighbor_list("ijd", atoms, 0.5)
    if len(i_c):
        pairs = sorted(
            {
                (min(i, j), max(i, j), round(float(d), 3))
                for i, j, d in zip(i_c, j_c, d_c)
            }
        )
        lines = "\n".join(
            f"  {atoms[i].symbol}{i}–{atoms[j].symbol}{j}: {d} Å"
            for i, j, d in pairs[:10]
        )
        raise ValueError(
            f"Найдено {len(pairs)} пар атомов ближе 0.5 Å (первые 10):\n{lines}\n"
            "Если среди них есть водороды — координаты H в CIF неверны; "
            "если тяжёлые атомы — разупорядоченность или ошибка симметрии."
        )

    cutoffs = natural_cutoffs(atoms, mult=scale)
    ii, jj, SS = neighbor_list("ijS", atoms, cutoffs)

    adj = defaultdict(list)
    for i, j, s in zip(ii, jj, SS):
        if i == j:
            raise ValueError(
                "Атом связан со своим периодическим образом — это не молекулярный "
                "кристалл (полимер / каркас)."
            )
        adj[i].append((j, s))

    shift = {}
    molecules = []
    for start in range(n):
        if start in shift:
            continue
        shift[start] = np.zeros(3, dtype=int)
        stack, members = [start], [start]
        while stack:
            i = stack.pop()
            for j, s in adj[i]:
                sj = shift[i] + s
                if j not in shift:
                    shift[j] = sj
                    stack.append(j)
                    members.append(j)
                elif not np.array_equal(shift[j], sj):
                    raise ValueError(
                        "Молекула замыкается через границу ячейки (бесконечная "
                        "цепь/слой). Это не молекулярный кристалл."
                    )
        members = sorted(members)
        local = {g: k for k, g in enumerate(members)}
        bonds = sorted(
            {
                (min(local[i], local[j]), max(local[i], local[j]))
                for i in members
                for j, _ in adj[i]
            }
        )
        coords = np.array([pos[i] + shift[i] @ cell for i in members])
        symbols = [atoms[i].symbol for i in members]
        molecules.append(dict(idx=members, symbols=symbols, coords=coords, bonds=bonds))
    return molecules


# ----------------------------------------------------------------------------
# Определение химии молекулы (порядки связей, заряды)
# ----------------------------------------------------------------------------
def perceive_with_rdkit(mol_data, charges):
    rw = Chem.RWMol()
    for sym in mol_data["symbols"]:
        a = Chem.Atom(sym)
        a.SetNoImplicit(True)
        rw.AddAtom(a)
    for i, j in mol_data["bonds"]:
        rw.AddBond(int(i), int(j), Chem.BondType.SINGLE)
    m = rw.GetMol()
    conf = Chem.Conformer(m.GetNumAtoms())
    for k, (x, y, z) in enumerate(mol_data["coords"]):
        conf.SetAtomPosition(k, (float(x), float(y), float(z)))
    m.AddConformer(conf, assignId=True)

    last_err = None
    for q in charges:
        trial = Chem.Mol(m)
        try:
            rdDetermineBonds.DetermineBondOrders(
                trial, charge=q, allowChargedFragments=True, embedChiral=True
            )
            Chem.SanitizeMol(trial)
            Chem.AssignStereochemistryFrom3D(trial)
            off = Molecule.from_rdkit(
                trial, allow_undefined_stereo=True, hydrogens_are_explicit=True
            )
            return off, q
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise RuntimeError(
        f"Не удалось определить порядки связей (пробовал заряды {charges}): {last_err}\n"
        "Проверьте, что в CIF есть все водороды, или передайте --smiles."
    )


def match_with_smiles(mol_data, templates):
    g = nx.Graph()
    for k, sym in enumerate(mol_data["symbols"]):
        g.add_node(k, el=sym)
    g.add_edges_from(mol_data["bonds"])
    for t in templates:
        tg = nx.Graph()
        for a in t.atoms:
            tg.add_node(a.molecule_atom_index, el=a.symbol)
        for b in t.bonds:
            tg.add_edge(b.atom1_index, b.atom2_index)
        gm = nx.isomorphism.GraphMatcher(
            g, tg, node_match=lambda x, y: x["el"] == y["el"]
        )
        if gm.is_isomorphic():
            xyz = np.zeros((t.n_atoms, 3))
            for src, dst in gm.mapping.items():
                xyz[dst] = mol_data["coords"][src]
            off = Molecule(t)
            off.clear_conformers()
            off.add_conformer(xyz * unit.angstrom)
            return off
    return None


def build_templates(mols, args):
    """Для каждой молекулы элементарной ячейки возвращает openff Molecule с конформером."""
    smiles_templates = [
        Molecule.from_smiles(s, allow_undefined_stereo=True) for s in args.smiles
    ]
    charges = [args.charge] if args.charge is not None else [0, 1, -1, 2, -2]
    out = []
    for k, m in enumerate(mols):
        off = match_with_smiles(m, smiles_templates) if smiles_templates else None
        if off is None:
            off, q = perceive_with_rdkit(m, charges)
            if q != 0:
                print(f"  ⚠️ молекула {k}: определён ненулевой заряд {q:+d}")
        out.append(off)
    return out


# ----------------------------------------------------------------------------
# Суперъячейка
# ----------------------------------------------------------------------------
def perp_widths(cell):
    vol = abs(np.linalg.det(cell))
    return np.array(
        [
            vol / np.linalg.norm(np.cross(cell[(i + 1) % 3], cell[(i + 2) % 3]))
            for i in range(3)
        ]
    )


def reduce_tilt(cell):
    """
    Приводит нижнетреугольную ячейку к виду, допустимому в LAMMPS:
    |xy|, |xz| <= lx/2 и |yz| <= ly/2. Решётка не меняется, только выбор векторов.
    """
    a, b, c = (np.array(v, dtype=float) for v in cell)
    b = b - round(b[0] / a[0]) * a
    c = c - round(c[1] / b[1]) * b
    c = c - round(c[0] / a[0]) * a
    return np.array([a, b, c])


def patch_lammps_box(data_file, sc_cell):
    """Interchange пишет только прямоугольный бокс: вписываем реальные тильт-факторы."""
    xy, xz, yz = sc_cell[1, 0], sc_cell[2, 0], sc_cell[2, 1]
    tilt_line = f"{xy:.8f} {xz:.8f} {yz:.8f} xy xz yz\n"
    with open(data_file) as f:
        lines = f.readlines()
    for k, line in enumerate(lines):
        if re.search(r"\bxy\s+xz\s+yz\b", line):
            lines[k] = tilt_line
            break
    else:
        for k, line in enumerate(lines):
            if re.search(r"\bzlo\s+zhi\b", line):
                lines.insert(k + 1, tilt_line)
                break
        else:
            raise RuntimeError(
                "В .data не найдена строка 'zlo zhi' — не могу вписать тильты."
            )
    with open(data_file, "w") as f:
        f.writelines(lines)


def choose_supercell(n_cells, cell):
    """Среди всех na*nb*nc == n_cells берём ту, у которой минимальная толщина ячейки максимальна."""
    best, best_key = None, None
    for na in range(1, n_cells + 1):
        if n_cells % na:
            continue
        for nb in range(1, n_cells // na + 1):
            if (n_cells // na) % nb:
                continue
            nc = n_cells // na // nb
            ww = perp_widths(reduce_tilt(cell * np.array([na, nb, nc])[:, None]))
            key = (round(ww.min(), 6), -round(ww.max(), 6))
            if best_key is None or key > best_key:
                best, best_key = (na, nb, nc), key
    return best


def build_supercell(templates, mols, cell, sc):
    """Возвращает (список openff Molecule с конформерами, суперъячейка 3x3)."""
    sc_cell = reduce_tilt(cell * np.array(sc)[:, None])
    inv = np.linalg.inv(sc_cell)
    placed = []
    for tmpl, m in zip(templates, mols):
        base = np.asarray(tmpl.conformers[0].m_as(unit.angstrom))
        for ia, ib, ic in itertools.product(range(sc[0]), range(sc[1]), range(sc[2])):
            xyz = base + ia * cell[0] + ib * cell[1] + ic * cell[2]
            # заносим молекулу целиком в ячейку по геометрическому центру
            frac = xyz.mean(axis=0) @ inv
            xyz = xyz - np.floor(frac) @ sc_cell
            mol = Molecule(tmpl)
            mol.clear_conformers()
            mol.add_conformer(xyz * unit.angstrom)
            placed.append(mol)
    return placed, sc_cell


# ----------------------------------------------------------------------------
# Жидкая ячейка: случайная упаковка жёстких молекул
# ----------------------------------------------------------------------------
def random_rotation(rng):
    """Равномерно распределённая случайная матрица поворота (через кватернион)."""
    q = rng.normal(size=4)
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _wrap(d, cell, inv):
    """Минимальный образ вектора(ов) d в периодической ячейке."""
    f = d @ inv
    f -= np.round(f)
    return f @ cell


def place_random(rel, cell, min_dist, rng, max_attempts=3000, floor=1.2, step=0.1):
    """
    Случайно размещает жёсткие молекулы (rel[k] — координаты относительно центра)
    в периодической ячейке. Контакт между атомами разных молекул >= порога;
    если упаковать не удаётся, порог снижается на step до floor.
    Возвращает (список координат, итоговый порог).
    """
    inv = np.linalg.inv(cell)
    n = len(rel)
    rad = np.array([np.linalg.norm(r, axis=1).max() for r in rel])
    order = sorted(range(n), key=lambda k: -len(rel[k]))  # крупные первыми
    placed, centers, done = [None] * n, np.zeros((n, 3)), []
    thr = min_dist
    for k in order:
        while True:
            hit = None
            for _ in range(max_attempts):
                c = rng.random(3) @ cell
                xyz = rel[k] @ random_rotation(rng).T + c
                if not done:
                    hit = xyz
                    break
                idx = np.array(done)
                dc = np.linalg.norm(_wrap(centers[idx] - c, cell, inv), axis=1)
                near = idx[dc < rad[k] + rad[idx] + thr]
                if len(near) == 0:
                    hit = xyz
                    break
                other = np.concatenate([placed[j] for j in near])
                diff = (xyz[:, None, :] - other[None, :, :]).reshape(-1, 3)
                if np.linalg.norm(_wrap(diff, cell, inv), axis=1).min() >= thr:
                    hit = xyz
                    break
            if hit is not None:
                placed[k], centers[k] = hit, hit.mean(axis=0)
                done.append(k)
                break
            thr = round(thr - step, 3)
            if thr < floor:
                raise RuntimeError(
                    f"Не удалось упаковать молекулу {len(done) + 1} из {n} даже при "
                    f"контакте {floor} Å. Увеличьте --liquid-z-factor (например, 1.3)."
                )
            print(f"  порог контакта снижен до {thr:.1f} Å (размещено {len(done)}/{n})")
    return placed, thr


def build_liquid(molecules, liq_cell, min_dist, rng):
    """Берёт те же молекулы, что и в кристалле, и случайно расставляет/вращает их."""
    rel = []
    for m in molecules:
        xyz = np.asarray(m.conformers[0].m_as(unit.angstrom))
        rel.append(xyz - xyz.mean(axis=0))
    placed, thr = place_random(rel, liq_cell, min_dist, rng)
    out = []
    for m, xyz in zip(molecules, placed):
        mol = Molecule(m)
        mol.clear_conformers()
        mol.add_conformer(xyz * unit.angstrom)
        out.append(mol)
    return out, thr


# ----------------------------------------------------------------------------
# Вывод
# ----------------------------------------------------------------------------
def write_pdb(path, molecules, sc_cell):
    a, b, c, al, be, ga = cell_to_cellpar(sc_cell)
    with open(path, "w") as f:
        f.write(
            f"CRYST1{a:9.3f}{b:9.3f}{c:9.3f}{al:7.2f}{be:7.2f}{ga:7.2f} P 1           1\n"
        )
        serial = 0
        for r, mol in enumerate(molecules, 1):
            xyz = mol.conformers[0].m_as(unit.angstrom)
            for k, atom in enumerate(mol.atoms):
                serial += 1
                name = f"{atom.symbol}{k + 1}"[:4]
                x, y, z = xyz[k]
                f.write(
                    f"ATOM  {serial % 100000:5d} {name:<4s} MOL A{r % 10000:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {atom.symbol:>2s}\n"
                )
            f.write("TER\n")
        f.write("END\n")


def export_lammps(molecules, cell, ff, data_file, input_file):
    """Параметризация через Interchange и запись LAMMPS .data / .in."""
    topology = Topology.from_molecules(molecules)
    # Interchange умеет экспортировать в LAMMPS только прямоугольный бокс:
    # отдаём ему диагональ, а реальные тильты вписываем в .data после экспорта.
    box_vectors = np.diag(np.diag(cell)) * unit.angstrom
    topology.box_vectors = box_vectors
    positions = (
        np.concatenate([m.conformers[0].m_as(unit.angstrom) for m in molecules])
        * unit.angstrom
    )
    print(f"  молекул: {topology.n_molecules}, атомов: {topology.n_atoms}")
    interchange = Interchange.from_smirnoff(
        force_field=ff, topology=topology, box=box_vectors, positions=positions
    )
    interchange.to_lammps_datafile(data_file)
    interchange.to_lammps_input(input_file, data_file=data_file)
    patch_lammps_box(data_file, cell)
    print(
        f"  бокс: lx,ly,lz = {np.diag(cell).round(4)}, "
        f"xy,xz,yz = {cell[1, 0]:.4f}, {cell[2, 0]:.4f}, {cell[2, 1]:.4f}"
    )
    print(f"  {data_file}\n  {input_file}")


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    args = parse_args()
    stem = Path(args.cif).stem
    outdir = Path(args.output) / stem
    outdir.mkdir(parents=True, exist_ok=True)

    atoms = ase_read(args.cif)
    atoms.pbc = True
    cell = np.array(atoms.cell)
    print(
        f"CIF: {args.cif}, атомов в ячейке: {len(atoms)}, "
        f"формула: {atoms.get_chemical_formula()}"
    )

    # --- молекулы в элементарной ячейке
    mols = find_molecules(atoms, args.bond_scale)
    z = len(mols)
    print(f"Молекул в элементарной ячейке (Z): {z}")

    print("Определение химии молекул...")
    templates = build_templates(mols, args)
    for k, t in enumerate(templates):
        print(
            f"  молекула {k}: {t.to_smiles(explicit_hydrogens=False)}  ({t.n_atoms} атомов)"
        )

    # --- суперъячейка
    if args.supercell is not None:
        sc = tuple(args.supercell)
    else:
        if args.n_molecules % z:
            lo = (args.n_molecules // z) * z
            sys.exit(
                f"Ошибка: N={args.n_molecules} не кратно Z={z}. "
                f"Ближайшие допустимые: {lo or z} или {lo + z}."
            )
        sc = choose_supercell(args.n_molecules // z, cell)
    n_total = int(z * np.prod(sc))
    prefix = f"{stem}_n{n_total}"
    print(f"Суперъячейка: {sc[0]} x {sc[1]} x {sc[2]}  ->  молекул: {n_total}")

    molecules, sc_cell = build_supercell(templates, mols, cell, sc)
    widths = perp_widths(sc_cell)
    print("Параметры ячейки (a b c α β γ):", np.round(cell_to_cellpar(sc_cell), 4))
    print("Толщины ячейки между гранями, Å:", np.round(widths, 2))
    if widths.min() < 2 * args.cutoff:
        print(
            f"⚠️ Минимальная толщина {widths.min():.1f} Å < 2*cutoff "
            f"({2 * args.cutoff:.1f} Å): минимальные образы в LAMMPS (особенно для "
            "связных взаимодействий) могут считаться неверно. Возьмите больше молекул."
        )

    mass_total = atoms.get_masses().sum() * np.prod(sc)  # а.е.м.
    dens = lambda cell_: (
        mass_total * 1.66054 / abs(np.linalg.det(cell_))
    )  # г/см³  # noqa: E731
    print(f"Плотность кристалла: {dens(sc_cell):.3f} г/см³")

    # --- жидкая ячейка
    liquid, liq_cell = None, None
    if args.liquid:
        # те же a, b (сечение), c строго вдоль z -> без скосов по z
        liq_cell = np.array(
            [
                sc_cell[0],
                sc_cell[1],
                [0.0, 0.0, sc_cell[2, 2] * args.liquid_z_factor],
            ]
        )
        print(
            f"\nЖидкая ячейка: lx,ly,lz = {np.diag(liq_cell).round(3)}, "
            f"xy = {liq_cell[1, 0]:.3f}, плотность {dens(liq_cell):.3f} г/см³"
        )
        lw = perp_widths(liq_cell)
        if lw.min() < 2 * args.cutoff:
            print(f"⚠️ Минимальная толщина жидкой ячейки {lw.min():.1f} Å < 2*cutoff.")
        print(f"Случайная упаковка {n_total} молекул (seed={args.seed})...")
        rng = np.random.default_rng(args.seed)
        liquid, thr = build_liquid(molecules, liq_cell, args.min_dist, rng)
        print(f"  готово, минимальный контакт между молекулами >= {thr:.1f} Å")
        if thr < 1.8:
            print(
                "  ⚠️ контакты тесные: перед MD сделайте минимизацию / мягкий "
                "push-off, либо увеличьте --liquid-z-factor."
            )

    if args.write_pdb:
        write_pdb(outdir / f"{prefix}.pdb", molecules, sc_cell)
        print(f"PDB: {outdir / (prefix + '.pdb')}")
        if liquid is not None:
            write_pdb(outdir / f"{prefix}_liquid.pdb", liquid, liq_cell)
            print(f"PDB: {outdir / (prefix + '_liquid.pdb')}")
    if args.no_lammps:
        return

    # --- параметризация и запись
    print(f"\nЗагрузка силового поля: {args.forcefield}")
    ff = ForceField(args.forcefield)

    print("Параметризация кристалла (AM1-BCC может занять минуту)...")
    export_lammps(
        molecules,
        sc_cell,
        ff,
        str(outdir / f"{prefix}.data"),
        str(outdir / f"{prefix}.in"),
    )
    if liquid is not None:
        print("Параметризация жидкости...")
        export_lammps(
            liquid,
            liq_cell,
            ff,
            str(outdir / f"{prefix}_liquid.data"),
            str(outdir / f"{prefix}_liquid.in"),
        )

    print("\n✅ Готово!")


if __name__ == "__main__":
    main()
