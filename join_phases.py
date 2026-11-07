#!/usr/bin/env python3
"""
Склейка двух LAMMPS data-файлов (atom_style full) вдоль z: кристалл внизу, жидкость сверху.

  python3 join_phases.py crystal_eq.data liquid_eq.data joined.data [gap]

* Атомы разворачиваются по image-флагам (молекулы целые), затем КАЖДАЯ молекула целиком
  переносится на векторы своей ячейки так, чтобы её центр лежал внутри ячейки своей фазы
  (иначе молекулы жидкости, ушедшие за границу при диффузии, попали бы в кристалл).
* Жидкость ставится над кристаллом; gap (Å, по умолчанию 0) оставляется на КАЖДОЙ из двух
  границ раздела (кристалл|жидкость и жидкость|кристалл через PBC), чтобы не было перекрытий.
* Коробка: a, b и наклон кристалла, высота = Lc + Ll + 2*gap.
* Скорости обеих фаз сохраняются (если секция Velocities есть в обоих файлах).
* Сечения (lx, ly, xy) у файлов должны совпадать; нумерация типов — тоже.
"""
import math
import re
import sys

SECTIONS = {
    "Atoms", "Velocities", "Masses", "Pair Coeffs", "PairIJ Coeffs", "Bond Coeffs",
    "Angle Coeffs", "Dihedral Coeffs", "Improper Coeffs", "Bonds", "Angles",
    "Dihedrals", "Impropers", "BondBond Coeffs", "BondAngle Coeffs",
    "MiddleBondTorsion Coeffs", "EndBondTorsion Coeffs", "AngleTorsion Coeffs",
    "AngleAngleTorsion Coeffs", "BondBond13 Coeffs", "AngleAngle Coeffs",
}
TOPO = {"Bonds": 2, "Angles": 3, "Dihedrals": 4, "Impropers": 4}


def parse(path):
    with open(path) as f:
        lines = f.read().splitlines()
    header, secs, order, cur = [], {}, [], None
    for ln in lines[1:]:
        t = ln.split("#")[0].strip()
        if t in SECTIONS:
            cur = t
            order.append(t)
            secs[t] = {"title": ln, "rows": []}
        elif cur is None:
            header.append(ln)
        elif ln.strip():
            secs[cur]["rows"].append(ln)
    h = {"other": [], "count": {}, "box": {}, "tilt": (0.0, 0.0, 0.0)}
    for ln in header:
        s = ln.split("#")[0].strip()
        if not s:
            continue
        m = re.match(r"^(\d+)\s+(atoms|bonds|angles|dihedrals|impropers)$", s)
        if m:
            h["count"][m.group(2)] = int(m.group(1))
            continue
        m = re.match(r"^(\S+)\s+(\S+)\s+([xyz])lo\s+[xyz]hi$", s)
        if m:
            h["box"][m.group(3)] = (float(m.group(1)), float(m.group(2)))
            continue
        m = re.match(r"^(\S+)\s+(\S+)\s+(\S+)\s+xy\s+xz\s+yz$", s)
        if m:
            h["tilt"] = tuple(float(m.group(i)) for i in (1, 2, 3))
            continue
        h["other"].append(s)
    return h, secs, order


def phase_atoms(h, secs):
    """{id: [mol, type, q, x, y, z]}: молекулы целые, центр каждой молекулы внутри ячейки фазы."""
    xlo, ylo, zlo = (h["box"][d][0] for d in "xyz")
    lx, ly, lz = (h["box"][d][1] - h["box"][d][0] for d in "xyz")
    xy, xz, yz = h["tilt"]
    atoms = {}
    for row in secs["Atoms"]["rows"]:
        f = row.split("#")[0].split()
        i, mol, typ, q = int(f[0]), int(f[1]), int(f[2]), f[3]
        x, y, z = float(f[4]), float(f[5]), float(f[6])
        nx, ny, nz = (int(f[7]), int(f[8]), int(f[9])) if len(f) >= 10 else (0, 0, 0)
        x += nx * lx + ny * xy + nz * xz
        y += ny * ly + nz * yz
        z += nz * lz
        atoms[i] = [mol, typ, q, x, y, z]
    mols = {}
    for i, a in atoms.items():
        mols.setdefault(a[0], []).append(i)
    for ids in mols.values():
        n = len(ids)
        cx = sum(atoms[i][3] for i in ids) / n - xlo
        cy = sum(atoms[i][4] for i in ids) / n - ylo
        cz = sum(atoms[i][5] for i in ids) / n - zlo
        fz = cz / lz
        fy = (cy - fz * yz) / ly
        fx = (cx - fy * xy - fz * xz) / lx
        nx, ny, nz = math.floor(fx), math.floor(fy), math.floor(fz)
        sx = -(nx * lx + ny * xy + nz * xz)
        sy = -(ny * ly + nz * yz)
        sz = -nz * lz
        for i in ids:
            atoms[i][3] += sx
            atoms[i][4] += sy
            atoms[i][5] += sz
    return atoms, (lx, ly, lz)


def main(fc, fl, fout, gap=0.0):
    hc, sc, order = parse(fc)
    hl, sl, _ = parse(fl)
    if hc["other"] != hl["other"]:
        sys.exit("Заголовки (число типов и т.п.) в файлах различаются: типы не совпадают.")
    mc = [r.split("#")[0].split() for r in sc["Masses"]["rows"]]
    ml = [r.split("#")[0].split() for r in sl["Masses"]["rows"]]
    if mc != ml:
        sys.exit("Секции Masses различаются: нумерация типов в файлах не совпадает.")

    ac, (lxc, lyc, lzc) = phase_atoms(hc, sc)
    al, (lxl, lyl, lzl) = phase_atoms(hl, sl)
    if abs(lxc - lxl) > 1e-4 or abs(lyc - lyl) > 1e-4 or abs(hc["tilt"][0] - hl["tilt"][0]) > 1e-4:
        sys.exit(f"Сечения не совпадают: кристалл {lxc:.6f} x {lyc:.6f}, жидкость {lxl:.6f} x {lyl:.6f}")

    # жидкость: нижняя грань её ячейки -> верхняя грань кристалла + gap
    dx = hc["box"]["x"][0] - hl["box"]["x"][0]
    dy = hc["box"]["y"][0] - hl["box"]["y"][0]
    dz = hc["box"]["z"][0] + lzc + gap - hl["box"]["z"][0]
    off_atom = max(ac)
    off_mol = max(a[0] for a in ac.values())
    zlo_new = hc["box"]["z"][0]
    zhi_new = zlo_new + lzc + lzl + 2.0 * gap

    out = ["LAMMPS data file: crystal + liquid (join_phases.py)", ""]
    counts = {k: hc["count"].get(k, 0) + hl["count"].get(k, 0) for k in hc["count"]}
    for k in ("atoms", "bonds", "angles", "dihedrals", "impropers"):
        if k in counts:
            out.append(f"{counts[k]} {k}")
    out += hc["other"] + [""]
    out.append("{:.10f} {:.10f} xlo xhi".format(*hc["box"]["x"]))
    out.append("{:.10f} {:.10f} ylo yhi".format(*hc["box"]["y"]))
    out.append(f"{zlo_new:.10f} {zhi_new:.10f} zlo zhi")
    out.append("{:.10f} {:.10f} {:.10f} xy xz yz".format(*hc["tilt"]))

    have_vel = "Velocities" in sc and "Velocities" in sl
    for name in order:
        if name == "Velocities" and not have_vel:
            continue
        out += ["", sc[name]["title"], ""]
        if name == "Atoms":
            for i in sorted(ac):
                mol, typ, q, x, y, z = ac[i]
                out.append(f"{i} {mol} {typ} {q} {x:.10f} {y:.10f} {z:.10f} 0 0 0")
            for i in sorted(al):
                mol, typ, q, x, y, z = al[i]
                out.append(f"{i + off_atom} {mol + off_mol} {typ} {q} "
                           f"{x + dx:.10f} {y + dy:.10f} {z + dz:.10f} 0 0 0")
        elif name == "Velocities":
            vc = {int(r.split()[0]): r.split()[1:] for r in sc[name]["rows"]}
            vl = {int(r.split()[0]): r.split()[1:] for r in sl[name]["rows"]}
            for i in sorted(vc):
                out.append(" ".join([str(i)] + vc[i]))
            for i in sorted(vl):
                out.append(" ".join([str(i + off_atom)] + vl[i]))
        elif name in TOPO:
            n = TOPO[name]
            rows_c = [r.split("#")[0].split() for r in sc[name]["rows"]]
            rows_l = [r.split("#")[0].split() for r in sl[name]["rows"]]
            off_t = max((int(r[0]) for r in rows_c), default=0)
            for r in rows_c:
                out.append(" ".join(r))
            for r in rows_l:
                ids = [str(int(v) + off_atom) for v in r[2:2 + n]]
                out.append(" ".join([str(int(r[0]) + off_t), r[1]] + ids))
        else:
            out += sc[name]["rows"]
    with open(fout, "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"join_phases: {counts.get('atoms')} атомов, коробка по z = {zhi_new - zlo_new:.4f} "
          f"(кристалл {lzc:.4f} + жидкость {lzl:.4f} + 2x зазор {gap}), "
          f"скорости {'сохранены' if have_vel else 'НЕ сохранены (нет в файле)'}")


if __name__ == "__main__":
    if len(sys.argv) not in (4, 5):
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]) if len(sys.argv) == 5 else 0.0)
