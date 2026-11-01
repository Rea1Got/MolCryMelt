#!/usr/bin/env python3
"""
Склейка двух LAMMPS data-файлов (atom_style full) вдоль z: кристалл внизу, жидкость сверху.

  python3 join_phases.py crystal_eq.data liquid_eq.data joined.data

Атомы разворачиваются по image-флагам (молекулы становятся целыми), жидкость сдвигается
на верхнюю грань кристалла, коробка = a, b, c кристалла + высота жидкости по z.
Image-флаги в результате обнулены (LAMMPS сам обернёт атомы при read_data).
Сечение (lx, ly, xy) у обоих файлов должно совпадать. Velocities не переносятся.
"""

import re
import sys

SECTIONS = {
    "Atoms",
    "Velocities",
    "Masses",
    "Pair Coeffs",
    "PairIJ Coeffs",
    "Bond Coeffs",
    "Angle Coeffs",
    "Dihedral Coeffs",
    "Improper Coeffs",
    "Bonds",
    "Angles",
    "Dihedrals",
    "Impropers",
    "BondBond Coeffs",
    "BondAngle Coeffs",
    "MiddleBondTorsion Coeffs",
    "EndBondTorsion Coeffs",
    "AngleTorsion Coeffs",
    "AngleAngleTorsion Coeffs",
    "BondBond13 Coeffs",
    "AngleAngle Coeffs",
}
TOPO = {
    "Bonds": "bonds",
    "Angles": "angles",
    "Dihedrals": "dihedrals",
    "Impropers": "impropers",
}
NATOM = {"Bonds": 2, "Angles": 3, "Dihedrals": 4, "Impropers": 4}


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


def unwrapped_atoms(h, secs):
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
        atoms[i] = (mol, typ, q, x, y, z)
    return atoms, (lx, ly, lz)


def main(fc, fl, fout):
    hc, sc, order = parse(fc)
    hl, sl, _ = parse(fl)
    if hc["other"] != hl["other"]:
        sys.exit(
            "Заголовки (число типов и т.п.) в файлах различаются: типы не совпадают."
        )
    mc = [r.split("#")[0].split() for r in sc["Masses"]["rows"]]
    ml = [r.split("#")[0].split() for r in sl["Masses"]["rows"]]
    if mc != ml:
        sys.exit("Секции Masses различаются: нумерация типов в файлах не совпадает.")

    ac, (lxc, lyc, lzc) = unwrapped_atoms(hc, sc)
    al, (lxl, lyl, lzl) = unwrapped_atoms(hl, sl)
    if (
        abs(lxc - lxl) > 1e-4
        or abs(lyc - lyl) > 1e-4
        or abs(hc["tilt"][0] - hl["tilt"][0]) > 1e-4
    ):
        sys.exit(
            f"Сечения не совпадают: кристалл {lxc:.6f} x {lyc:.6f}, жидкость {lxl:.6f} x {lyl:.6f}"
        )

    # сдвиг жидкости: нижняя грань её коробки -> верхняя грань кристалла
    dx = hc["box"]["x"][0] - hl["box"]["x"][0]
    dy = hc["box"]["y"][0] - hl["box"]["y"][0]
    dz = hc["box"]["z"][0] + lzc - hl["box"]["z"][0]
    off_atom = max(ac)
    off_mol = max(a[0] for a in ac.values())

    new_box = {
        "x": hc["box"]["x"],
        "y": hc["box"]["y"],
        "z": (hc["box"]["z"][0], hc["box"]["z"][0] + lzc + lzl),
    }
    out = ["LAMMPS data file: crystal + liquid (join_phases.py)", ""]
    counts = {k: hc["count"].get(k, 0) + hl["count"].get(k, 0) for k in hc["count"]}
    for k in ("atoms", "bonds", "angles", "dihedrals", "impropers"):
        if k in counts:
            out.append(f"{counts[k]} {k}")
    out += hc["other"] + [""]
    for d in "xyz":
        out.append(f"{new_box[d][0]:.10f} {new_box[d][1]:.10f} {d}lo {d}hi")
    out.append("{:.10f} {:.10f} {:.10f} xy xz yz".format(*hc["tilt"]))

    for name in order:
        if name == "Velocities":
            continue
        out += ["", sc[name]["title"], ""]
        if name == "Atoms":
            for i in sorted(ac):
                mol, typ, q, x, y, z = ac[i]
                out.append(f"{i} {mol} {typ} {q} {x:.10f} {y:.10f} {z:.10f} 0 0 0")
            for i in sorted(al):
                mol, typ, q, x, y, z = al[i]
                out.append(
                    f"{i + off_atom} {mol + off_mol} {typ} {q} "
                    f"{x + dx:.10f} {y + dy:.10f} {z + dz:.10f} 0 0 0"
                )
        elif name in TOPO:
            n = NATOM[name]
            rows_c = [r.split("#")[0].split() for r in sc[name]["rows"]]
            rows_l = [r.split("#")[0].split() for r in sl[name]["rows"]]
            off_t = max((int(r[0]) for r in rows_c), default=0)
            for r in rows_c:
                out.append(" ".join(r))
            for r in rows_l:
                ids = [int(v) + off_atom for v in r[2 : 2 + n]]
                out.append(
                    " ".join([str(int(r[0]) + off_t), r[1]] + [str(v) for v in ids])
                )
        else:
            out += sc[name]["rows"]
    with open(fout, "w") as f:
        f.write("\n".join(out) + "\n")
    print(
        f"join_phases: {counts.get('atoms')} атомов, коробка по z = {lzc + lzl:.4f} "
        f"(кристалл {lzc:.4f} + жидкость {lzl:.4f})"
    )


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(*sys.argv[1:])
