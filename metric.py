#!/usr/bin/env python3
"""
f(z): fraction of "crystal-like" molecules along z (1 = crystal, 0 = liquid) from a LAMMPS dump.

Idea (template matching of the local environment against the CIF):
  * Reference: for every molecule of the CIF unit cell, take the heavy-atom positions of all
    OTHER molecules within --radius of its centroid (vectors relative to the centroid, lab frame).
  * Simulation: for every molecule, put the reference vectors at its centroid and count the
    fraction of them that have a heavy atom within --tol (periodic box, triclinic OK).
    Score in [0, 1]; the best of the Z reference molecules is taken. score >= threshold -> crystal.
  * The crystal in the box keeps the orientation of the CIF-built supercell, so no rotation
    fitting is needed. Lattice strain (thermal expansion) is fitted once on frame 0 (axis scales)
    and then follows lx, ly of the box.

Needs: numpy, scipy; ASE only to read the CIF; matplotlib only for --plot.

Example:
  python3 crystal_fraction.py T400_P1_n120.lammpstrj --cif structures/610381.cif \
      --data T400_P1_n120_joined_start.data --n-mol-cryst 120 --stride 5 --nbins 40 --plot
Dump columns required: id mol type x y z (ix iy iz recommended: molecules are unwrapped with them).
"""

import argparse
import itertools
import math
import sys

import numpy as np
from scipy.spatial import cKDTree


# ----------------------------------------------------------------------------
# LAMMPS data / dump
# ----------------------------------------------------------------------------
def read_masses(data_file):
    """{atom type: mass} from the Masses section of a LAMMPS data file."""
    masses, inside = {}, False
    with open(data_file) as f:
        for line in f:
            s = line.split("#")[0].strip()
            if s == "Masses":
                inside = True
                continue
            if inside:
                if not s:
                    if masses:
                        break
                    continue
                parts = s.split()
                if len(parts) < 2 or not parts[0].isdigit():
                    break
                masses[int(parts[0])] = float(parts[1])
    if not masses:
        sys.exit(f"Masses section not found in {data_file}")
    return masses


def make_cell(b, tri):
    """cell rows (a, b, c) and origin from the dump BOX BOUNDS lines."""
    xlo_b, xhi_b = float(b[0][0]), float(b[0][1])
    ylo_b, yhi_b = float(b[1][0]), float(b[1][1])
    zlo, zhi = float(b[2][0]), float(b[2][1])
    xy = xz = yz = 0.0
    if tri:
        xy, xz, yz = float(b[0][2]), float(b[1][2]), float(b[2][2])
    xlo = xlo_b - min(0.0, xy, xz, xy + xz)
    xhi = xhi_b - max(0.0, xy, xz, xy + xz)
    ylo = ylo_b - min(0.0, yz)
    yhi = yhi_b - max(0.0, yz)
    cell = np.array([[xhi - xlo, 0, 0], [xy, yhi - ylo, 0], [xz, yz, zhi - zlo]])
    return cell, np.array([xlo, ylo, zlo])


def iter_dump(path, stride=1, step_every=0):
    """Yield (step, cell, origin, columns, data) for the selected frames."""
    with open(path) as f:
        k = -1
        while True:
            line = f.readline()
            if not line:
                return
            if not line.startswith("ITEM: TIMESTEP"):
                continue
            step = int(f.readline())
            f.readline()
            n = int(f.readline())
            tri = "xy" in f.readline()
            b = [f.readline().split() for _ in range(3)]
            cols = f.readline().split()[2:]
            k += 1
            if k % stride != 0 or (step_every and step % step_every != 0):
                for _ in range(n):
                    f.readline()
                continue
            data = np.array([f.readline().split() for _ in range(n)], dtype=float)
            cell, origin = make_cell(b, tri)
            yield step, cell, origin, cols, data


# ----------------------------------------------------------------------------
# reference environment from the crystal (CIF)
# ----------------------------------------------------------------------------
def reference_from_cif(path, bond_scale=1.2):
    """cell, unwrapped positions, heavy mask and molecule labels of the CIF unit cell (needs ASE)."""
    from ase.io import read
    from ase.neighborlist import natural_cutoffs, neighbor_list

    atoms = read(path)
    atoms.pbc = True
    cell = np.array(atoms.cell)
    pos = atoms.get_positions()
    ii, jj, SS = neighbor_list("ijS", atoms, natural_cutoffs(atoms, mult=bond_scale))
    adj = {}
    for i, j, s in zip(ii, jj, SS):
        adj.setdefault(i, []).append((j, s))
    shift, label, unwrapped = {}, np.full(len(atoms), -1), pos.copy()
    nmol = 0
    for start in range(len(atoms)):
        if start in shift:
            continue
        shift[start] = np.zeros(3, dtype=int)
        stack = [start]
        label[start] = nmol
        while stack:
            i = stack.pop()
            for j, s in adj.get(i, []):
                if j not in shift:
                    shift[j] = shift[i] + s
                    label[j] = nmol
                    stack.append(j)
        nmol += 1
    for i, sh in shift.items():
        unwrapped[i] = pos[i] + sh @ cell
    heavy = np.array([s not in ("H", "D") for s in atoms.get_chemical_symbols()])
    return cell, unwrapped, heavy, label


def perp_widths(cell):
    vol = abs(np.linalg.det(cell))
    return np.array(
        [
            vol / np.linalg.norm(np.cross(cell[(i + 1) % 3], cell[(i + 2) % 3]))
            for i in range(3)
        ]
    )


def build_reference(cell, pos, heavy, label, radius):
    """List (one entry per reference molecule) of (M, 3) vectors from the heavy-atom centroid to
    the heavy atoms of other molecules within `radius`."""
    nmol = label.max() + 1
    cent = np.array([pos[(label == m) & heavy].mean(axis=0) for m in range(nmol)])
    n = np.ceil(radius / perp_widths(cell)).astype(int) + 1
    shifts = np.array(list(itertools.product(*[range(-k, k + 1) for k in n])))
    hp, hl = pos[heavy], label[heavy]
    refs = []
    for m in range(nmol):
        pts, own = [], []
        for sh in shifts:
            p = hp + sh @ cell
            d = np.linalg.norm(p - cent[m], axis=1)
            keep = d <= radius
            is_self = (hl == m) & (not sh.any())  # the central molecule itself
            keep &= ~is_self
            pts.append(p[keep] - cent[m])
        refs.append(np.concatenate(pts))
    return refs


# ----------------------------------------------------------------------------
# scoring of simulation frames
# ----------------------------------------------------------------------------
def frame_molecules(data, cols, cell, origin, heavy_types):
    """Molecule ids, heavy-atom centroids (wrapped into the box), fractional z, atom tree."""
    idx = {c: i for i, c in enumerate(cols)}
    mol = data[:, idx["mol"]].astype(int)
    typ = data[:, idx["type"]].astype(int)
    pos = data[:, [idx["x"], idx["y"], idx["z"]]]
    if "ix" in idx:
        unwrapped = pos + data[:, [idx["ix"], idx["iy"], idx["iz"]]] @ cell
    else:
        unwrapped = pos
    heavy = np.isin(typ, list(heavy_types))
    inv = np.linalg.inv(cell)

    # atoms for the tree: wrapped heavy atoms plus the 27 periodic images
    fa = (pos[heavy] - origin) @ inv
    pw = origin + (fa - np.floor(fa)) @ cell
    shifts = np.array(list(itertools.product((-1, 0, 1), repeat=3))) @ cell
    tree = cKDTree((pw[None, :, :] + shifts[:, None, :]).reshape(-1, 3))

    # centroids of whole molecules (heavy atoms)
    uniq, inverse = np.unique(mol[heavy], return_inverse=True)
    cs = np.zeros((len(uniq), 3))
    cnt = np.zeros(len(uniq))
    np.add.at(cs, inverse, unwrapped[heavy])
    np.add.at(cnt, inverse, 1.0)
    cent = cs / cnt[:, None]
    f = (cent - origin) @ inv
    f -= np.floor(f)
    return uniq, origin + f @ cell, f[:, 2], tree


def mol_scores(cents, tree, refs, scale, tol, soft_sigma=None):
    """Best (over reference molecules) fraction of reference atoms matched within tol.
    soft_sigma: smooth variant (gaussian weight of the distance), used for fitting the lattice scale.
    """
    best = np.zeros(len(cents))
    bound = tol if soft_sigma is None else 3.0 * soft_sigma
    for v in refs:
        q = cents[:, None, :] + (v * scale)[None, :, :]
        d, _ = tree.query(q.reshape(-1, 3), distance_upper_bound=bound, workers=-1)
        if soft_sigma is None:
            w = np.isfinite(d).astype(float)
        else:
            w = np.where(
                np.isfinite(d),
                np.exp(-((np.where(np.isfinite(d), d, 0.0) / soft_sigma) ** 2)),
                0.0,
            )
        best = np.maximum(best, w.reshape(len(cents), -1).mean(axis=1))
    return best


def robust_mean(x):
    x = np.sort(x)
    return x[int(0.3 * len(x)) :].mean()


def fit_scale(score_fn):
    """Axis scales of the reference lattice (thermal expansion), coordinate-wise grid search."""
    s = np.ones(3)
    for lo, hi, n in ((0.92, 1.12, 21), (-0.015, 0.015, 13)):
        for ax in range(3):
            grid = np.linspace(lo, hi, n) if lo > 0 else s[ax] + np.linspace(lo, hi, n)
            vals = []
            for g in grid:
                t = s.copy()
                t[ax] = g
                vals.append(score_fn(t))
            s[ax] = grid[int(np.argmax(vals))]
    return s


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def analyze(args):
    masses = read_masses(args.data)
    heavy_types = {t for t, m in masses.items() if m > 1.5}
    cell_r, pos_r, heavy_r, label_r = reference_from_cif(args.cif)
    refs = build_reference(cell_r, pos_r, heavy_r, label_r, args.radius)
    print(
        f"Reference: {len(refs)} molecules in the cell, "
        f"{', '.join(str(len(r)) for r in refs)} neighbour atoms within {args.radius} A"
    )

    scale0, lx0, ly0, thr = np.ones(3), None, None, args.threshold
    rows_long, rows_sum, frames = [], [], []
    first = True
    for step, cell, origin, cols, data in iter_dump(
        args.traj, args.stride, args.step_every
    ):
        mols, cents, zf, tree = frame_molecules(data, cols, cell, origin, heavy_types)
        if first:
            lx0, ly0 = cell[0, 0], cell[1, 1]
            if args.n_mol_cryst:
                cr = np.where(mols <= args.n_mol_cryst)[0]
                sub = cr[np.linspace(0, len(cr) - 1, min(100, len(cr))).astype(int)]
                scale0 = fit_scale(
                    lambda s: robust_mean(
                        mol_scores(
                            cents[sub],
                            tree,
                            refs,
                            s,
                            args.tol,
                            soft_sigma=0.5 * args.tol,
                        )
                    )
                )
                print(
                    f"Fitted lattice scales (x, y, z) on frame 0: {np.round(scale0, 3)}"
                )
                sc = mol_scores(cents, tree, refs, scale0, args.tol)
                c, l = sc[mols <= args.n_mol_cryst], sc[mols > args.n_mol_cryst]
                if args.threshold is None:
                    thr = (
                        0.5 * (np.percentile(c, 25) + np.percentile(l, 95))
                        if np.percentile(c, 25) > np.percentile(l, 95)
                        else 0.5 * (np.median(c) + np.median(l))
                    )
                print(
                    f"Frame 0 scores: crystal median {np.median(c):.2f} (p25 {np.percentile(c, 25):.2f}), "
                    f"liquid median {np.median(l):.2f} (p95 {np.percentile(l, 95):.2f}); threshold {thr:.2f}"
                )
            elif thr is None:
                thr = 0.5
                print("No --n-mol-cryst: scales = 1, threshold 0.5")
            first = False
        scale = scale0 * np.array([cell[0, 0] / lx0, cell[1, 1] / ly0, 1.0])
        score = mol_scores(cents, tree, refs, scale, args.tol)
        cryst = (score >= thr).astype(float)

        # optional centring of the crystal slab (circular mean of z over crystalline molecules)
        zc = zf.copy()
        if args.center and cryst.sum() > 0:
            ang = np.angle(np.sum(cryst * np.exp(2j * np.pi * zf)))
            zc = (zf - ang / (2 * np.pi) + 0.5) % 1.0
        lz = cell[2, 2]
        b = np.minimum((zc * args.nbins).astype(int), args.nbins - 1)
        n = np.bincount(b, minlength=args.nbins).astype(float)
        fb = np.bincount(b, weights=cryst, minlength=args.nbins) / np.maximum(n, 1)
        fs = np.bincount(b, weights=score, minlength=args.nbins) / np.maximum(n, 1)
        for k in range(args.nbins):
            rows_long.append(
                (
                    step,
                    lz,
                    k,
                    (k + 0.5) / args.nbins,
                    (k + 0.5) / args.nbins * lz,
                    n[k],
                    fb[k],
                    fs[k],
                )
            )
        rows_sum.append((step, lz, cryst.mean()))
        frames.append((step, fb))
        print(f"step {step}: crystalline fraction {cryst.mean():.3f}, lz {lz:.2f} A")

    with open(f"{args.out}_fz.csv", "w") as f:
        f.write("step,lz,bin,z_frac,z_A,n_mol,f_binary,f_score\n")
        for r in rows_long:
            f.write("{},{:.4f},{},{:.4f},{:.3f},{:.0f},{:.4f},{:.4f}\n".format(*r))
    with open(f"{args.out}_summary.csv", "w") as f:
        f.write("step,lz,crystal_fraction\n")
        for r in rows_sum:
            f.write("{},{:.4f},{:.4f}\n".format(*r))
    print(f"Wrote {args.out}_fz.csv and {args.out}_summary.csv (threshold {thr:.2f})")

    if args.plot and frames:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps = [s for s, _ in frames]
        M = np.array([fb for _, fb in frames]).T
        fig, ax = plt.subplots(figsize=(7, 4))
        im = ax.imshow(
            M,
            origin="lower",
            aspect="auto",
            vmin=0,
            vmax=1,
            cmap="viridis",
            extent=[steps[0], steps[-1] if len(steps) > 1 else steps[0] + 1, 0, 1],
        )
        ax.set_xlabel("step")
        ax.set_ylabel("z / lz")
        fig.colorbar(im, label="f(z): 1 crystal, 0 liquid")
        fig.tight_layout()
        fig.savefig(f"{args.out}_fz.png", dpi=150)
        print(f"Wrote {args.out}_fz.png")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "traj", help="LAMMPS dump (.lammpstrj): id mol type x y z [ix iy iz]"
    )
    p.add_argument(
        "--cif", required=True, help="CIF of the crystal (reference environment)"
    )
    p.add_argument(
        "--data",
        required=True,
        help="LAMMPS data file (Masses: which types are hydrogen)",
    )
    p.add_argument(
        "--n-mol-cryst",
        type=int,
        default=0,
        help="molecules with mol id <= N belong to the initial crystal (lattice-strain fit, threshold)",
    )
    p.add_argument(
        "--stride", type=int, default=1, help="analyse every K-th dump frame"
    )
    p.add_argument(
        "--step-every",
        type=int,
        default=0,
        help="analyse only frames with step %% N == 0",
    )
    p.add_argument("--radius", type=float, default=8.0, help="environment radius, A")
    p.add_argument("--tol", type=float, default=1.0, help="matching tolerance, A")
    p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="crystal if score >= threshold (default: auto)",
    )
    p.add_argument("--nbins", type=int, default=40, help="bins along z")
    p.add_argument(
        "--no-center",
        dest="center",
        action="store_false",
        help="do not centre the crystalline slab in z",
    )
    p.add_argument("--out", default="crystal", help="output prefix")
    p.add_argument(
        "--plot",
        action="store_true",
        help="save f(z, step) heat map (needs matplotlib)",
    )
    analyze(p.parse_args())


if __name__ == "__main__":
    main()
