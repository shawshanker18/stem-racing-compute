"""Measured map on a real carved car, and its test on shapes outside its modes (BUILD_SPEC 7
step 4, 12.4, 12.7).

Every car here is Part 1's start block (cutter.start_block) at one fixed skin, built and timed by
Part 5's run_car exactly as Part 2's reshape study does, and timed on the BASE car's mesh
reshaped onto it (morph_from) from the base car's flow (warm_from): the set-up of Part 2's lowest
timing noise. So each car differs from the base only by its field:
  * mode k, +/-: the base field minus/plus a Gaussian of height A_MM and radius R_MM (mirrored
    in y) centred on the block surface: the surface grows/shrinks by about A_MM there;
  * cut j: one Part 1 cutter.cut step on the block with a random smooth dT_dn (1 to 3 Gaussian
    patches of random sign, 6 to 20 mm), its 90th-percentile move capped at CUT_DEPTHS mm.
The real normal move of every car, b (m, + = grown), is measured at the base car's own
body_half.stl vertices as the signed distance to that car's body_half.stl, so rule zones, milling
and anything else the builder holds still are in b as built, not as asked.

Map: Phi_k = (b_k+ - b_k-)/2 and c_k = (T_k+ - T_k-)/2 (race time at the mass target, s);
m = Phi^T (Phi diag(A) Phi^T)^-1 c (12.7), so sum(m b A) predicts any small shape's dT.
Test: each cut's measured T - T_base against its prediction, scored only on cuts whose change is
>= 3 x its noise (sigma from Part 2, x sqrt 2 for a difference), with BUILD_SPEC 12.4's marks:
sign right >= 80 %, Pearson r >= 0.8, >= 25 qualifying, sign not fitted. Drag and lift are
reported the same way for information.

    python car_modes.py plan              <out>             prints the job matrix (JSON)
    python car_modes.py run   <ref_dir|-> <out> <items...>  items: base | m<k>+ | m<k>- | c<j>
    python car_modes.py judge <dir with results>
(run/plan from part5-simulation; GitHub only: p3_car_modes.yml.)
Env: P3_NMODES 60, P3_MODE_R_MM 10, P3_MODE_A_MM 1, P3_NCUTS 40, P3_CUT_DEPTHS "1 1.5",
P3_SKIN_MM 3.95, P3_CFD_ITERS 1000, P3_SIGMA_MS 0.36, P3_CHUNK 8.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

LAYOUT = {"W": 120.3, "x_front": 46.0, "d_halo": 43.72}      # run_car's defaults, as Part 2's cut study
NMODES = int(os.environ.get("P3_NMODES", "60"))
R_M = float(os.environ.get("P3_MODE_R_MM", "10")) / 1e3
A_M = float(os.environ.get("P3_MODE_A_MM", "1")) / 1e3
NCUTS = int(os.environ.get("P3_NCUTS", "40"))
DEPTHS = [float(t) for t in os.environ.get("P3_CUT_DEPTHS", "1 1.5").split()]
SKIN = os.environ.get("P3_SKIN_MM", "3.95")
ITERS = os.environ.get("P3_CFD_ITERS", "1000")
RES = os.environ.get("P3_RES", "")       # required, no default mesh (user 2026-11-13: high mesh)
LOOP_MESHES = ("fine", "resolved")       # a carving-loop timing (P5_LOOP=1) may only use these


def mesh() -> str:
    """The mesh for this timing: P3_RES must be set, and a loop timing must be fine or resolved."""
    if not RES:
        raise SystemExit("P3_RES is not set: choose the mesh explicitly")
    if os.environ.get("P5_LOOP") == "1" and RES not in LOOP_MESHES:
        raise SystemExit(f"carving-loop timing on {RES!r}: the loop runs on fine or resolved only")
    return RES
SIGMA_S = float(os.environ.get("P3_SIGMA_MS", "0.36")) / 1e3
CHUNK = int(os.environ.get("P3_CHUNK", "8"))
SPACING_MM = float(os.environ.get("P3_SPACING_MM", "1.0"))   # block grid; tests use 2 mm


def items():
    return [f"m{k}{s}" for k in range(NMODES) for s in "+-"] + [f"c{j}" for j in range(NCUTS)]


def block():
    import cutter
    return cutter.start_block(LAYOUT, spacing_mm=SPACING_MM)


def surface(g):
    """Block half surface: vertices and outward unit normals."""
    import unified_phi as up
    h = up.extract_half_surface(g)
    V = np.asarray(h.vertices, float)
    N = np.asarray(h.vertex_normals, float)
    if h.volume < 0:
        N = -N
    return h, V, N


def centres(V, N):
    """NMODES centres spread over the body (farthest point from the nose tip), off the closing face."""
    P = V[N[:, 1] > -0.9]
    pick = [int(np.argmin(P[:, 0]))]
    d = np.linalg.norm(P - P[pick[0]], axis=1)
    while len(pick) < NMODES:
        pick.append(int(np.argmax(d)))
        d = np.minimum(d, np.linalg.norm(P - P[pick[-1]], axis=1))
    return P[pick]


def grid_xyz(g):
    r = g.region
    from geometry_contract import GRID_SPACING_M as dx     # set by start_block (1 mm)
    ax = [r.origin_m[i] + np.arange(r.shape[i]) * dx for i in range(3)]
    return np.meshgrid(*ax, indexing="ij")


def field(item: str) -> np.ndarray:
    """The carved field (float32 metres, Part 1's save_field format) for one item."""
    import carve                                          # Part 1's one carve path: rule zones and milling kept
    g = block()
    if item == "base":
        return np.asarray(g.phi.grid, np.float32)
    h, V, N = surface(g)
    C = centres(V, N) * 1e3                               # mm
    a_mm, r_mm = A_M * 1e3, R_M * 1e3
    if item[0] == "m":                                    # mode k: grow (+) or thin (-) by A at its centre
        k, s = int(item[1:-1]), (1.0 if item[-1] == "+" else -1.0)
        edit = {"bumps": [{"c_mm": list(C[k]), "amp_mm": s * a_mm, "sigma_mm": r_mm}], "unit": False}
        label = item
    elif item[0] == "s":                                  # a loop step: sum a_k * mode k (P5_STEPS file)
        a = json.loads(Path(os.environ["P5_STEPS"]).read_text())[item]
        edit = {"bumps": [{"c_mm": list(C[int(k)]), "amp_mm": ak * a_mm, "sigma_mm": r_mm} for k, ak in a],
                "unit": False}
        label = item
    else:                                                 # random test cut j: 1-3 patches, depth-capped
        j = int(item[1:])
        rng = np.random.default_rng(42000 + j)
        ok = np.where(N[:, 1] > -0.9)[0]
        bumps = []
        for _ in range(int(rng.integers(1, 4))):
            c = V[rng.choice(ok)] * 1e3
            sig = rng.uniform(6.0, 20.0)
            bumps.append({"c_mm": list(c), "amp_mm": rng.choice([-1.0, 1.0]), "sigma_mm": sig})
        depth = DEPTHS[j % len(DEPTHS)]
        edit = {"bumps": bumps, "unit": True, "trust_mm": depth}
        label = f"cut {j}: depth cap {depth:g} mm,"
    g2, reps = carve.carve(LAYOUT, [edit], spacing_mm=SPACING_MM)
    rep = reps[0]
    print(f"LAB {label} status {rep.get('status')} {rep.get('reason', '')},"
          f" move p90 {1e3 * (rep.get('disp_p90_m') or 0):.3f} mm", flush=True)
    if rep.get("status") != "ok":
        raise RuntimeError(f"{item} refused: {rep.get('reason')}")
    return np.asarray(g2.phi.grid, np.float32)


def run_car(item: str, ref: str | None, out: Path) -> Path:
    o = out / item
    shutil.rmtree(o, ignore_errors=True)
    o.mkdir(parents=True)
    np.save(o / "field.npy", field(item))
    opt = {"layer_overrides": {"nSurfaceLayers": 0}}
    if ref:
        opt.update(morph_from=ref, warm_from=ref)
    runs = Path("../part2-simulation/cfd_case_template/runs")
    shutil.rmtree(runs, ignore_errors=True)
    cmd = [sys.executable, "-W", "ignore", "run_car.py", "--out", str(o), "--skin-mm", SKIN, "--support", "beam",
           "--phi-npy", str(o / "field.npy"), "--cfd", "--cfd-if-legal", "--cfd-iters", ITERS, "--res", mesh(), "--np", "2",
           "--keep-runs", "--cfd-json", json.dumps(opt)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        marker = os.environ.get("P2_CHECKPOINT_FILE")
        if marker and Path(marker).exists():          # solve hit its time budget; case saved (CHECKPOINTS)
            print(f"LAB {item}: solve saved at its time budget; the next link continues it", flush=True)
            raise SystemExit(3)
        print(p.stdout[-3000:], p.stderr[-3000:])
        raise RuntimeError(f"run_car failed on {item}")
    (o / "field.npy").unlink()
    S = json.loads((o / "summary.json").read_text())
    failed = sorted(k for k, v in S["legality"]["checks"].items() if not v["pass"])
    if failed:                                            # the referee's check: an illegal car is never timed
        rec = {"item": item, "legal": False, "failed": failed, "T_s": None}
        (o / "result.json").write_text(json.dumps(rec))
        shutil.rmtree(runs, ignore_errors=True)
        print(f"LAB {item}: ILLEGAL ({', '.join(failed)}), not timed", flush=True)
        return o
    resumed = os.environ.get("P2_RESUME_CASE")
    run = Path(resumed) if resumed else next(runs.iterdir())
    if item == "base":
        dst = out / "refmesh"
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(run, dst, ignore=shutil.ignore_patterns("processor*", "logs", "postProcessing"))
    c = S["cfd_initial"]
    rec = {"item": item, "legal": True, "T_s": c["T_at_target_s"], "D_N": c["D20_N"], "L_N": c["L_N"],
           "converged": c["converged"], "morph_fallback": (run / "MORPH_FALLBACK").exists(),
           "parts_D_N": {k: v["D_N"] for k, v in (c.get("parts") or {}).items()}}
    (o / "result.json").write_text(json.dumps(rec, default=float))
    shutil.rmtree(runs, ignore_errors=True)
    print(f"LAB {item}: T {rec['T_s'] * 1e3:.3f} ms, drag {rec['D_N']:.5f} N, lift {rec['L_N']:.5f} N,"
          f" converged {rec['converged']}, reshape fell back {rec['morph_fallback']}", flush=True)
    return o


def measure_b(base_stl: Path, o: Path):
    """b at the base body's vertices: signed distance to this car's body, + = this car is bigger there."""
    import trimesh
    base = trimesh.load(str(base_stl), force="mesh")
    car = trimesh.load(str(o / "body_half.stl"), force="mesh")
    b = trimesh.proximity.ProximityQuery(car).signed_distance(base.vertices)    # + inside the car
    np.save(o / "b.npy", b.astype(np.float32))


def run(ref: str | None, out: Path, its):
    out.mkdir(parents=True, exist_ok=True)
    for it in its:
        o = run_car(it, None if it == "base" else ref, out)
        if it == "base":
            import trimesh
            m = trimesh.load(str(o / "body_half.stl"), force="mesh")
            A = np.zeros(len(m.vertices))
            np.add.at(A, m.faces.ravel(), np.repeat(m.area_faces / 3.0, 3))
            np.savez(out / "base_surface.npz", V=m.vertices, A=A)
            shutil.copy(o / "body_half.stl", out / "base_body_half.stl")
        else:
            measure_b(Path(ref).parent / "base_body_half.stl", o)
        for f in o.iterdir():                    # keep only what the judge reads
            if f.name not in ("result.json", "b.npy", "summary.json"):
                shutil.rmtree(f) if f.is_dir() else f.unlink()


def judge(d: Path):
    d = Path(d)
    R = {json.loads(p.read_text())["item"]: (json.loads(p.read_text()), p.parent) for p in d.rglob("result.json")
         if json.loads(p.read_text()).get("legal", True)}           # illegal items are never in a map
    A = np.load(next(d.rglob("base_surface.npz")))["A"]
    base = R["base"][0]
    bad = sorted(k for k, (r, _) in R.items() if not r["converged"] or r["morph_fallback"])
    print(f"LAB base: T {base['T_s'] * 1e3:.3f} ms, drag {base['D_N']:.5f} N, lift {base['L_N']:.5f} N; "
          f"{len(R) - 1} other cars; not converged or fell back to a fresh mesh (left out): {bad}")
    use = {k: v for k, v in R.items() if k not in bad}
    b = {k: np.load(p / "b.npy").astype(float) for k, (_, p) in use.items() if k != "base"}
    modes = [k for k in range(NMODES) if f"m{k}+" in b and f"m{k}-" in b]
    missing = [k for k in range(NMODES) if k not in modes]
    print(f"LAB modes usable {len(modes)} of {NMODES}; missing {missing}")
    if not modes:
        raise SystemExit("no usable modes: nothing to judge")
    Phi = np.array([(b[f"m{k}+"] - b[f"m{k}-"]) / 2 for k in modes])
    amp = np.abs(Phi).max(axis=1)
    print(f"LAB mode peak move (mm) p5 p50 p95: {np.round(np.percentile(amp, [5, 50, 95]) * 1e3, 3).tolist()}"
          f" (asked {A_M * 1e3:g}); +/- asymmetry median"
          f" {np.median([np.linalg.norm(b[f'm{k}+'] + b[f'm{k}-']) / np.linalg.norm(b[f'm{k}+'] - b[f'm{k}-']) for k in modes]):.2f}")
    G = (Phi * A) @ Phi.T
    cuts = sorted((k for k in b if k[0] == "c"), key=lambda s: int(s[1:]))
    for q, key, unit, sig in (("race time", "T_s", 1e3, SIGMA_S), ("drag", "D_N", 1e3, None), ("lift", "L_N", 1e3, None)):
        c = np.array([(use[f"m{k}+"][0][key] - use[f"m{k}-"][0][key]) / 2 for k in modes])
        m = Phi.T @ np.linalg.lstsq(G, c, rcond=1e-10)[0]
        pred = np.array([np.sum(m * b[j] * A) for j in cuts])
        meas = np.array([use[j][0][key] - base[key] for j in cuts])
        if sig is None:
            r = np.corrcoef(pred, meas)[0, 1] if len(cuts) > 2 else np.nan
            print(f"LAB {q} (information): {len(cuts)} cuts, r {r:+.3f}, sign right {np.mean(np.sign(pred) == np.sign(meas)):.0%},"
                  f" slope {np.polyfit(pred, meas, 1)[0]:.2f}")
            continue
        se = sig * np.sqrt(2)
        qual = np.abs(meas) >= 3 * se
        for j, p_, m_, ok in zip(cuts, pred, meas, qual):
            print(f"LAB cut {j}: {q} measured {m_ * unit:+.3f} ms, predicted {p_ * unit:+.3f} ms"
                  f"{'' if ok else '  (below 3 x noise, not scored)'}")
        n = int(qual.sum())
        r = np.corrcoef(pred[qual], meas[qual])[0, 1] if n > 2 else np.nan
        sgn = float(np.mean(np.sign(pred[qual]) == np.sign(meas[qual]))) if n else np.nan
        ok = n >= 25 and sgn >= 0.8 and r >= 0.8
        print(f"LAB {q}: {n} of {len(cuts)} cuts clear 3 x noise ({3 * se * 1e3:.2f} ms); sign right {sgn:.0%}"
              f" (mark 80 %), r {r:+.3f} (mark 0.8), slope {np.polyfit(pred[qual], meas[qual], 1)[0] if n > 1 else np.nan:.2f};"
              f" {'PASS' if ok else 'FAIL'}" + ("" if n >= 25 else " (fewer than 25 qualify)"))
        np.save(d / "map_dT_dn.npy", m)


if __name__ == "__main__":
    sys.path[:0] = [".", str(Path(__file__).resolve().parent)]
    cmd = sys.argv[1]
    if cmd == "run":
        import pattern, solver                          # noqa: E402,F401  (these put the other parts on the path)
    if cmd == "plan":
        its = items()
        print(json.dumps([" ".join(its[i:i + CHUNK]) for i in range(0, len(its), CHUNK)]))
    elif cmd == "run":
        run(None if sys.argv[2] == "-" else str(Path(sys.argv[2]).resolve()), Path(sys.argv[3]), sys.argv[4:])
    elif cmd == "judge":
        judge(Path(sys.argv[2]))
    else:
        raise SystemExit(__doc__)
