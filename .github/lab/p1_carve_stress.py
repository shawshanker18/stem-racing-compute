"""p1_carve_stress: Part 1's carve stress test (BUILD_SPEC Step 3), carving through the cutter.

    python ../.github/lab/p1_carve_stress.py <seed> <n_cars>    (from part5-simulation, on GitHub)

Uses Part 5's random-car tester (checks/stress_cars.py) as a library, unchanged; since
2026-11-14 its carving is Part 1's one carve path (carve.py), so this lab builds the very bodies
the batches judge. Below are only diagnostic switches.
Prints one "LAB {json}" line per car (status, rules broken, cut verdicts, error).
"""
import json
import sys
from pathlib import Path

ROOT = Path.cwd().parent
for p in ("part1-simulation", "part2-simulation", "part3-simulation", "part4-simulation"):
    sys.path.insert(0, str(ROOT / p))
sys.path.insert(0, str(Path.cwd() / "checks"))
sys.path.insert(0, str(Path.cwd()))
import numpy as np  # noqa: E402

import stress_cars as sc  # noqa: E402


if "span" in (sys.argv[3] if len(sys.argv) > 3 else ""):   # span: what cuts the front wing's span, and where
    import legality as _lg
    import trimesh as _tm
    _sb = _lg._span_blocker_y
    def _span(elements, others):
        y = _sb(elements, others)
        for e in elements:
            h = e.convex_hull
            print("LAB span wing", np.round(e.bounds, 2).tolist(), flush=True)
            for i, m in enumerate(others):
                if not m.is_volume:
                    print("LAB span other", i, "not a volume", np.round(m.bounds, 1).tolist(), flush=True)
                    continue
                x = _tm.boolean.intersection([m, h], engine="manifold")
                if x is not None and not x.is_empty and abs(x.volume) > 1e-3:
                    print("LAB span hit", i, round(abs(x.volume), 2), np.round(x.bounds, 2).tolist(),
                          "other", np.round(m.bounds, 1).tolist(), flush=True)
        print("LAB span cut_y", y, flush=True)
        return y
    _lg._span_blocker_y = _span
if "cutwarn" in (sys.argv[3] if len(sys.argv) > 3 else ""):   # cutwarn: say when cut_boxes leaves a mesh uncut
    import exact_zones as _ez
    _cb = _ez.cut_boxes
    def _cut_boxes(g, mesh):
        out = _cb(g, mesh)
        print("LAB cutbox", "UNCUT" if out is mesh else "cut", len(mesh.faces), flush=True)
        return out
    _ez.cut_boxes = _cut_boxes
if "where" in (sys.argv[3] if len(sys.argv) > 3 else ""):   # where: the slivers of each sliver-illegal car, file by file
    import tempfile as _tf
    import trimesh as _tm
    _mk, _dirs = _tf.mkdtemp, []
    _tf.mkdtemp = lambda *a, **k: (_dirs.append(_mk(*a, **k)) or _dirs[-1])
    _b1 = sc.build_one

    def _where(c):
        row = _b1(c)
        if "mesh_slivers_body" in row.get("failed", []):
            d = Path(_dirs[-1])
            for p in sorted(d.rglob("*.stl")):
                m = _tm.load(p, force="mesh")
                if len(m.faces) == 0:
                    continue
                s = np.degrees(m.face_angles.min(axis=1)) < 5.0
                cen = m.triangles_center[s] * (1e3 if m.bounds[1][0] < 1.0 else 1.0)
                bins = {}
                for q in np.round(cen / 5.0) * 5.0:
                    k = tuple(q.tolist())
                    bins[k] = bins.get(k, 0) + 1
                top = sorted(bins.items(), key=lambda t: -t[1])[:6]
                print("LAB where", json.dumps({"file": str(p.relative_to(d)), "faces": len(m.faces),
                                               "sliver_pct": round(100 * float(s.mean()), 2), "top5mm": top}), flush=True)
        return row
    sc.build_one = _where
if "hash" in (sys.argv[3] if len(sys.argv) > 3 else ""):   # hash: a hash of every mesh each car writes (same-body proof)
    import hashlib as _hl
    import tempfile as _tf
    import trimesh as _tm
    import shutil as _sh
    _rm, _n, _seed = _sh.rmtree, [0], int(sys.argv[1])

    def _rm_hash(d, *a, **k):          # build_one deletes its folder at the end: hash it first
        _n[0] += 1
        for p in sorted(Path(d).rglob("*")):
            if p.suffix == ".stl":
                m = _tm.load(p, force="mesh")
                h = _hl.sha256(np.round(np.asarray(m.vertices), 9).tobytes() + np.asarray(m.faces).tobytes()).hexdigest()[:16]
            elif p.suffix == ".npy":
                h = _hl.sha256(np.load(p).tobytes()).hexdigest()[:16]
            else:
                continue
            print("LAB hash", json.dumps({"seed": _seed, "car": _n[0] - 1, "file": str(p.relative_to(d)), "sha": h}), flush=True)
        return _rm(d, *a, **k)
    sc.shutil.rmtree = _rm_hash
seed, n = int(sys.argv[1]), int(sys.argv[2])
MODE = sys.argv[3] if len(sys.argv) > 3 else ""     # nocorner | nobars: switch off one 11-Nov cutter change
if "nocorner" in MODE:                                # corner relief for every body (6b95197) off
    import machining
    _ra = machining.reachable_air
    machining.reachable_air = lambda A, r, any_side=False: np.ones(A.shape, bool) if any_side else _ra(A, r)
if "nobars" in MODE:                                  # loose pieces joined (e63d51e) off: dropped again
    import param_body
    param_body.BRIDGE_MIN_MM3 = 1e18
out = Path("../stress_p1")
sys.argv = ["stress_cars.py", "--seed", str(seed), "--n", str(n), "--out", str(out), "--carve"]
sc.main()
for p in sorted(out.glob(f"car_{seed}_*.json"), key=lambda p: int(p.stem.split("_")[-1])):
    r = json.loads(p.read_text())
    print("LAB " + json.dumps({"car": p.stem, "status": r["status"], "failed": r.get("failed"),
                               "notes": r.get("notes"), "error": r.get("error"), "where": r.get("where"),
                               "cuts": r.get("layout", {}).get("cuts"), "hard": r.get("layout", {}).get("hard"),
                               "seconds": r.get("seconds"), "peak_gb": r.get("peak_gb")}), flush=True)
