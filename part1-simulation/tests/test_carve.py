"""The one carve path (carve.py): the same edits give the identical field every time and
through every caller (stress_cars.carve_field), and a cut keeps the rule zones. 2 mm block."""
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

LAYOUT = {"W": 120.3, "x_front": 46.0, "d_halo": 43.72}
EDITS = [{"bias_mm": -1.0, "bumps": [{"u": [0.4, 0.8, 0.6], "amp_mm": 3.0, "sigma_mm": 10.0},
                                     {"c_mm": [120.0, 10.0, 20.0], "amp_mm": -2.0, "sigma_mm": 8.0}]}] * 2


def _sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def test_same_edits_same_field(tmp_path):
    import carve
    g1, r1 = carve.carve(LAYOUT, EDITS, spacing_mm=2.0)
    g2, _ = carve.carve(LAYOUT, EDITS, spacing_mm=2.0)
    assert [r["status"] for r in r1] == ["ok", "ok"]
    assert _sha(g1.phi.grid) == _sha(g2.phi.grid)
    solid = g1.phi.grid < 0
    assert not (solid & g1.phi.hard_mask_air).any() and not (~solid & g1.phi.hard_mask_solid).any()
    carve.save(LAYOUT, EDITS, tmp_path / "f.npy", spacing_mm=2.0)
    assert _sha(np.load(tmp_path / "f.npy")) == _sha(g1.phi.grid.astype(np.float32))


def test_stress_cars_goes_through_carve(tmp_path, monkeypatch):
    """stress_cars.carve_field is carve.save with the stress recipe: same field, byte for byte."""
    import carve
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "part5-simulation" / "checks"))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "part5-simulation"))
    import stress_cars as sc
    c = {"layout": dict(LAYOUT), "carve": {"steps": 1, "bias_mm": -0.5,
                                           "blobs": [{"u": [0.5, 0.7, 0.5], "amp_mm": 2.0, "sigma_mm": 9.0}]}}
    real = carve.save
    monkeypatch.setattr(carve, "save", lambda lay, ed, p: real(lay, ed, p, spacing_mm=2.0))
    sc.carve_field(c, tmp_path / "a.npy")
    real(LAYOUT, sc.carve_edits(c), tmp_path / "b.npy", spacing_mm=2.0)
    assert _sha(np.load(tmp_path / "a.npy")) == _sha(np.load(tmp_path / "b.npy"))
