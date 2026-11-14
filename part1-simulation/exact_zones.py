"""Exact signed distances to the forced-air rule zones (team lead, 2026-11-09).

The rule masks are yes/no per cell, so a zone's face snaps to a cell boundary and moves
by up to half a cell between grids: the same carved body weighed 0.5 to 2 % differently
at 1.0, 0.75 and 0.5 mm, against a 0.5 % bar. Here every zone that bounds the body is a
distance field in mm, positive inside the zone, written from the same numbers as its mask
builder (named on each function). The finished body takes phi = max(phi, zone + margin),
so its surface lands on the zone's true face, MARGIN_MM clear of it, at any grid.

Slanted faces (the T7.9 wedges) are scaled to true distance; box edges and corners use
the exact box distance. tests/test_exact_zones.py checks every field against its mask.
"""
from __future__ import annotations

import numpy as np

MARGIN_MM = 0.05     # clear of every zone by the same amount at any grid (referee's zone checks)


def _box(X, Y, Z, x=(None, None), y=(None, None), z=(None, None)):
    """Signed distance to an axis box, positive inside. None = unbounded on that side."""
    outs = []
    for c, (lo, hi) in ((X, x), (Y, y), (Z, z)):
        d = np.full(np.broadcast(X, Y, Z).shape, -np.inf)
        if lo is not None:
            d = np.maximum(d, lo - c)
        if hi is not None:
            d = np.maximum(d, c - hi)
        outs.append(d)
    m = np.maximum(np.maximum(outs[0], outs[1]), outs[2])
    out = np.sqrt(sum(np.maximum(o, 0.0) ** 2 for o in outs))
    return -(out + np.minimum(m, 0.0))


def _extend_abutting(specs):
    """Box specs ((x lo, hi), (|y| lo, hi), (z lo, hi)), None = open, with every box that
    abuts a neighbour whose cross-section holds its own run on through that neighbour: the
    union is unchanged, but the shared face is gone. On such a face max() reads 0 for points
    inside the union, and with the margin they counted as inside the zone (the nose and main
    body boxes at Ref A, 2026-11-10)."""
    def lo_(v):
        return -np.inf if v is None else v

    def hi_(v):
        return np.inf if v is None else v

    specs = [list(map(list, s)) for s in specs]
    for _ in range(2):
        for i, bi in enumerate(specs):
            for j, bj in enumerate(specs):
                if i == j:
                    continue
                for a in range(3):
                    if not all(lo_(bj[k][0]) <= lo_(bi[k][0]) and hi_(bi[k][1]) <= hi_(bj[k][1])
                               for k in range(3) if k != a):
                        continue
                    if bi[a][1] is not None and bj[a][0] is not None and abs(bi[a][1] - bj[a][0]) < 1e-9:
                        bi[a][1] = bj[a][1]
                    if bi[a][0] is not None and bj[a][1] is not None and abs(bi[a][0] - bj[a][1]) < 1e-9:
                        bi[a][0] = bj[a][0]
    return [tuple(map(tuple, s)) for s in specs]


def envelope_air(X, aY, Z, bv, re, d_mm=1.0):  # d_mm None: points, max() (exact outside)
    """Outside the legal envelope (unified_phi._zone_masks): the union of the nose, sidepod,
    rearpod and main-body boxes; positive outside all of them."""
    mm = 1e3
    specs = [
        ((None, bv.ref_plane_A_m * mm), (None, re.y_nose_half_m * mm), (re.z_floor_m * mm, re.z_nose_top_m * mm)),
        ((bv.sidepod_x_min_m * mm, bv.sidepod_x_max_m * mm), (re.y_sidepod_inner_m * mm, re.y_sidepod_outer_m * mm),
         (re.z_floor_m * mm, re.z_sidepod_top_m * mm)),
        # rear face at Ref B + 40 exactly (T9.4.2): the pod's whole cells ran 0.08 mm past it (Part 4, 2026-11-11)
        ((bv.rearpod.origin_m[0] * mm, min(bv.rearpod.x_max_m(), bv.ref_plane_B_m + re.rearpod_max_length_m) * mm),
         (None, re.y_body_half_m * mm),
         (re.z_floor_m * mm, re.z_rearpod_top_m * mm)),
        ((bv.ref_plane_A_m * mm, bv.rearpod.origin_m[0] * mm), (None, re.y_body_half_m * mm),
         (re.z_floor_m * mm, re.z_body_top_m * mm)),
    ]
    if not d_mm:
        specs = _extend_abutting(specs)
    boxes = [_box(X, aY, Z, x=s[0], y=s[1], z=s[2]) for s in specs]
    return -(union(boxes, d_mm) if d_mm else np.maximum.reduce(boxes))


def wheel_columns(X, aY, Z, W_mm, x_front_mm):
    """unified_phi.wheel_keepclear_mask: full-height columns over the four wheels."""
    from geometry_contract import (FRONT_WHEEL_INNER_Y_M, REAR_WHEEL_INNER_Y_M,
                                   R_WHEEL_M, WHEEL_CLEARANCE_M, WHEEL_WIDTH_M)
    from unified_phi import WHEEL_KEEPOUT_MARGIN_M as k
    r = (R_WHEEL_M + WHEEL_CLEARANCE_M + k) * 1e3
    out = []
    for ax, inner in ((x_front_mm, FRONT_WHEEL_INNER_Y_M), (x_front_mm + W_mm, REAR_WHEEL_INNER_Y_M)):
        y0 = (inner - WHEEL_CLEARANCE_M - k) * 1e3
        y1 = (inner + WHEEL_WIDTH_M + WHEEL_CLEARANCE_M) * 1e3
        out.append(_box(X, aY, Z, x=(ax - r, ax + r), y=(y0, y1)))
    return np.maximum(out[0], out[1])


def claw_rings(X, aY, Z, W_mm, x_front_mm):
    """T7.13: the claw ring 1.5 mm in from each wheel's inner face over its outer 1.5 mm of
    radius (legality T7.13, CLAW_LIP_MM). The wheel columns already cover it; kept explicit
    so the carve can never leave body there (44130/0, 44430/0, Part 4, 2026-11-14)."""
    from geometry_contract import FRONT_WHEEL_INNER_Y_M, REAR_WHEEL_INNER_Y_M, R_WHEEL_M
    lip, R = 1.5, R_WHEEL_M * 1e3
    out = []
    for ax, inner in ((x_front_mm, FRONT_WHEEL_INNER_Y_M), (x_front_mm + W_mm, REAR_WHEEL_INNER_Y_M)):
        yi = inner * 1e3
        rho = np.hypot(X - ax, Z - R)
        out.append(np.minimum(_box(X, aY, Z, y=(yi - lip, yi)), lip / 2 - np.abs(rho - (R - lip / 2))))
    return np.maximum(out[0], out[1])


def t79_zones(X, aY, Z, W_mm, x_front_mm):
    """wheel_visibility_zones.build_t79_forbidden_mask: the T7.9.1 and T7.9.4 rectangles and
    the T7.9.2 / T7.9.3 wedges, without the mask's one-cell growth (MARGIN_MM instead)."""
    import wheel_visibility_zones as wv
    from geometry_contract import R_WHEEL_M
    R = R_WHEEL_M * 1e3
    yf, yr = wv.FRONT_INNER_GAP_ACTUAL_MM / 2.0, wv.REAR_INNER_GAP_ACTUAL_MM / 2.0
    zt = wv.T79_ZONE_HEIGHT_MM
    ff, fa = x_front_mm - R, x_front_mm + R
    rf, ra = x_front_mm + W_mm - R, x_front_mm + W_mm + R
    rect_f = _box(X, aY, Z, x=(ff - wv.T79_RECT_DEPTH_MM, ff), y=(yf, None), z=(0.0, zt))
    rect_r = _box(X, aY, Z, x=(ra, ra + wv.T79_RECT_DEPTH_MM), y=(yr, None), z=(0.0, zt))

    def wedge(x_wheel, near, ang, aft, yi):
        a = np.radians(ang)
        depth = (X - x_wheel) if aft else (x_wheel - X)
        far = (near + (aY - yi) / np.tan(a) - depth) * np.sin(a)      # true distance to the slant
        slab = _box(depth, aY, Z, x=(0.0, None), y=(yi, yi + wv.T79_WIDTH_Y_MM), z=(0.0, zt))
        return np.minimum(slab, far)
    w_f = wedge(fa, wv.T79_FRONT_NEAR_X_MM, wv.T79_FRONT_ANGLE_DEG, True, yf)
    w_r = wedge(rf, wv.T79_REAR_NEAR_X_MM, wv.T79_REAR_ANGLE_DEG, False, yr)
    return np.maximum.reduce([rect_f, rect_r, w_f, w_r])


def halo_pocket_and_ballast(X, aY, Z, ref_A_mm, d_halo_mm, d_mm=1.0, ballast=True):
    """halo_pocket.build_halo_pocket_forbidden_mask and build_ballast_container_forbidden_mask
    (the capsule at its true size; the mask's one-cell inset is not needed here)."""
    import halo_pocket as hp
    box = hp.compute_halo_pocket_box_m(ref_A_mm / 1e3, d_halo_mm)
    b = {k: v * 1e3 for k, v in box.items()}
    pocket = _box(X, aY, Z, x=(b["x_min_m"], b["x_max_m"]), y=(None, b["y_max_m"]),
                  z=(b["z_min_m"], b["z_max_m"]))
    r = hp.BALLAST_SLOT_WIDTH_MM / 2.0
    hs = max(hp.BALLAST_SLOT_LENGTH_MM / 2.0 - r, 0.0)
    cx = b["x_min_m"] + hp.BALLAST_CENTRE_FROM_POCKET_FRONT_MM
    z_top, z_bot = b["z_min_m"], b["z_min_m"] - hp.BALLAST_DEPTH_MM
    plan = r - np.hypot(np.maximum(np.abs(X - cx) - hs, 0.0), aY)
    capsule = np.minimum(plan, np.minimum(Z - z_bot, z_top - Z))
    if not ballast:
        return pocket
    return union([pocket, capsule], d_mm) if d_mm else np.maximum(pocket, capsule)


def block_slot(X, aY, Z):
    """unified_phi.model_block_slot_mask: the block's 6 x 6 mm underside slot."""
    from geometry_contract import MODEL_BLOCK_SLOT_TOP_Z_MM, MODEL_BLOCK_SLOT_WIDTH_MM
    return _box(X, aY, Z, y=(None, MODEL_BLOCK_SLOT_WIDTH_MM / 2.0), z=(None, MODEL_BLOCK_SLOT_TOP_Z_MM))


def t851_cap(X, aY, Z, ref_A_mm):
    """unified_phi.build_rule_masks' T8.5.1 cap: within 1 mm of Ref A the body stays inside
    14.5 mm of the centreline and 24.5 mm of the track (the mask adds a cell; not needed)."""
    return np.minimum(ref_A_mm + 1.0 - X, np.maximum(aY - 14.5, Z - 24.5))
def _shift(a, s, axis, fill=0.0):
    """a moved by s cells along axis (value at p is a[p - s]); the cells it uncovers get fill."""
    out = np.full_like(a, fill)
    src = [slice(None)] * 3
    dst = [slice(None)] * 3
    if s > 0:
        src[axis], dst[axis] = slice(0, -s), slice(s, None)
    else:
        src[axis], dst[axis] = slice(-s, None), slice(0, s)
    out[tuple(dst)] = a[tuple(src)]
    return out


def union(fields, d_mm):
    """Signed distance to the union of shapes, positive inside, from each shape's own field.

    max() of the fields is exact outside the union but only a lower bound inside: where two
    shapes touch (the envelope's boxes at Ref A, the pocket on the ballast slot) it is 0 on
    the seam, deep inside, and a body stamped with it was grooved along every seam (a 1 g
    loss, 2026-11-09). Inside, the distance is taken from the nearest cells that are truly
    outside, up to two cells along each axis: |step| - their outside distance, exact for a
    flat face. Further in than two cells it is 2 cells (deep enough for the stamp)."""
    lb = np.maximum.reduce(fields)
    out = np.maximum(-lb, 0.0)
    est = np.full(lb.shape, 2.0 * d_mm)
    for ax in range(3):
        for s in (1, -1, 2, -2):
            o = _shift(out, s, ax)
            est = np.where(o > 0, np.minimum(est, abs(s) * d_mm - o), est)
    return np.where(lb > 0, np.maximum(lb, est), lb)


def air_field_mm(g, X, Y, Z) -> np.ndarray:
    """All the zones above for geometry g, cell centres X, Y, Z in mm: positive inside any.
    Limits the referee checks (envelope, wheels, T7.9, T8.5.1) carry MARGIN_MM; the pocket,
    ballast slot and block slot, which Part 4 cuts exactly, keep cells on their faces solid."""
    from bounding_volumes import default_rule_envelope
    d = float(X[1, 0, 0] - X[0, 0, 0]) if X.shape[0] > 1 else 1.0
    aY = np.abs(Y)
    ref_A = g.landmarks["ref_plane_A_m"] * 1e3
    exact = -1e-6
    return union([
        envelope_air(X, aY, Z, g.bv, default_rule_envelope(), d) + MARGIN_MM,
        wheel_columns(X, aY, Z, g.W_mm, g.x_front_mm) + MARGIN_MM,
        t79_zones(X, aY, Z, g.W_mm, g.x_front_mm) + MARGIN_MM,
        claw_rings(X, aY, Z, g.W_mm, g.x_front_mm) + MARGIN_MM,
        t851_cap(X, aY, Z, ref_A) + MARGIN_MM,
        # The pocket only. Stamping the ballast capsule at its exact size let the rounding of
        # its top rim eat the pocket floor up to 0.85 mm beyond it, under the T4.2 cargo (6-20
        # mm3, Part 4, 2026-11-10). The mask draws it a cell small; Part 4 cuts it exactly.
        halo_pocket_and_ballast(X, aY, Z, ref_A, g.d_halo_mm, d, ballast=False) + exact,
        block_slot(X, aY, Z) + exact,
        halo_zones(X, aY, Z, ref_A, g.d_halo_mm, part="above") + exact,
        halo_zones(X, aY, Z, ref_A, g.d_halo_mm, _canister(g), part="shadows") + MARGIN_MM,
    ], d)


def stamp(g, X, Y, Z) -> int:
    """phi = max(phi, zone distance) within two cells of the zones' faces, so the body's
    surface sits on (or MARGIN_MM clear of) each zone's true face at any grid. Cells the
    rules force solid are left alone (air wins only where the masks themselves overlap).
    Returns the cells changed."""
    d_mm = float(X[1, 0, 0] - X[0, 0, 0]) if X.shape[0] > 1 else 1.0
    a = air_field_mm(g, X, Y, Z)
    near = (a > -2.0 * d_mm) & ~g.phi.hard_mask_solid
    new = np.maximum(g.phi.grid[near], (a[near] / 1e3).astype(np.float32))
    changed = int((new != g.phi.grid[near]).sum())
    g.phi.grid[near] = new
    # The halo's view zones and the loft ceiling also overrule forced solid: the loft skin
    # and halo deck are forced cell by cell and poked up to 2.5 mm above the exact ceiling
    # (2026-11-10). Elsewhere forced solid stands (the cargo may share the ballast slot).
    b = halo_zones(X, np.abs(Y), Z, g.landmarks["ref_plane_A_m"] * 1e3, g.d_halo_mm, _canister(g),
                   part="shadows") + MARGIN_MM
    over = (b > -2.0 * d_mm) & g.phi.hard_mask_solid
    new = np.maximum(g.phi.grid[over], (b[over] / 1e3).astype(np.float32))
    changed += int((new != g.phi.grid[over]).sum())
    g.phi.grid[over] = new
    g.phi.hard_mask_solid[over & (b > 0)] = False      # and they are no longer required solid
    return changed


def air_at_points(g, P):
    """The zones' air field (as air_field_mm, with its margins) at points P (n, 3) in mm.
    Not the ballast slot: the cargo may share it (T4.2) and Part 4 cuts it exactly.
    Exact where it is positive (inside a zone), which is all project_out needs."""
    from bounding_volumes import default_rule_envelope
    X, Y, Z = P[:, 0], P[:, 1], P[:, 2]
    aY = np.abs(Y)
    ref_A = g.landmarks["ref_plane_A_m"] * 1e3
    return np.maximum.reduce([
        envelope_air(X, aY, Z, g.bv, default_rule_envelope(), None) + MARGIN_MM,
        wheel_columns(X, aY, Z, g.W_mm, g.x_front_mm) + MARGIN_MM,
        claw_rings(X, aY, Z, g.W_mm, g.x_front_mm) + MARGIN_MM,
        t79_zones(X, aY, Z, g.W_mm, g.x_front_mm) + MARGIN_MM,
        t851_cap(X, aY, Z, ref_A) + MARGIN_MM,
        block_slot(X, aY, Z) - 1e-6,
        halo_zones(X, aY, Z, ref_A, g.d_halo_mm, part="above") - 1e-6,
        halo_zones(X, aY, Z, ref_A, g.d_halo_mm, _canister(g), part="shadows") + MARGIN_MM,
    ])


def cargo_faces(g):
    """The T4.2 cargo wedge's faces as unit normals N (6, 3) and offsets c (mm): inside where
    N @ p < c on every face. Order: front, rear, +y side, -y side, bottom, top. None without
    a cargo."""
    import virtual_cargo as vc
    p = (getattr(g, "landmarks", None) or {}).get("cargo_placement")
    if not p:
        return None
    x0 = p["x_start_m"] * 1e3
    x1 = x0 + vc.CARGO_LENGTH_MM
    zb = max(p["z_base_m"], g.region.origin_m[2]) * 1e3      # as the field builds it
    w0, w1 = vc.CARGO_WIDE_WIDTH_MM / 2.0, vc.CARGO_NARROW_WIDTH_MM / 2.0
    if p.get("flip", False):
        w0, w1 = w1, w0
    k = (w1 - w0) / vc.CARGO_LENGTH_MM                     # |y| <= w0 + k (x - x0)
    s = np.hypot(1.0, k)
    N = np.array([[-1, 0, 0], [1, 0, 0], [-k / s, 1 / s, 0], [-k / s, -1 / s, 0], [0, 0, -1], [0, 0, 1]], float)
    c = np.array([-x0, x1, (w0 - k * x0) / s, (w0 - k * x0) / s, -zb, zb + vc.CARGO_HEIGHT_MM])
    return N, c


def out_of_cargo(g, P_mm):
    """Surface vertices (mm) inside the T4.2 cargo moved out through its nearest face, 0.9 um
    clear: the cargo is forced solid a cell past its faces, but marching cubes and smoothing
    left a carved car's slanted side walls 6-13 um inside it (5.8-10.7 mm3, 2 of 20 carved
    cars, Part 4, 2026-11-10). Not through the top (it is the pocket floor, the air side's)
    nor the y = 0 cap; the zones' projection runs after this and wins where they overlap."""
    f = cargo_faces(g)
    if f is None:
        return P_mm
    N, c = f
    s = P_mm @ N.T - c                                       # < 0 inside, per face
    j = s.argmax(axis=1)                                     # the nearest face
    m = (s.max(axis=1) < 7e-4) & (j != 5) & (P_mm[:, 1] > 1e-6)
    P = P_mm.copy()
    P[m] += (9e-4 - s[m, j[m]])[:, None] * N[j[m]]
    return P


def project_out(g, V_m, iters: int = 16):
    """Mesh vertices (metres) that smoothing pushed into a zone moved back onto its face.
    Taubin smoothing rounds every sharp edge a little; where the edge is a zone's (the block
    slot's top corners, the block's faces) the rounding put body inside the zone: 5.4 mm3 in
    the slot on Part 4's start car, 2026-11-09. Returns the vertices and how many moved."""
    P = out_of_cargo(g, np.asarray(V_m, dtype=np.float64) * 1e3)
    moved = np.zeros(len(P), bool)
    h = 1e-3
    for _ in range(iters):
        a = air_at_points(g, P)
        m = a > -7e-4                 # at least 0.7 um clear of every face
        if not m.any():
            break
        moved |= m
        Q = P[m]
        grad = np.stack([(air_at_points(g, Q + e) - air_at_points(g, Q - e)) / (2 * h)
                         for e in np.eye(3) * h], axis=1)
        # A step only where the field has a slope: on a ridge between two zones the central
        # difference is near zero and a/|grad|^2 threw a vertex 1,000 mm out (2026-11-10).
        g2 = (grad ** 2).sum(1)
        # 0.9 um past the face, repeated until every vertex is 0.7 um clear: landing exactly on
        # it left Newton a hair short on curved faces (3 vertices 0.002 mm in, 2026-11-10). Not
        # 3 um: the pocket floor stood 3 um under the T4.2 cargo top (2026-11-10). cut_boxes grows
        # its boxes 0.1-0.3 um, so no vertex sits on a box face (that made the boolean fail and
        # the slot went uncut, 28-35 mm3, Part 4, 2026-11-10).
        P[m] = Q - np.where(g2 > 0.25, (a[m] + 9e-4) / np.maximum(g2, 1e-12), 0.0)[:, None] * grad
    return P / 1e3, int(moved.sum())


def halo_zones(X, aY, Z, ref_A_mm, d_halo_mm, canister=None, part="all"):
    """fixed_hardware's halo void, halo_visibility_air_mask and halo_canister_loft_air_mask,
    exact: the halo is the box over the pocket, |y| <= HALO_CROSS_SECTION_HALF_WIDTH_MM, z
    from HALO_MIN_Z up to its top. Air: above it (top view, T4.4.3), ahead of it and outboard
    of it above the 4 mm fillet (front and side views, T4.4.2), and above the smoothstep deck
    from the halo's rear face to the bore's front (the loft ceiling)."""
    import fixed_hardware as fh
    import halo_pocket as hp
    from geometry_contract import HALO_MIN_Z_MM
    box = hp.compute_halo_pocket_box_m(ref_A_mm / 1e3, d_halo_mm)
    x0, x1 = box["x_min_m"] * 1e3, box["x_max_m"] * 1e3
    hw, z0, zt = fh.HALO_CROSS_SECTION_HALF_WIDTH_MM, HALO_MIN_Z_MM, fh.HALO_CROSS_SECTION_TOP_MM
    zf = z0 + fh.HALO_VISIBILITY_FILLET_MM
    above = _box(X, aY, Z, x=(x0, x1), y=(None, hw), z=(z0, None))           # halo and above
    if part == "above":              # its floor is the pocket floor, exactly 24 mm: no margin
        return above
    out = [_box(X, aY, Z, x=(None, x0), y=(None, hw), z=(zf, zt)),          # its front shadow
           _box(X, aY, Z, x=(x0, x1), y=(hw, None), z=(zf, zt))]            # its side shadows
    if canister is not None:
        r = fh.CANISTER_CLEARANCE_RADIUS_MM
        zc = canister.z_center_m * 1e3
        xc = (canister.x_center_m - canister.x_half_width_m) * 1e3
        if xc > x1:
            t = np.clip((X - x1) / (xc - x1), 0.0, 1.0)
            s = t * t * (3.0 - 2.0 * t)
            yv = min(hw, r)                       # where both the halo and bore profiles exist
            z_h = fh.HALO_REAR_FACE_TOP_MM
            z_can = zc + np.sqrt(np.maximum(r * r - np.minimum(aY, yv) ** 2, 0.0))
            crown = z_h + s * (zc + r - z_h)      # the highest valid profile point, at y = 0
            ceil_ = np.where(aY <= yv, z_h + s * (z_can - z_h), crown)
            # The deck from the halo's rear face to the bore front, then flat at the bore's top
            # to the tail; each piece with true end faces. The mask puts each join on the cell
            # holding it, and a one-sided switch there put mask cells 2-6 mm out (2026-11-10).
            # (positive inside: a piece is the min of its faces, the union the max)
            out.append(np.minimum.reduce([X - x1, xc - X, Z - ceil_]))
            # The flat part from the halo's rear, not the bore front: ahead of the bore it is
            # inside the deck part anyway, and a face at the join gave points above both
            # ceilings no depth there; the projection bounced them across it (2026-11-10).
            out.append(np.minimum(X - x1, Z - (zc + r)))
            # The deck run on forward over the halo, MARGIN_MM inside the air above it, so
            # the halo's rear face is no seam for points above the deck (they sat on it,
            # 2026-11-10) and the margin added later does not reach past that exact zone.
            out.append(np.minimum.reduce([X - x0 - MARGIN_MM, hw - MARGIN_MM - aY, xc - X, Z - ceil_]))
    if part == "shadows":
        return np.maximum.reduce(out)
    return np.maximum.reduce(out + [above])


def _canister(g):
    return getattr(getattr(g, "fixed_hardware", None), "canister_cylinder", None)


def _air_boxes(g):
    """The box-shaped air zones cut into the body, as ((x lo, hi), (|y| lo, hi), (z lo, hi))
    in mm (None = open): their edges are inside corners of the body that marching cubes
    chamfers into the zone (the block slot's top edges: 5.5 mm3, Part 4, 2026-11-09)."""
    import halo_pocket as hp
    import wheel_visibility_zones as wv
    from geometry_contract import (FRONT_WHEEL_INNER_Y_M, MODEL_BLOCK_SLOT_TOP_Z_MM,
                                   MODEL_BLOCK_SLOT_WIDTH_MM, R_WHEEL_M, REAR_WHEEL_INNER_Y_M,
                                   WHEEL_CLEARANCE_M, WHEEL_WIDTH_M)
    from unified_phi import WHEEL_KEEPOUT_MARGIN_M as k
    ref_A = g.landmarks["ref_plane_A_m"] * 1e3
    W, xf = g.W_mm, g.x_front_mm
    out = [((None, None), (None, MODEL_BLOCK_SLOT_WIDTH_MM / 2.0), (None, MODEL_BLOCK_SLOT_TOP_Z_MM))]
    m = MARGIN_MM                   # zones the stamp keeps MARGIN_MM clear: edges moved out too
    r = (R_WHEEL_M + WHEEL_CLEARANCE_M + k) * 1e3 + m
    for ax, inner in ((xf, FRONT_WHEEL_INNER_Y_M), (xf + W, REAR_WHEEL_INNER_Y_M)):
        out.append(((ax - r, ax + r), ((inner - WHEEL_CLEARANCE_M - k) * 1e3 - m,
                    (inner + WHEEL_WIDTH_M + WHEEL_CLEARANCE_M) * 1e3 + m), (None, None)))
    R = R_WHEEL_M * 1e3
    ff, ra = xf - R, xf + W + R
    out.append(((ff - wv.T79_RECT_DEPTH_MM - m, ff + m), (wv.FRONT_INNER_GAP_ACTUAL_MM / 2.0 - m, None),
                (None, wv.T79_ZONE_HEIGHT_MM + m)))
    out.append(((ra - m, ra + wv.T79_RECT_DEPTH_MM + m), (wv.REAR_INNER_GAP_ACTUAL_MM / 2.0 - m, None),
                (None, wv.T79_ZONE_HEIGHT_MM + m)))
    b = {kk: v * 1e3 for kk, v in hp.compute_halo_pocket_box_m(ref_A / 1e3, g.d_halo_mm).items()}
    out.append(((b["x_min_m"], b["x_max_m"]), (None, b["y_max_m"]), (b["z_min_m"], b["z_max_m"])))
    # the halo and the air above it (exact), and its front and side view shadows (T4.4.2,
    # margin): 480 of 1,008 carved cars had 6-13 mm3 ahead of the halo (Part 4, 2026-11-10)
    import fixed_hardware as fh
    from geometry_contract import HALO_MIN_Z_MM
    hw, zt = fh.HALO_CROSS_SECTION_HALF_WIDTH_MM, fh.HALO_CROSS_SECTION_TOP_MM
    zf = HALO_MIN_Z_MM + fh.HALO_VISIBILITY_FILLET_MM
    x0, x1 = b["x_min_m"], b["x_max_m"]
    out.append(((x0, x1), (None, hw), (HALO_MIN_Z_MM, None)))
    out.append(((None, x0 + m), (None, hw + m), (zf - m, zt + m)))
    out.append(((x0 - m, x1 + m), (hw - m, None), (zf - m, zt + m)))
    return out


def cut_boxes(g, mesh):
    """The half body (metres) minus the box air zones, by an exact solid boolean: their edges
    come out sharp, where marching cubes cut across them (the block slot's top corners, 5.5
    mm3 inside the slot, Part 4, 2026-11-09). Snapping vertices onto the edges instead broke
    the mesh (not a volume). Returns the mesh unchanged if the boolean does not give a volume."""
    import halo_pocket as hp
    import manifold3d as m3
    import trimesh
    try:
        body = m3.Manifold(m3.Mesh(vert_properties=np.asarray(mesh.vertices, np.float32),
                                   tri_verts=np.asarray(mesh.faces, np.uint32)))
    except Exception:  # noqa: BLE001
        return mesh
    if body.is_empty():
        return mesh
    import io
    # Each box a hair (0.1-0.3 um) past its face: project_out leaves every vertex at least
    # 0.7 um clear of the faces, and a boolean of coincident faces leaves zero-width slivers
    # whose corners an STL's float32 merges, after which the mesh is not a volume (2026-11-10).
    # The hair stays under 1 um so the pocket floor stays on the T4.2 cargo top. Edges shorter
    # than 0.2 um (where several boxes' edges meet: 16 bad edges at x 180.68, y 11.85 on the
    # 0.5 mm body, 2026-11-10) are collapsed before the STL. Checked as delivered, through an
    # STL and back; other offsets tried if that reload still is not a volume.
    # Five of 40 carved bodies still came back uncut from (2e-4, 3e-4, 1e-4) alone, leaving the
    # block slot and halo corners 35-40 mm3 illegal (Jdiejei 43154160454, 2026-11-12): more
    # offsets, then a smaller edge collapse, before giving up.
    body0 = body
    for tol, e in [(t, e) for t in (2e-7, 5e-7, 1e-7) for e in (2e-4, 3e-4, 1e-4, 1.5e-4, 2.5e-4)]:
        body = body0
        cut = None
        for box in _air_boxes(g):
            lo = [(b[0] - e) if b[0] is not None else (-1.0 if a == 1 else -500.0) for a, b in enumerate(box)]
            hi = [(b[1] + e) if b[1] is not None else 500.0 for b in box]
            c = m3.Manifold.cube([(h - l) / 1e3 for l, h in zip(lo, hi)]).translate([l / 1e3 for l in lo])
            cut = c if cut is None else cut + c
        # The ballast slot, exact (the field draws none: the rounding of a drawn slot's rim
        # took the pocket floor up to 0.025 mm under the T4.2 cargo top, 2026-11-10)
        cap = hp.ballast_capsule_mesh_m(g.landmarks["ref_plane_A_m"], g.d_halo_mm, grow_mm=e)
        # The T4.2 cargo, exact, added to the body: vertices pushed out of it still left
        # triangles cutting across its vertical edges (two 3.3 mm3 pieces at its front corners,
        # Part 4, 2026-11-11). Its y = 0 face 0.2 um inside the body (no face on the cap) and
        # its top 0.5 um under the floor's cut, so no face of it lies on another one.
        cf = cargo_faces(g)
        if cf is not None:
            N, c = cf
            x0, x1, zb, zt = -c[0], c[1], -c[4], c[5] - 5e-4
            k, s = -N[2, 0] / N[2, 1], 1.0 / N[2, 1]
            w = lambda x: c[2] * s + k * x                    # half-width at x
            pts = [(x, y, z) for x in (x0, x1) for y in (2e-4, None) for z in (zb, zt)]
            pts = [(x, w(x) if y is None else y, z) for x, y, z in pts]
            body = body + m3.Manifold.hull_points(np.asarray(pts) / 1e3)
        cut = cut + m3.Manifold(m3.Mesh(vert_properties=np.asarray(cap.vertices, np.float32),
                                        tri_verts=np.asarray(cap.faces, np.uint32)))
        o = (body - cut).simplify(tol).to_mesh()
        raw = trimesh.Trimesh(np.asarray(o.vert_properties)[:, :3].astype(np.float64), np.asarray(o.tri_verts),
                              process=False)
        out = trimesh.load(io.BytesIO(raw.export(file_type="stl")), file_type="stl", force="mesh")
        if out.is_volume:
            return out
    import warnings
    warnings.warn("cut_boxes: the boolean did not give a volume; mesh left uncut", RuntimeWarning, stacklevel=2)
    return mesh
