#!/usr/bin/env python3
"""Inspect USD mesh: stage metersPerUnit and world-space aggregate bounds (mm).

Requires Isaac Sim Python (pxr). `pxr` is available after Isaac Sim starts; this
script uses `SimulationApp` then imports `pxr`.

Run via repo helper (uses `AIC_ISAACLAB_PYTHON` from `.env` when present):

  ./scripts/run-isaaclab.sh aic_utils/aic_isaac/aic_isaaclab/scripts/inspect_usd_mesh_extents.py [path.usd]

For unbuffered prints (handy when piping), prefix with `PYTHONUNBUFFERED=1`.
"""

from __future__ import annotations

import argparse
import os
import sys

from isaacsim import SimulationApp

# Start Isaac Sim before importing pxr (USD bindings live on the Kit path).
_sim_app = SimulationApp({"headless": True})

from pxr import Gf, Usd, UsdGeom


def _bbox_from_points(pts) -> tuple[Gf.Vec3d, Gf.Vec3d] | None:
    if pts is None or len(pts) == 0:
        return None
    p0 = pts[0]
    mn = Gf.Vec3d(float(p0[0]), float(p0[1]), float(p0[2]))
    mx = Gf.Vec3d(float(p0[0]), float(p0[1]), float(p0[2]))
    for p in pts[1:]:
        x, y, z = float(p[0]), float(p[1]), float(p[2])
        mn = Gf.Vec3d(min(mn[0], x), min(mn[1], y), min(mn[2], z))
        mx = Gf.Vec3d(max(mx[0], x), max(mx[1], y), max(mx[2], z))
    return mn, mx


def _centroid_from_points(pts) -> Gf.Vec3d | None:
    if pts is None or len(pts) == 0:
        return None
    n = float(len(pts))
    sx = sy = sz = 0.0
    for p in pts:
        sx += float(p[0])
        sy += float(p[1])
        sz += float(p[2])
    return Gf.Vec3d(sx / n, sy / n, sz / n)


def _transform_point(m: Gf.Matrix4d, p: Gf.Vec3d) -> Gf.Vec3d:
    """Apply matrix to a point (translation + linear part)."""
    return m.Transform(p)


def _vec_mm(v: Gf.Vec3d, scale_mm: float) -> tuple[float, float, float]:
    return (v[0] * scale_mm, v[1] * scale_mm, v[2] * scale_mm)


def _union_range(a: Gf.Range3d | None, b: Gf.Range3d) -> Gf.Range3d:
    if a is None:
        return b
    mn = Gf.Vec3d(
        min(a.GetMin()[0], b.GetMin()[0]),
        min(a.GetMin()[1], b.GetMin()[1]),
        min(a.GetMin()[2], b.GetMin()[2]),
    )
    mx = Gf.Vec3d(
        max(a.GetMax()[0], b.GetMax()[0]),
        max(a.GetMax()[1], b.GetMax()[1]),
        max(a.GetMax()[2], b.GetMax()[2]),
    )
    return Gf.Range3d(mn, mx)


def main() -> int:
    # .../aic/aic_utils/aic_isaac/aic_isaaclab/scripts/this_file.py -> repo root is 4 parents up
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    default_usd = os.path.join(
        repo_root,
        "aic_utils",
        "aic_isaac",
        "aic_isaaclab",
        "source",
        "aic_task",
        "aic_task",
        "tasks",
        "manager_based",
        "aic_task",
        "Intrinsic_assets",
        "assets",
        "Task Board Base",
        "base_visual.usd",
    )
    p = argparse.ArgumentParser(description="Report USD stage units and mesh bounds in millimeters.")
    p.add_argument(
        "usd_path",
        nargs="?",
        default=default_usd,
        help="Path to .usd file (default: Task Board Base/base_visual.usd)",
    )
    args = p.parse_args()
    path = os.path.abspath(os.path.expanduser(args.usd_path))
    if not os.path.isfile(path):
        print(f"File not found: {path}", file=sys.stderr)
        return 1

    stage = Usd.Stage.Open(path)
    if not stage:
        print(f"Failed to open USD stage: {path}", file=sys.stderr)
        return 1

    meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
    up_axis = UsdGeom.GetStageUpAxis(stage)
    tc = Usd.TimeCode.Default()
    purpose = UsdGeom.Tokens.default_

    mesh_prims: list[Usd.Prim] = []
    union: Gf.Range3d | None = None
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh_prims.append(prim)
        imageable = UsdGeom.Imageable(prim)
        world_bound = imageable.ComputeWorldBound(tc, purpose)
        box = world_bound.ComputeAlignedBox()
        union = _union_range(union, box)

    mesh_count = len(mesh_prims)

    print("=== USD mesh inspection ===")
    print(f"path: {path}")
    print(f"stage upAxis: {up_axis}")
    print(f"stage metersPerUnit: {meters_per_unit}")
    print(
        "Interpretation: 1 USD length unit = "
        f"{meters_per_unit} m (world coordinates are in USD length units; "
        "multiply by metersPerUnit to get SI meters)."
    )
    print(f"Mesh prim count: {mesh_count}")

    if union is None:
        print("No UsdGeom.Mesh prims found — no bounds.")
        return 0

    mn = union.GetMin()
    mx = union.GetMax()
    extent_usd = Gf.Vec3d(mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2])
    # World positions are in USD units; meters = usd * metersPerUnit; mm = m * 1000
    scale_mm = float(meters_per_unit) * 1000.0
    extent_mm = Gf.Vec3d(extent_usd[0] * scale_mm, extent_usd[1] * scale_mm, extent_usd[2] * scale_mm)
    center_usd = Gf.Vec3d((mn[0] + mx[0]) * 0.5, (mn[1] + mx[1]) * 0.5, (mn[2] + mx[2]) * 0.5)

    print("\n--- Aggregate axis-aligned world bounds (all meshes, union) ---")
    print(f"min (USD units): ({mn[0]:.6f}, {mn[1]:.6f}, {mn[2]:.6f})")
    print(f"max (USD units): ({mx[0]:.6f}, {mx[1]:.6f}, {mx[2]:.6f})")
    print(f"extent width/depth/height (USD units): ({extent_usd[0]:.6f}, {extent_usd[1]:.6f}, {extent_usd[2]:.6f})")
    print(f"center (USD units): ({center_usd[0]:.6f}, {center_usd[1]:.6f}, {center_usd[2]:.6f})")
    print("\n--- Extents in millimeters (axis-aligned, per USD stage axes; upAxis above) ---")
    print(f"extent X x Y x Z (mm): {extent_mm[0]:.4f} x {extent_mm[1]:.4f} x {extent_mm[2]:.4f}")

    # Mesh origin vs geometric center (local + world)
    print("\n--- Mesh origin vs geometric center ---")
    print(
        "Mesh local origin is (0,0,0) in the mesh prim's space (UsdGeom.Mesh points are in this frame)."
    )
    print(
        "Geometric center: vertex AABB center and vertex mean (centroid) from the points attribute."
    )
    for prim in mesh_prims:
        mesh = UsdGeom.Mesh(prim)
        pts = mesh.GetPointsAttr().Get(tc)
        bb = _bbox_from_points(pts)
        if bb is None:
            print(f"\n[{prim.GetPath()}] no points — skip.")
            continue
        mn_l, mx_l = bb
        local_aabb_center = Gf.Vec3d((mn_l[0] + mx_l[0]) * 0.5, (mn_l[1] + mx_l[1]) * 0.5, (mn_l[2] + mx_l[2]) * 0.5)
        centroid_l = _centroid_from_points(pts)
        if centroid_l is None:
            continue

        # Offset from local origin (0,0,0) to geometric centers (same vector in local space)
        off_aabb_l = local_aabb_center
        off_centroid_l = centroid_l

        xform = UsdGeom.Xformable(prim)
        l2w = xform.ComputeLocalToWorldTransform(tc)
        origin_w = _transform_point(l2w, Gf.Vec3d(0.0, 0.0, 0.0))
        geom_aabb_w = _transform_point(l2w, local_aabb_center)
        geom_centroid_w = _transform_point(l2w, centroid_l)

        # World offset = geometric_center_world - mesh_origin_world (rigid transform: R*center + t - t = R*center)
        off_aabb_w = Gf.Vec3d(
            geom_aabb_w[0] - origin_w[0],
            geom_aabb_w[1] - origin_w[1],
            geom_aabb_w[2] - origin_w[2],
        )
        off_centroid_w = Gf.Vec3d(
            geom_centroid_w[0] - origin_w[0],
            geom_centroid_w[1] - origin_w[1],
            geom_centroid_w[2] - origin_w[2],
        )

        print(f"\nprim: {prim.GetPath()}")
        print(
            f"  local offset origin → AABB center (USD units): "
            f"({off_aabb_l[0]:.6f}, {off_aabb_l[1]:.6f}, {off_aabb_l[2]:.6f})"
        )
        print(f"  |offset| (USD units): {off_aabb_l.GetLength():.6f}")
        print(
            f"  local offset origin → vertex centroid (USD units): "
            f"({off_centroid_l[0]:.6f}, {off_centroid_l[1]:.6f}, {off_centroid_l[2]:.6f})"
        )
        print(f"  |offset| (USD units): {off_centroid_l.GetLength():.6f}")
        ax, ay, az = _vec_mm(off_aabb_l, scale_mm)
        print(f"  local offset origin → AABB center (mm): ({ax:.4f}, {ay:.4f}, {az:.4f})")
        cx, cy, cz = _vec_mm(off_centroid_l, scale_mm)
        print(f"  local offset origin → centroid (mm): ({cx:.4f}, {cy:.4f}, {cz:.4f})")

        print(
            f"  world offset origin → AABB center (USD units): "
            f"({off_aabb_w[0]:.6f}, {off_aabb_w[1]:.6f}, {off_aabb_w[2]:.6f})"
        )
        wx, wy, wz = _vec_mm(off_aabb_w, scale_mm)
        print(f"  world offset origin → AABB center (mm): ({wx:.4f}, {wy:.4f}, {wz:.4f})")
        print(
            f"  world offset origin → centroid (USD units): "
            f"({off_centroid_w[0]:.6f}, {off_centroid_w[1]:.6f}, {off_centroid_w[2]:.6f})"
        )
        wcx, wcy, wcz = _vec_mm(off_centroid_w, scale_mm)
        print(f"  world offset origin → centroid (mm): ({wcx:.4f}, {wcy:.4f}, {wcz:.4f})")

    sys.stdout.flush()

    return 0


if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    finally:
        _sim_app.close()
    raise SystemExit(rc)
