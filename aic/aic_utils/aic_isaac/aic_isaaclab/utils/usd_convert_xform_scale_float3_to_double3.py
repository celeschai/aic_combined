#!/usr/bin/env python3
"""Convert USD xformOp:scale from float precision (float3 / vector3f) to double3.

Uses Pixar USD Python bindings (``pxr``). Iterates all prims on a stage, finds
``xformOp:scale`` attributes stored as ``float3`` or ``vector3f``, removes the
property on the authoring layer, and recreates it as ``double3``
(``Sdf.ValueTypeNames.Double3``) with the same numeric values.

Requires the Isaac Lab / Omniverse Python that provides ``pxr`` (see CLAUDE.md).

Example::

    /path/to/env_isaaclab/bin/python3 usd_convert_xform_scale_float3_to_double3.py \\
        "Intrinsic_assets/.../task_board_assembly.usd" -o task_board_assembly_fixed.usd

    # Preview only (no write)
    ... task_board_assembly.usd --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pxr import Gf, Sdf, Usd


# Types that should become vector3d for xform scale ops.
_FLOAT3_SCALE_TYPES = frozenset(
    {
        Sdf.ValueTypeNames.Float3,
        Sdf.ValueTypeNames.Vector3f,
    }
)


def _is_float3_scale_type(type_name: Sdf.ValueTypeName) -> bool:
    return type_name in _FLOAT3_SCALE_TYPES


def _find_authoring_layer_for_property(
    stage: Usd.Stage, prim_path: Sdf.Path, prop_name: str
) -> Sdf.Layer | None:
    """Return the strongest layer that actually authors ``prop_name`` on ``prim_path``."""
    for layer in stage.GetLayerStack():
        prim_spec = layer.GetPrimAtPath(prim_path)
        if prim_spec is None:
            continue
        props = getattr(prim_spec, "properties", None)
        if props is not None and prop_name in props:
            return layer
    return None


def _remove_property_spec(prim_spec: Sdf.PrimSpec, prop_name: str) -> bool:
    props = getattr(prim_spec, "properties", None)
    if props is None or prop_name not in props:
        return False
    prim_spec.RemoveProperty(props[prop_name])
    return True


def _vec3f_to_vec3d(val) -> Gf.Vec3d:
    return Gf.Vec3d(float(val[0]), float(val[1]), float(val[2]))


def convert_xform_scale_to_double3(
    stage: Usd.Stage,
    *,
    dry_run: bool = False,
) -> tuple[int, list[str]]:
    """Convert float-precision ``xformOp:scale`` to ``double3``.

    Returns ``(converted_count, warnings)``.
    """
    attr_name = "xformOp:scale"
    converted = 0
    warnings: list[str] = []

    for prim in Usd.PrimRange(stage.GetPseudoRoot()):
        if not prim.IsValid():
            continue
        attr = prim.GetAttribute(attr_name)
        if not attr.IsValid():
            continue
        if not _is_float3_scale_type(attr.GetTypeName()):
            continue

        val = attr.Get()
        if val is None:
            warnings.append(f"{prim.GetPath()}: {attr_name} has no value — skipped")
            continue

        if dry_run:
            converted += 1
            continue

        layer = _find_authoring_layer_for_property(stage, prim.GetPath(), attr_name)
        if layer is None:
            warnings.append(
                f"{prim.GetPath()}: {attr_name} not authored in any layer in the stack — skipped "
                "(likely from a locked reference; edit that file instead)"
            )
            continue

        with Usd.EditContext(stage, layer):
            prim_spec = layer.GetPrimAtPath(prim.GetPath())
            if prim_spec is None or not _remove_property_spec(prim_spec, attr_name):
                warnings.append(
                    f"{prim.GetPath()}: could not remove {attr_name} from layer "
                    f"{getattr(layer, 'identifier', layer)} — skipped"
                )
                continue

            p = stage.GetPrimAtPath(prim.GetPath())
            new_attr = p.CreateAttribute(attr_name, Sdf.ValueTypeNames.Double3)
            new_attr.Set(_vec3f_to_vec3d(val))

        converted += 1

    return converted, warnings


def _open_stage(path: Path) -> Usd.Stage:
    stage = Usd.Stage.Open(str(path))
    if not stage:
        raise SystemExit(f"Failed to open USD stage: {path}")
    return stage


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Set all xformOp:scale ops from float3/vector3f to double3."
    )
    parser.add_argument(
        "usd_path",
        type=Path,
        help="Input .usd / .usda / .usdc file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output path. Default: <input_stem>_scale_double3<suffix> next to the input file.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Save over the input file (use with care).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count prims that would change; do not write.",
    )
    args = parser.parse_args(argv)

    usd_path = args.usd_path.expanduser().resolve()
    if not usd_path.is_file():
        print(f"Not a file: {usd_path}", file=sys.stderr)
        return 1

    if args.in_place and args.output is not None:
        print("Use either --in-place or -o, not both.", file=sys.stderr)
        return 1

    stage = _open_stage(usd_path)

    converted, warnings = convert_xform_scale_to_double3(stage, dry_run=args.dry_run)
    print(f"{'Would convert' if args.dry_run else 'Converted'} xformOp:scale (float3/vector3f→double3): {converted}")
    for w in warnings:
        print(f"warning: {w}", file=sys.stderr)

    if args.dry_run:
        return 0

    if args.in_place:
        out_path = usd_path
    elif args.output is not None:
        out_path = args.output.expanduser().resolve()
    else:
        out_path = usd_path.with_name(f"{usd_path.stem}_scale_double3{usd_path.suffix}")

    # Export ensures a single file even if the stage had multiple anonymous layers.
    root = stage.GetRootLayer()
    if args.in_place:
        root.Save()
        print(f"Saved in place: {out_path}")
    else:
        stage.Export(str(out_path))
        print(f"Wrote: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
