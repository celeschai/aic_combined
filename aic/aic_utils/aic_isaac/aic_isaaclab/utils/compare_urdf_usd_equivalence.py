#!/usr/bin/env python3
"""Compare robot.urdf (Gazebo) with aic_arm_only.usd (Isaac Lab) for equivalence.

Checks links, joints, mass, inertia, and joint positions to understand
gravity compensation differences between Gazebo and Isaac Lab.

Uses pxr for USD; run with Isaac Lab python:
  ./scripts/run-isaaclab.sh aic_utils/aic_isaac/aic_isaaclab/scripts/compare_urdf_usd_equivalence.py
  ./scripts/run-isaaclab.sh aic_utils/aic_isaac/aic_isaaclab/scripts/compare_urdf_usd_equivalence.py --urdf robot.urdf --usd path/to/aic_arm_only.usd
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import xml.etree.ElementTree as ET

# pxr requires Isaac Sim; use AppLauncher for headless
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Compare URDF vs USD for links, joints, mass, inertia"
)
parser.add_argument(
    "--urdf",
    default=None,
    help="Path to robot.urdf (default: repo root robot.urdf)",
)
parser.add_argument(
    "--usd",
    default=None,
    help="Path to aic_arm_only.usd (default: Intrinsic_assets/aic_arm_only.usd)",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import Gf, Usd, UsdGeom, UsdPhysics

# Paths
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR)))
)
DEFAULT_URDF = os.path.join(_REPO_ROOT, "robot.urdf")
AIC_ASSET_DIR = os.path.join(
    _SCRIPT_DIR,
    "../source/aic_task/aic_task/tasks/manager_based/aic_task/Intrinsic_assets",
)
DEFAULT_USD = os.path.join(AIC_ASSET_DIR, "aic_arm_only.usd")

# Map USD prim names to URDF link names (fallback when normalized match fails)
USD_TO_URDF_LINK = {
    "base": "base_link_inertia",
    "base_link_inertia": "base_link_inertia",
    "shoulder": "shoulder_link",
    "shoulder_link": "shoulder_link",
    "upper_arm": "upper_arm_link",
    "upper_arm_link": "upper_arm_link",
    "forearm": "forearm_link",
    "forearm_link": "forearm_link",
    "wrist_1": "wrist_1_link",
    "wrist_1_link": "wrist_1_link",
    "wrist_2": "wrist_2_link",
    "wrist_2_link": "wrist_2_link",
    "wrist_3": "wrist_3_link",
    "wrist_3_link": "wrist_3_link",
}

TOL_MASS = 0.01
TOL_POS = 1e-4
TOL_INERTIA = 1e-4


def parse_urdf(urdf_path):
    """Parse URDF and extract links (inertial) and joints."""
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    links = {}
    for link in root.findall("link"):
        name = link.get("name")
        if name is None:
            continue
        inertial = link.find("inertial")
        if inertial is None:
            continue
        mass_elem = inertial.find("mass")
        mass = float(mass_elem.get("value", 0)) if mass_elem is not None else 0
        origin = inertial.find("origin")
        xyz = (0, 0, 0)
        rpy = (0, 0, 0)
        if origin is not None:
            xyz_str = origin.get("xyz", "0 0 0")
            rpy_str = origin.get("rpy", "0 0 0")
            xyz = tuple(float(x) for x in xyz_str.split())
            rpy = tuple(float(x) for x in rpy_str.split())
        inertia = inertial.find("inertia")
        ixx = iyy = izz = ixy = ixz = iyz = 0
        if inertia is not None:
            ixx = float(inertia.get("ixx", 0))
            iyy = float(inertia.get("iyy", 0))
            izz = float(inertia.get("izz", 0))
            ixy = float(inertia.get("ixy", 0))
            ixz = float(inertia.get("ixz", 0))
            iyz = float(inertia.get("iyz", 0))
        links[name] = {
            "mass": mass,
            "com_xyz": xyz,
            "com_rpy": rpy,
            "inertia": {"ixx": ixx, "iyy": iyy, "izz": izz, "ixy": ixy, "ixz": ixz, "iyz": iyz},
        }

    joints = {}
    for joint in root.findall("joint"):
        name = joint.get("name")
        if name is None:
            continue
        jtype = joint.get("type", "fixed")
        parent = joint.find("parent")
        child = joint.find("child")
        parent_link = parent.get("link") if parent is not None else None
        child_link = child.get("link") if child is not None else None
        origin = joint.find("origin")
        xyz = (0, 0, 0)
        rpy = (0, 0, 0)
        if origin is not None:
            xyz_str = origin.get("xyz", "0 0 0")
            rpy_str = origin.get("rpy", "0 0 0")
            xyz = tuple(float(x) for x in xyz_str.split())
            rpy = tuple(float(x) for x in rpy_str.split())
        axis = (0, 0, 1)
        axis_elem = joint.find("axis")
        if axis_elem is not None:
            axis_str = axis_elem.get("xyz", "0 0 1")
            axis = tuple(float(x) for x in axis_str.split())
        joints[name] = {
            "type": jtype,
            "parent": parent_link,
            "child": child_link,
            "origin_xyz": xyz,
            "origin_rpy": rpy,
            "axis": axis,
        }

    return links, joints


def extract_usd_links(stage):
    """Extract mass, inertia, and local transform from USD prims with MassAPI."""
    results = []
    for prim in stage.Traverse():
        if not prim.IsValid():
            continue
        if not prim.HasAPI(UsdPhysics.MassAPI):
            continue
        mass_api = UsdPhysics.MassAPI(prim)
        mass_attr = mass_api.GetMassAttr()
        if not mass_attr:
            continue
        mass = mass_attr.Get()
        if mass is None:
            continue
        mass = float(mass)

        # Diagonal inertia (principal moments)
        diag = None
        diag_attr = mass_api.GetDiagonalInertiaAttr()
        if diag_attr:
            val = diag_attr.Get()
            if val is not None:
                diag = [float(val[0]), float(val[1]), float(val[2])]

        # Center of mass (if authored)
        com = None
        com_attr = mass_api.GetCenterOfMassAttr()
        if com_attr:
            val = com_attr.Get()
            if val is not None:
                com = (float(val[0]), float(val[1]), float(val[2]))

        path = str(prim.GetPath())
        name = prim.GetName()
        results.append({
            "path": path,
            "name": name,
            "mass": mass,
            "diagonal_inertia": diag,
            "center_of_mass": com,
        })
    return results


def extract_usd_joints(stage):
    """Extract joint data from UsdPhysics joints."""
    results = []
    for prim in stage.Traverse():
        if not prim.IsValid():
            continue
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)

        body0_targets = joint.GetBody0Rel().GetTargets()
        body1_targets = joint.GetBody1Rel().GetTargets()
        body0 = str(body0_targets[0]) if body0_targets else None
        body1 = str(body1_targets[0]) if body1_targets else None

        pos0_attr = joint.GetLocalPos0Attr()
        pos1_attr = joint.GetLocalPos1Attr()
        rot0_attr = joint.GetLocalRot0Attr()
        rot1_attr = joint.GetLocalRot1Attr()

        pos0 = pos0_attr.Get() if pos0_attr else Gf.Vec3f(0, 0, 0)
        pos1 = pos1_attr.Get() if pos1_attr else Gf.Vec3f(0, 0, 0)
        rot0 = rot0_attr.Get() if rot0_attr else Gf.Quatf(1, 0, 0, 0)
        rot1 = rot1_attr.Get() if rot1_attr else Gf.Quatf(1, 0, 0, 0)

        jtype = "fixed"
        axis = (0, 0, 1)
        if prim.IsA(UsdPhysics.RevoluteJoint):
            jtype = "revolute"
            rev = UsdPhysics.RevoluteJoint(prim)
            axis_attr = rev.GetAxisAttr()
            if axis_attr:
                a = axis_attr.Get()
                if a is not None:
                    # USD may use token "X", "Y", "Z" or vector (Gf.Vec3f)
                    try:
                        axis = (float(a[0]), float(a[1]), float(a[2]))
                    except (TypeError, ValueError, IndexError):
                        tok = str(a).upper()
                        axis = {"X": (1, 0, 0), "Y": (0, 1, 0), "Z": (0, 0, 1)}.get(tok, (0, 0, 1))

        path = str(prim.GetPath())
        name = prim.GetName()
        results.append({
            "path": path,
            "name": name,
            "type": jtype,
            "body0": body0,
            "body1": body1,
            "local_pos0": (float(pos0[0]), float(pos0[1]), float(pos0[2])),
            "local_pos1": (float(pos1[0]), float(pos1[1]), float(pos1[2])),
            "local_rot0": (rot0.GetReal(), rot0.GetImaginary()[0], rot0.GetImaginary()[1], rot0.GetImaginary()[2]),
            "local_rot1": (rot1.GetReal(), rot1.GetImaginary()[0], rot1.GetImaginary()[1], rot1.GetImaginary()[2]),
            "axis": axis,
        })
    return results


def _normalize_name(name):
    """Normalize link/joint name for matching: URDF 'ati/base_link' -> 'ati_base_link'."""
    return name.replace("/", "_").replace("-", "_").lower()


def _find_usd_link_for_urdf(usd_links, urdf_link):
    """Find the best matching USD prim for a URDF link."""
    urdf_norm = _normalize_name(urdf_link)
    # Exact normalized match
    for u in usd_links:
        if _normalize_name(u["name"]) == urdf_norm:
            return u
    # Fallback: USD_TO_URDF_LINK mapping (for legacy names like base->base_link_inertia)
    for u in usd_links:
        usd_norm = _normalize_name(u["name"])
        mapped = USD_TO_URDF_LINK.get(usd_norm)
        if mapped and _normalize_name(mapped) == urdf_norm:
            return u
    return None


def _find_usd_joint_for_urdf(usd_joints, urdf_joint):
    """Find matching USD joint for a URDF joint by normalized name."""
    urdf_norm = _normalize_name(urdf_joint)
    for u in usd_joints:
        if _normalize_name(u["name"]) == urdf_norm:
            return u
    return None


def main():
    urdf_path = args_cli.urdf or DEFAULT_URDF
    usd_path = args_cli.usd or DEFAULT_USD

    if not os.path.isfile(urdf_path):
        print(f"URDF not found: {urdf_path}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(usd_path):
        print(f"USD not found: {usd_path}", file=sys.stderr)
        sys.exit(1)

    def log(msg=""):
        print(msg, flush=True)

    log("=" * 80)
    log("URDF vs USD Equivalence Check (for gravity comp debugging)")
    log("=" * 80)
    log(f"  URDF: {urdf_path}")
    log(f"  USD:  {usd_path}")
    log()

    # Parse URDF
    urdf_links, urdf_joints = parse_urdf(urdf_path)

    # Parse USD
    stage = Usd.Stage.Open(usd_path)
    usd_links = extract_usd_links(stage)
    usd_joints = extract_usd_joints(stage)

    # --- LINKS: mass, inertia, CoM (all links with inertial) ---
    log("-" * 80)
    log("LINKS (mass, inertia, CoM) — full comparison")
    log("-" * 80)

    for urdf_link in sorted(urdf_links.keys()):
        urdf_data = urdf_links[urdf_link]

        # Find matching USD prim
        usd_match = _find_usd_link_for_urdf(usd_links, urdf_link)

        log(f"  {urdf_link}")
        log(f"    URDF mass: {urdf_data['mass']:.4f} kg")
        log(f"    URDF CoM:  xyz={urdf_data['com_xyz']}")
        log(f"    URDF inertia: ixx={urdf_data['inertia']['ixx']:.6f} iyy={urdf_data['inertia']['iyy']:.6f} izz={urdf_data['inertia']['izz']:.6f}")
        if usd_match:
            mass_ok = abs(usd_match["mass"] - urdf_data["mass"]) < TOL_MASS
            log(f"    USD mass:  {usd_match['mass']:.4f} kg  {'OK' if mass_ok else 'MISMATCH'}")
            if usd_match["diagonal_inertia"]:
                diag = usd_match["diagonal_inertia"]
                log(f"    USD diag_inertia: [{diag[0]:.6f}, {diag[1]:.6f}, {diag[2]:.6f}]")
                # Compare with URDF principal diagonal (URDF is full matrix; diagonal approx)
                urdf_diag = (urdf_data["inertia"]["ixx"], urdf_data["inertia"]["iyy"], urdf_data["inertia"]["izz"])
                diff = [abs(diag[i] - urdf_diag[i]) for i in range(3)]
                log(f"    Inertia diff (USD - URDF diag): {[f'{d:.6f}' for d in diff]}  {'OK' if max(diff) < TOL_INERTIA else 'CHECK'}")
            if usd_match["center_of_mass"]:
                com = usd_match["center_of_mass"]
                if all(math.isfinite(x) for x in com):
                    log(f"    USD CoM:   xyz={com}")
                else:
                    log(f"    USD CoM:   (invalid/undefined)")
        else:
            log(f"    USD: NO MATCHING PRIM")
        log()

    # --- USD links not in URDF ---
    matched_usd_paths = set()
    for urdf_link in urdf_links:
        u = _find_usd_link_for_urdf(usd_links, urdf_link)
        if u:
            matched_usd_paths.add(u["path"])
    unmatched_usd = [u for u in usd_links if u["path"] not in matched_usd_paths]
    if unmatched_usd:
        log("-" * 80)
        log("USD LINKS without URDF match")
        log("-" * 80)
        for u in sorted(unmatched_usd, key=lambda x: x["path"]):
            log(f"  {u['path']}: mass={u['mass']:.4f} kg")
        log()

    # --- JOINTS: origin, axis (all joints) ---
    log("-" * 80)
    log("JOINTS (origin in parent frame, axis) — full comparison")
    log("-" * 80)

    for urdf_joint in sorted(urdf_joints.keys()):
        urdf_data = urdf_joints[urdf_joint]

        # Find matching USD joint
        usd_match = _find_usd_joint_for_urdf(usd_joints, urdf_joint)

        log(f"  {urdf_joint} ({urdf_data['parent']} -> {urdf_data['child']}) type={urdf_data['type']}")
        log(f"    URDF origin_xyz: {urdf_data['origin_xyz']}")
        log(f"    URDF origin_rpy: {urdf_data['origin_rpy']}")
        if usd_match:
            # USD LocalPos0 is joint frame in body0 (parent) frame
            pos0 = usd_match["local_pos0"]
            log(f"    USD localPos0:  {pos0}")
            diff_xyz = [abs(pos0[i] - urdf_data["origin_xyz"][i]) for i in range(3)]
            log(f"    Pos diff (USD pos0 - URDF xyz): {[f'{d:.6f}' for d in diff_xyz]}  {'OK' if max(diff_xyz) < TOL_POS else 'MISMATCH'}")
            if urdf_data["type"] in ("revolute", "prismatic"):
                log(f"    URDF axis:       {urdf_data['axis']}")
                log(f"    USD axis:        {usd_match['axis']}")
                axis_diff = [abs(usd_match["axis"][i] - urdf_data["axis"][i]) for i in range(3)]
                log(f"    Axis diff: {[f'{d:.6f}' for d in axis_diff]}  {'OK' if max(axis_diff) < TOL_POS else 'CHECK'}")
        else:
            log(f"    USD: NO MATCHING JOINT")
        log()

    # --- USD joints not in URDF ---
    matched_usd_joint_paths = set()
    for urdf_joint in urdf_joints:
        u = _find_usd_joint_for_urdf(usd_joints, urdf_joint)
        if u:
            matched_usd_joint_paths.add(u["path"])
    unmatched_usd_joints = [u for u in usd_joints if u["path"] not in matched_usd_joint_paths]
    if unmatched_usd_joints:
        log("-" * 80)
        log("USD JOINTS without URDF match")
        log("-" * 80)
        for u in sorted(unmatched_usd_joints, key=lambda x: x["path"]):
            log(f"  {u['path']}: type={u['type']}")
        log()

    # --- Summary: total mass ---
    log("-" * 80)
    log("TOTAL MASS")
    log("-" * 80)
    urdf_total_mass = sum(d["mass"] for d in urdf_links.values())
    usd_total_mass = sum(u["mass"] for u in usd_links)
    log(f"  URDF (all links with inertial): {urdf_total_mass:.3f} kg")
    log(f"  USD (all MassAPI prims):        {usd_total_mass:.3f} kg")
    log()

    # --- List all USD prims/joints for reference ---
    log("-" * 80)
    log("USD PRIMS WITH MassAPI (all)")
    log("-" * 80)
    for u in sorted(usd_links, key=lambda x: x["path"]):
        log(f"  {u['path']}: mass={u['mass']:.4f}")
    log()
    log("-" * 80)
    log("USD JOINTS (all)")
    log("-" * 80)
    for u in sorted(usd_joints, key=lambda x: x["path"]):
        log(f"  {u['path']}: type={u['type']} body0={u['body0']} body1={u['body1']}")
        log(f"    localPos0={u['local_pos0']} localPos1={u['local_pos1']}")
    log()

    log("=" * 80)
    log("Done. Use this to identify mass/inertia/joint differences affecting gravity comp.")
    log("=" * 80)

    simulation_app.close()


if __name__ == "__main__":
    main()
