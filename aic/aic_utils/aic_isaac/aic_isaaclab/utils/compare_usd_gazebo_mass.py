#!/usr/bin/env python3
"""Compare mass/inertia from AIC USD (Isaac Lab) with Gazebo UR5e (ur_description).

Uses pxr to read aic_arm_only.usd. Gazebo uses ur_description physical_parameters.
Run with Isaac Lab python: ./scripts/run-isaaclab.sh this_script --headless
"""

from __future__ import annotations

import argparse
import os
import sys

# pxr requires Isaac Sim; use AppLauncher for headless
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Compare USD vs Gazebo mass/inertia")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import Usd, UsdPhysics

# Paths
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AIC_ASSET_DIR = os.path.join(
    _SCRIPT_DIR,
    "../source/aic_task/aic_task/tasks/manager_based/aic_task/Intrinsic_assets",
)
USD_PATH = os.path.join(AIC_ASSET_DIR, "aic_arm_only.usd")

# UR5e link masses from ur_description physical_parameters.yaml (Gazebo)
# Source: Universal_Robots_ROS2_Description config/ur5e/physical_parameters.yaml
GAZEBO_UR5E_LINK_MASSES = {
    "base_link": 4.0,
    "shoulder_link": 3.761,
    "upper_arm_link": 8.058,
    "forearm_link": 2.846,
    "wrist_1_link": 1.37,
    "wrist_2_link": 1.3,
    "wrist_3_link": 0.365,
    # Gripper / tool links (not in ur_description arm)
    "ee_link": 0.0,
    "tool0": 0.0,
    "gripper_tcp": 0.0,
}

# Gazebo aic_assets: gripper, ATI, cameras (ur_gz.urdf.xacro includes these)
# Source: aic_assets models/*.xacro
GAZEBO_EXTRA_MASSES = {
    "ati_base_link": 0.36,  # Axia80 M20
    "ati_tool_link": 0.001,
    "gripper_hande_base_link": 1.0,  # Robotiq Hand-E
    "gripper_hande_finger_link_l": 0.001,
    "gripper_hande_finger_link_r": 0.001,
    "cam_mount_cam_mount_link": 0.001,
    # 3x Basler cameras: camera_link + optical + sensor_link each
    "center_camera_camera_link": 0.001,
    "center_camera_optical": 0.001,
    "center_camera_sensor_link": 0.001,
    "left_camera_camera_link": 0.001,
    "left_camera_optical": 0.001,
    "left_camera_sensor_link": 0.001,
    "right_camera_camera_link": 0.001,
    "right_camera_optical": 0.001,
    "right_camera_sensor_link": 0.001,
}

# Map USD body names to Gazebo link names (best-effort)
USD_TO_GAZEBO = {
    "base": "base_link",
    "shoulder": "shoulder_link",
    "upper_arm": "upper_arm_link",
    "forearm": "forearm_link",
    "wrist_1": "wrist_1_link",
    "wrist_2": "wrist_2_link",
    "wrist_3": "wrist_3_link",
    "ee_link": "ee_link",
    "tool0": "tool0",
    "gripper": "gripper_tcp",
    "gripper_tcp": "gripper_tcp",
}


def _match_gazebo_link(prim_path: str) -> str | None:
    """Map USD prim path/name to Gazebo link name."""
    name = prim_path.split("/")[-1].lower()
    for key, gazebo in USD_TO_GAZEBO.items():
        if key in name:
            return gazebo
    return None


def extract_usd_mass(stage: Usd.Stage) -> list[dict]:
    """Extract mass and inertia from all prims with MassAPI."""
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
        if mass is None or mass == 0:
            # Try density-based; skip for now
            continue
        diag_inertia = None
        diag_attr = mass_api.GetDiagonalInertiaAttr()
        if diag_attr:
            val = diag_attr.Get()
            if val is not None:
                diag_inertia = [float(val[0]), float(val[1]), float(val[2])]
        path = str(prim.GetPath())
        results.append({
            "path": path,
            "name": prim.GetName(),
            "mass": float(mass),
            "diagonal_inertia": diag_inertia,
        })
    return results


def main():
    out_path = os.environ.get("COMPARE_MASS_OUT", "/tmp/compare_usd_gazebo_mass.txt")
    out = open(out_path, "w")

    def log(msg=""):
        print(msg, flush=True)
        out.write(msg + "\n")
        out.flush()

    if not os.path.isfile(USD_PATH):
        log(f"USD not found: {USD_PATH}")
        sys.exit(1)

    stage = Usd.Stage.Open(USD_PATH)
    usd_data = extract_usd_mass(stage)

    if not usd_data:
        log("No prims with explicit mass (MassAPI) found in USD.")
        log("The USD may use density-based mass. Listing all prims with MassAPI:")
        for prim in stage.Traverse():
            if prim.IsValid() and prim.HasAPI(UsdPhysics.MassAPI):
                mass_api = UsdPhysics.MassAPI(prim)
                m = mass_api.GetMassAttr().Get() if mass_api.GetMassAttr() else None
                d = mass_api.GetDensityAttr().Get() if mass_api.GetDensityAttr() else None
                log(f"  {prim.GetPath()}: mass={m}, density={d}")
        log("\nGazebo ur_description masses (reference):")
        for k, v in GAZEBO_UR5E_LINK_MASSES.items():
            if isinstance(v, (int, float)):
                log(f"  {k}: {v} kg")
        out.close()
        simulation_app.close()
        return

    log("=" * 70)
    log("AIC USD (aic_arm_only.usd) — mass & inertia via pxr")
    log("=" * 70)
    log(f"  File: {USD_PATH}\n")

    total_usd = 0.0
    for r in sorted(usd_data, key=lambda x: x["path"]):
        total_usd += r["mass"]
        diag = r["diagonal_inertia"]
        diag_str = (
            f"[{diag[0]:.6f}, {diag[1]:.6f}, {diag[2]:.6f}]" if diag else "—"
        )
        prim_name = r["path"].split("/")[-1].lower()
        # Check extra masses first (ati, gripper, cameras) to avoid "base" matching ati_base_link
        gz_mass = GAZEBO_EXTRA_MASSES.get(prim_name)
        if gz_mass is None:
            gazebo_link = _match_gazebo_link(r["path"])
            gz_mass = GAZEBO_UR5E_LINK_MASSES.get(gazebo_link) if gazebo_link else None
        if isinstance(gz_mass, (int, float)):
            diff = r["mass"] - gz_mass
            diff_str = f"  (Δ={diff:+.3f} vs Gazebo)"
        else:
            diff_str = ""
        log(f"  {r['path']}")
        log(f"    mass (kg): {r['mass']:.4f}  {diff_str}")
        log(f"    diagonal_inertia: {diag_str}")
        log()

    log(f"  Total mass (USD): {total_usd:.3f} kg")
    gz_arm = sum(
        m
        for k, m in GAZEBO_UR5E_LINK_MASSES.items()
        if isinstance(m, (int, float)) and "link" in k and k != "base_link"
    )
    gz_arm_plus_base = sum(
        m for k, m in GAZEBO_UR5E_LINK_MASSES.items() if isinstance(m, (int, float))
    )
    gz_extra = sum(GAZEBO_EXTRA_MASSES.values())
    gz_full = gz_arm_plus_base + gz_extra
    log(f"  Total arm mass (Gazebo ur_description, excl. base): {gz_arm:.3f} kg")
    log(f"  Total arm+base (Gazebo ur_description): {gz_arm_plus_base:.3f} kg")
    log(f"  Total gripper+ATI+cameras (Gazebo aic_assets): {gz_extra:.3f} kg")
    log(f"  Total full robot (Gazebo ur_gz.urdf.xacro): {gz_full:.3f} kg")
    log()

    log("=" * 70)
    log("Gazebo UR5e link masses (ur_description physical_parameters.yaml)")
    log("=" * 70)
    for link, mass in GAZEBO_UR5E_LINK_MASSES.items():
        if isinstance(mass, (int, float)):
            log(f"  {link}: {mass} kg")
    log()
    log("Gazebo extra (aic_assets: gripper, ATI, cameras):")
    for link, mass in GAZEBO_EXTRA_MASSES.items():
        log(f"  {link}: {mass} kg")
    log()

    # Summary
    log("=" * 70)
    log("Summary")
    log("=" * 70)
    if abs(total_usd - gz_full) < 0.05:
        log(f"  MASS MATCH: USD={total_usd:.2f} kg vs Gazebo full={gz_full:.2f} kg")
    elif abs(total_usd - gz_full) < 0.5:
        log(f"  Mass totals are close: USD={total_usd:.2f} kg, Gazebo full={gz_full:.2f} kg")
    else:
        log(f"  MASS MISMATCH: USD total={total_usd:.2f} kg vs Gazebo full={gz_full:.2f} kg")
        log(f"  Ratio (USD/Gazebo full): {total_usd / gz_full:.3f}")

    log(f"\nOutput written to: {out_path}")
    out.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
