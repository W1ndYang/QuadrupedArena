"""Local-asset Isaac Lab arena, GPU physics checks and optional Go2 standing probe."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import traceback

from isaaclab.app import AppLauncher

project = Path(__file__).resolve().parents[1]
from arena_paths import workspace_root
workspace = workspace_root()
parser = argparse.ArgumentParser()
parser.add_argument("--run-dir", type=Path, required=True)
parser.add_argument("--config", type=Path, default=project / "configs/arena_v0.json")
parser.add_argument("--terrain", choices=["flat", "undulating", "gravel", "rough"], default="flat")
parser.add_argument("--terrain-config", type=Path, default=project / "configs/terrain_v1.json")
parser.add_argument("--steps", type=int, default=240)
parser.add_argument("--empty", action="store_true", help="Only floor, lighting and physics probe")
parser.add_argument("--capture", action="store_true")
parser.add_argument("--beauty", action="store_true", help="One 1280x960 path-traced image, PBR materials, hidden probes")
parser.add_argument("--interactive", action="store_true", help="Continue viewing after finite validation")
parser.add_argument("--robot", choices=["go2"], help="Optional existing asset as an engineering test, no gait policy")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.beauty:
    args.capture = True
if args.steps < 240:
    parser.error("--steps must be >=240 (two seconds for contact checks)")
if args.capture:
    args.enable_cameras = True
config_bytes = args.config.read_bytes()
config = json.loads(config_bytes)
terrain_bytes = args.terrain_config.read_bytes() if args.terrain != "flat" and not args.empty else None
terrain_config = json.loads(terrain_bytes) if terrain_bytes else None
if config["units"] != "meters" or config["up_axis"] != "Z":
    parser.error("Expected meters and Z-up")
app = AppLauncher(args).app

# Omniverse modules must be imported after startup.
import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg, Articulation
from pxr import UsdGeom


def box(path, size, position, color):
    cfg = sim_utils.CuboidCfg(
        size=tuple(size), collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(color)),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.7),
    )
    cfg.func(path, cfg, translation=tuple(position))


def marker(path, xy, radius, color):
    cfg = sim_utils.CylinderCfg(radius=radius, height=0.008,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color))
    cfg.func(path, cfg, translation=(*xy, 0.006))


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; GPU smoke test cannot pass")
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1/120, device=args.device, enable_scene_query_support=True))
    UsdGeom.SetStageMetersPerUnit(sim.stage, 1.0)
    UsdGeom.SetStageUpAxis(sim.stage, UsdGeom.Tokens.z)
    floor = config["floor_size"]
    box("/World/Floor", floor, (0, 0, -floor[2]/2), (0.22, 0.26, 0.29))
    terrain = None
    if terrain_config:
        from terrain import build_terrain
        terrain = build_terrain(args.terrain, terrain_config)
    light = sim_utils.DomeLightCfg(intensity=1800, color=(0.85, 0.9, 1.0))
    light.func("/World/Light", light)
    covers = [] if args.empty else config["covers"]
    for cover in covers:
        height = cover["size"][2]
        color = (0.30, 0.45, 0.58) if height > 1 else (0.85, 0.55, 0.20)
        box(f"/World/Covers/{cover['name']}", cover["size"], (*cover["center"], height/2), color)
    if not args.empty:
        for point in config["task_points"]:
            marker(f"/World/Tasks/{point['name']}", point["center"], point["radius"], (0.15, 0.65, 0.45))
        for i, point in enumerate(config["spawn_points"]):
            marker(f"/World/Spawns/S{i+1}", point, 0.35, (0.65, 0.25, 0.55))
    camera = config["camera"]
    sim.set_camera_view(camera["eye"], camera["target"])
    scene_dir = workspace / "simulators/scenes/quadruped-arena" / args.run_dir.name
    scene_dir.mkdir(parents=True, exist_ok=True)
    scene_path = scene_dir / ("empty.usda" if args.empty else f"arena_{args.terrain}.usda")
    # Export the clean reusable arena before adding validation probes or a robot.
    sim.stage.Export(str(scene_path))
    probes = []
    targets = [("Floor", (0.0, 0.0), 0.0, 0.1)]
    if covers:
        targets += [(c["name"], c["center"], c["size"][2], 0.1) for c in (covers[0], covers[4])]
    if terrain:
        targets += terrain["probes"]
    for name, xy, surface, radius in targets:
        # A small flat-bottomed block is a stable foot-contact probe on the stone top.
        # Spherical probes can roll off small support surfaces after impact.
        spawn_cls = sim_utils.CuboidCfg if name.startswith("Stone") else sim_utils.SphereCfg
        shape = {"size": (2*radius, 2*radius, 2*radius)} if name.startswith("Stone") else {"radius": radius}
        cfg = RigidObjectCfg(prim_path=f"/World/Probes/{name}",
            spawn=spawn_cls(**shape,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.1, 0.1))),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(*xy, surface+(0.15 if name.startswith("Stone") else 0.8))))
        probes.append((name, RigidObject(cfg), surface+radius))
    robot = None
    if args.robot:
        from unitree_rl_lab.assets.robots.unitree import UNITREE_GO2_CFG
        robot_cfg = UNITREE_GO2_CFG.copy()
        robot_cfg.prim_path = "/World/Robot"
        # Place nominal feet near the ground to avoid a 0.4 m drop-start impact.
        robot_cfg.init_state.pos = (*config["spawn_points"][0], 0.30)
        robot_cfg.spawn.usd_dir = str(workspace / "simulators/assets/quadruped-arena/go2")
        robot = Articulation(robot_cfg)
    annotator = product = None
    if args.capture and not args.beauty:
        import omni.replicator.core as rep
        cam = rep.create.camera(position=tuple(camera["eye"]), look_at=tuple(camera["target"]))
        product = rep.create.render_product(cam, tuple(camera["resolution"]))
        annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        annotator.attach([product])
    sim.reset()
    ray_checks = {}
    if terrain:
        import omni.physx
        # Rays start below the validation probes and above each target surface.
        query = omni.physx.get_physx_scene_query_interface()
        for name, xy, expected_z, expected_path in terrain["ray_targets"]:
            hit = query.raycast_closest((xy[0]+0.015, xy[1], expected_z+0.08), (0., 0., -1.), 0.3)
            actual_z = float(hit["position"][2]) if hit["hit"] else None
            ray_checks[name] = {"hit": bool(hit["hit"]), "collider": str(hit.get("collision", "")),
                "surface_z_m": actual_z, "expected_surface_z_m": expected_z,
                "passed": bool(hit["hit"] and str(hit["collision"]) == expected_path and abs(actual_z-expected_z)<0.005)}
    if robot:
        # Config defaults are buffers, not an automatic PhysX joint-state reset.
        root_state = robot.data.default_root_state.clone()
        robot.write_root_pose_to_sim(root_state[:, :7])
        robot.write_root_velocity_to_sim(root_state[:, 7:])
        robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone())
        robot.reset()
    started = time.monotonic()
    robot_min_z = float("inf")
    probe_trajectories = {name: [] for name, _, _ in probes}
    for step in range(args.steps):
        if not app.is_running():
            raise RuntimeError("Simulation closed before validation finished")
        if robot:
            robot.set_joint_position_target(robot.data.default_joint_pos)
            robot.write_data_to_sim()
        sim.step(render=not args.headless or args.capture)
        for name, probe, _ in probes:
            probe.update(sim.get_physics_dt())
            if step % 4 == 0:
                probe_trajectories[name].append([step*sim.get_physics_dt(), *probe.data.root_pos_w[0].tolist()])
        if robot:
            robot.update(sim.get_physics_dt())
            robot_min_z = min(robot_min_z, float(robot.data.root_pos_w[0, 2]))
    checks = {}
    for name, probe, expected in probes:
        z = float(probe.data.root_pos_w[0, 2])
        origin = next(t[1] for t in targets if t[0] == name)
        position = probe.data.root_pos_w[0].tolist()
        drift = float(np.linalg.norm(np.asarray(position[:2])-origin))
        tolerance = 0.01 if name.startswith(("Hill", "Stone")) else 0.035
        checks[name] = {"z_m": z, "expected_z_m": expected, "horizontal_drift_m": drift,
                        "passed": abs(z-expected)<tolerance and drift<0.05}
    report = {
        "status": "passed" if all(v["passed"] for v in [*checks.values(), *ray_checks.values()]) else "failed",
        "gpu": torch.cuda.get_device_name(0), "device": str(sim.device),
        "versions": {n: importlib.metadata.version(n) for n in ("isaacsim", "isaaclab", "torch")},
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "steps": args.steps, "simulation_seconds": args.steps*sim.get_physics_dt(),
        "step_wall_seconds": time.monotonic()-started, "cover_count": len(covers),
        "contacts": checks, "scene": str(scene_path), "robot": args.robot,
        "terrain": {"mode": terrain["mode"], "seed": terrain["seed"], "stone_count": terrain["stone_count"],
                    "collision_mesh_count": len(terrain["mesh_paths"]), "rays": ray_checks,
                    "config_sha256": hashlib.sha256(terrain_bytes).hexdigest()} if terrain else {"mode": "flat"},
    }
    if robot:
        quat = robot.data.root_quat_w[0]
        up_z = float(1 - 2 * (quat[1]**2 + quat[2]**2))
        xy = robot.data.root_pos_w[0, :2]
        drift = float(torch.linalg.vector_norm(xy - torch.tensor(config["spawn_points"][0], device=xy.device)))
        report["robot_probe"] = {"joint_count": robot.num_joints,
            "root_position_m": robot.data.root_pos_w[0].tolist(), "min_root_z_m": robot_min_z,
            "body_up_z": up_z, "horizontal_drift_m": drift,
            "finite": bool(torch.isfinite(robot.data.joint_pos).all()),
            "control": "constant nominal joint-position PD; no walking policy"}
        if robot_min_z < 0.18 or up_z < 0.8 or drift > 0.3 or not report["robot_probe"]["finite"]:
            report["status"] = "failed"
    if args.beauty:
        import omni.replicator.core as rep
        from beauty import prepare
        report["beauty"] = prepare(sim.stage, scene_dir / "textures")
        cam = rep.create.camera(position=tuple(camera["eye"]), look_at=tuple(camera["target"]), focal_length=28.)
        product = rep.create.render_product(cam, (1280, 960))
        annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        annotator.attach([product])
        report["beauty"]["resolution"] = [1280, 960]
        beauty_scene = scene_dir / "arena_beauty.usda"
        sim.stage.Export(str(beauty_scene))
        report["beauty"]["scene"] = str(beauty_scene)
    if annotator:
        from PIL import Image
        for _ in range(64 if args.beauty else 20):
            sim.render()
        pixels = annotator.get_data()
        if pixels.size == 0 or np.std(pixels[..., :3]) < 1:
            raise RuntimeError("No usable rendered image")
        image_path = args.run_dir / "overview.png"
        Image.fromarray(pixels).save(image_path)
        report["image"] = str(image_path)
        annotator.detach([product.path])
        if terrain and not args.beauty:
            # Sequential render products keep memory usage down on the 6GB GPU.
            product.destroy()
            detail = terrain_config["detail_camera"]
            cam = rep.create.camera(position=tuple(detail["eye"]), look_at=tuple(detail["target"]))
            product = rep.create.render_product(cam, tuple(detail["resolution"]))
            annotator.attach([product])
            for _ in range(25):
                sim.render()
            pixels = annotator.get_data()
            if pixels.size == 0 or np.std(pixels[..., :3]) < 1:
                raise RuntimeError("No usable terrain detail image")
            detail_path = args.run_dir / "terrain_detail.png"
            Image.fromarray(pixels).save(detail_path)
            report["detail_image"] = str(detail_path)
            annotator.detach([product.path])
    (args.run_dir / "result.json").write_text(json.dumps(report, indent=2)+"\n")
    (args.run_dir / "arena_config.json").write_bytes(config_bytes)
    (args.run_dir / "probe_trajectories.json").write_text(json.dumps(probe_trajectories)+"\n")
    if terrain_bytes:
        (args.run_dir / "terrain_config.json").write_bytes(terrain_bytes)
    print("ARENA_RESULT " + json.dumps(report), flush=True)
    if report["status"] != "passed":
        raise RuntimeError("Physics validation failed; inspect result.json")
    while args.interactive and app.is_running():
        if robot:
            robot.set_joint_position_target(robot.data.default_joint_pos)
            robot.write_data_to_sim()
        sim.step()
        if robot:
            robot.update(sim.get_physics_dt())


try:
    main()
except BaseException:
    # SimulationApp may terminate the interpreter while closing; persist errors first.
    error = traceback.format_exc()
    print(error, flush=True)
    (args.run_dir / "error.txt").write_text(error)
    result_path = args.run_dir / "result.json"
    failure = json.loads(result_path.read_text()) if result_path.exists() else {}
    failure.update(status="failed", error=error)
    result_path.write_text(json.dumps(failure, indent=2)+"\n")
    raise
finally:
    sim = sim_utils.SimulationContext.instance()
    if sim is not None:
        # Lab 2.3.2's standalone STOP callback waits for timeline resume.
        # Unsubscribe it through the public lifecycle API before stopping.
        sim.clear_all_callbacks()
        sim.clear_instance()
        sim.stop()
    # Images are read and saved synchronously; there are no background writers.
    app.close(wait_for_replicator=False)
