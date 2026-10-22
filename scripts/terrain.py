"""Deterministic static terrain with matching visible and collision geometry."""
import numpy as np
import trimesh


def hill_mesh(spec):
    width, length = spec["size"]
    xs = np.linspace(-width/2, width/2, round(width/spec["grid_spacing"])+1)
    ys = np.linspace(-length/2, length/2, round(length/spec["grid_spacing"])+1)
    x, y = np.meshgrid(xs, ys, indexing="ij")
    # Multiple gently joined bumps. A small central plateau supports a repeatable drop test.
    edge = np.sin(np.pi*(x/width+0.5))**2 * np.sin(np.pi*(y/length+0.5))**2
    ripple = 0.65 + 0.35*np.cos(2*np.pi*y/0.9)*np.cos(2*np.pi*x/0.8)
    z = spec["max_height"] * edge * ripple
    plateau = (np.abs(x) <= 0.20+1e-6) & (np.abs(y) <= 0.20+1e-6)
    z[plateau] = spec["max_height"]
    vertices = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    faces = []
    for i in range(len(xs)-1):
        for j in range(len(ys)-1):
            a = i*len(ys)+j
            b = a+len(ys)
            faces.extend(((a, b, a+1), (a+1, b, b+1)))
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def stone_mesh(radius, height, rng):
    angle = np.arange(6)*np.pi/3 + rng.uniform(0, np.pi/3)
    radii = radius*rng.uniform(0.8, 1.0, 6)
    xy = np.column_stack((radii*np.cos(angle), radii*np.sin(angle)))
    bottom = np.column_stack((xy, np.zeros(6)))
    top = np.column_stack((xy*0.70, np.full(6, height)))
    return trimesh.convex.convex_hull(np.concatenate((bottom, top)))


def gravel_layout(spec, seed):
    rng = np.random.default_rng(seed)
    # Reserve one broad flat-topped central stone for the collision drop check.
    stones = [(0.0, 0.0, spec["radius_range"][1], spec["height_range"][1])]
    for _ in range(10000):
        if len(stones) >= spec["stone_count_per_patch"]:
            break
        radius = rng.uniform(*spec["radius_range"])
        x, y = rng.uniform(-np.asarray(spec["size"])/2+radius, np.asarray(spec["size"])/2-radius)
        if all(np.hypot(x-s[0], y-s[1]) > radius+s[2]+0.015 for s in stones):
            stones.append((float(x), float(y), radius, rng.uniform(*spec["height_range"])))
    if len(stones) != spec["stone_count_per_patch"]:
        raise ValueError("Gravel patch too dense for requested stone count")
    return stones


def build_terrain(mode, spec):
    import isaaclab.sim as sim_utils
    from isaaclab.terrains.utils import create_prim_from_mesh
    from pxr import UsdGeom, UsdPhysics
    from isaaclab.sim.utils import get_current_stage

    stage = get_current_stage()
    material = sim_utils.RigidBodyMaterialCfg(**spec["physics"])
    probes, ray_targets, mesh_paths = [], [], []

    def spawn(path, mesh, position, color, approximation="none"):
        create_prim_from_mesh(path, mesh, translation=tuple(position), physics_material=material,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(color), roughness=0.95))
        prim = stage.GetPrimAtPath(path+"/mesh")
        # Hills keep their triangles. Stones are already convex, so their hull matches the visible shape.
        UsdGeom.Mesh(prim).GetSubdivisionSchemeAttr().Set("none")
        UsdGeom.Mesh(prim).CreateDoubleSidedAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(approximation)
        mesh_paths.append(path+"/mesh")

    if mode in ("rough", "undulating"):
        cfg = spec["undulating"]
        mesh = hill_mesh(cfg)
        for i, center in enumerate(cfg["centers"]):
            path = f"/World/Terrain/Hill{i}"
            spawn(path, mesh, (*center, 0), cfg["color"])
            name = f"Hill{i}"
            probes.append((name, center, cfg["max_height"], 0.04))
            ray_targets.append((name, center, cfg["max_height"], path+"/mesh"))
    if mode in ("rough", "gravel"):
        cfg = spec["gravel"]
        stones = gravel_layout(cfg, spec["seed"])
        # Rotate a single generated patch by 180 degrees for the opposite side.
        for i, center in enumerate(cfg["centers"]):
            sign = 1 if i % 2 == 0 else -1
            bed = trimesh.creation.box(extents=(*cfg["size"], cfg["bed_height"]))
            spawn(f"/World/Terrain/Gravel{i}/Bed", bed, (*center, cfg["bed_height"]/2), cfg["bed_color"])
            rng = np.random.default_rng(spec["seed"]+1)
            for j, (x, y, radius, height) in enumerate(stones):
                mesh = stone_mesh(radius, height, rng)
                mesh.vertices[:, :2] *= sign
                color = np.clip(np.asarray(cfg["stone_color"])*rng.uniform(0.75, 1.25), 0, 1)
                path = f"/World/Terrain/Gravel{i}/Stone{j:03d}"
                xy = (center[0]+sign*x, center[1]+sign*y)
                spawn(path, mesh, (*xy, cfg["bed_height"]), color, approximation="convexHull")
                if j == 0:
                    name = f"Stone{i}"
                    z = cfg["bed_height"]+height
                    probes.append((name, xy, z, 0.025))
                    ray_targets.append((name, xy, z, path+"/mesh"))
    return {"probes": probes, "ray_targets": ray_targets, "mesh_paths": mesh_paths,
            "mode": mode, "seed": spec["seed"], "stone_count": len(stones)*len(spec["gravel"]["centers"]) if mode in ("rough", "gravel") else 0}
