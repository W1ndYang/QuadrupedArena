"""Render-only material/light overrides; never modify collision geometry."""
from pathlib import Path


def make_textures(directory):
    """Deterministic procedural PBR inputs, not a generated final scene image."""
    import numpy as np
    from PIL import Image

    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261019)
    for kind in ("concrete", "earth", "stone"):
        height = np.zeros((512, 512), dtype=np.float32)
        for grid, weight in ((8, .025), (32, .035), (128, .055), (512, .07)):
            layer = rng.random((grid, grid)).astype(np.float32)
            height += np.asarray(Image.fromarray(layer).resize((512, 512), Image.Resampling.BICUBIC)) * weight
        height -= height.mean()
        base = {"concrete": .64, "earth": .46, "stone": .48}[kind]
        albedo = np.clip(base + height * (1.2 if kind == "earth" else .65), .05, .85)
        pores = rng.random((512, 512)) < (.035 if kind == "earth" else .006)
        albedo[pores] *= .75
        Image.fromarray((albedo * 255).astype(np.uint8)).convert("RGB").save(directory / f"{kind}_color.png")
        dy, dx = np.gradient(height)
        normal = np.stack((-dx * 2, -dy * 2, np.ones_like(dx)), axis=-1)
        normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
        Image.fromarray(((normal * .5 + .5) * 255).astype(np.uint8)).save(directory / f"{kind}_normal.png")


def prepare(stage, texture_dir):
    import carb
    import isaacsim
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade

    installation = Path(isaacsim.__file__).parent
    textures = texture_dir
    make_textures(textures)
    mdl = installation / "kit/mdl/core/Base/OmniPBR.mdl"

    def material(name, tint, scale=1., bump=0.35, roughness=0.9, texture="concrete"):
        mat = UsdShade.Material.Define(stage, f"/World/BeautyMaterials/{name}")
        shader = UsdShade.Shader.Define(stage, str(mat.GetPath()) + "/Shader")
        shader.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
        shader.SetSourceAsset(Sdf.AssetPath(str(mdl)), "mdl")
        shader.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
        shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
        mat.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
        values = {
            "diffuse_texture": (Sdf.ValueTypeNames.Asset, Sdf.AssetPath(str(textures / f"{texture}_color.png"))),
            "normalmap_texture": (Sdf.ValueTypeNames.Asset, Sdf.AssetPath(str(textures / f"{texture}_normal.png"))),
            "diffuse_tint": (Sdf.ValueTypeNames.Color3f, Gf.Vec3f(*tint)),
            "reflection_roughness_constant": (Sdf.ValueTypeNames.Float, roughness),
            "bump_factor": (Sdf.ValueTypeNames.Float, bump),
            "project_uvw": (Sdf.ValueTypeNames.Bool, True),
            "world_or_object": (Sdf.ValueTypeNames.Bool, True),
            "texture_scale": (Sdf.ValueTypeNames.Float2, Gf.Vec2f(scale)),
        }
        for key, (kind, value) in values.items():
            shader.CreateInput(key, kind).Set(value)
        return mat

    materials = {
        "floor": material("Concrete", (0.68, 0.70, 0.72), 0.6),
        "high": material("BlueConcrete", (0.48, 0.65, 0.77), 0.8, 0.15),
        "low": material("OchreConcrete", (0.88, 0.66, 0.32), 0.8, 0.15),
        "soil": material("Earth", (0.85, 0.60, 0.32), 2.0, 0.5, 1., "earth"),
        "stone": material("Stone", (0.85, 0.81, 0.74), 3.0, 0.4, 0.9, "stone"),
    }
    assignments = {}
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Gprim):
            continue
        path = str(prim.GetPath())
        key = None
        if path.startswith("/World/Floor/"):
            key = "floor"
        elif path.startswith("/World/Covers/"):
            key = "high" if "/High" in path else "low"
        elif path.startswith("/World/Terrain/"):
            key = "stone" if "/Stone" in path else "soil"
        if key:
            # Bind only the visual purpose; retain the original physics material.
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(materials[key], UsdShade.Tokens.strongerThanDescendants)
            assignments[path] = key
        if path.startswith("/World/Terrain/Hill") and prim.IsA(UsdGeom.Mesh):
            # Smooth shading only; the collision vertices and faces stay unchanged.
            import numpy as np
            from pxr import Vt
            mesh = UsdGeom.Mesh(prim)
            points = np.asarray(mesh.GetPointsAttr().Get())
            faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
            normal = np.zeros_like(points)
            fn = np.cross(points[faces[:, 1]] - points[faces[:, 0]], points[faces[:, 2]] - points[faces[:, 0]])
            for i in range(3):
                np.add.at(normal, faces[:, i], fn)
            normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-8)
            mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(normal.astype(np.float32)))
            mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)

    probes = stage.GetPrimAtPath("/World/Probes")
    if probes.IsValid():
        for prim in Usd.PrimRange(probes):
            if prim.IsA(UsdGeom.Imageable):
                UsdGeom.Imageable(prim).MakeInvisible()
    dome = UsdLux.DomeLight(stage.GetPrimAtPath("/World/Light"))
    dome.GetIntensityAttr().Set(650.)
    dome.GetColorAttr().Set(Gf.Vec3f(0.85, 0.91, 1.))
    sun = UsdLux.DistantLight.Define(stage, "/World/BeautySun")
    sun.CreateIntensityAttr(1600.)
    sun.CreateAngleAttr(6.)
    sun.CreateColorAttr(Gf.Vec3f(1., 0.95, 0.87))
    UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-25., -35., -25.))

    settings = carb.settings.get_settings()
    settings.set("/rtx/rendermode", "PathTracing")
    settings.set("/rtx/pathtracing/spp", 8)
    settings.set("/rtx/pathtracing/totalSpp", 256)
    settings.set("/rtx/pathtracing/clampSpp", 256)
    settings.set("/rtx/pathtracing/maxBounces", 6)
    settings.set("/rtx/pathtracing/optixDenoiser/enabled", True)
    settings.set("/rtx/post/histogram/enabled", False)
    return {"renderer": settings.get("/rtx/rendermode"), "target_samples_per_pixel": 256,
            "samples_per_update": 8, "max_bounces": 6, "denoiser": True,
            "hidden_prim": "/World/Probes", "visual_materials": assignments,
            "texture_source": str(textures), "note": "Deterministic procedural albedo and normal maps; no AI image generation."}
