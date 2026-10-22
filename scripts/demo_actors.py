"""Visual-only URDF forward kinematics for an explicitly scripted presentation."""
import math
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np


def transform(xyz=(0, 0, 0), rpy=(0, 0, 0)):
    from scipy.spatial.transform import Rotation
    m = np.eye(4)
    m[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    m[:3, 3] = xyz
    return m


def origin(element):
    if element is None:
        return np.eye(4)
    return transform([float(x) for x in element.get("xyz", "0 0 0").split()],
                     [float(x) for x in element.get("rpy", "0 0 0").split()])


class Actor:
    def __init__(self, stage, path, kind, workspace):
        from pxr import Usd, UsdGeom, UsdShade, Sdf, Gf, Vt
        import trimesh
        self.kind, self.stage = kind, stage
        from arena_paths import dependency_path
        robot_dir = dependency_path('unitree_ros') / 'robots'
        urdf = robot_dir / {"go2":"go2_description/urdf/go2_description.urdf", "go2w":"go2w_description/urdf/go2w_description.urdf", "g1":"g1_description/g1_23dof.urdf"}[kind]
        root = ET.parse(urdf).getroot()
        self.joints = []
        links = [link.get("name") for link in root.findall("link")]
        children = set()
        for joint in root.findall("joint"):
            child = joint.find("child").get("link")
            children.add(child)
            axis = joint.find("axis")
            self.joints.append((joint.get("name"), joint.find("parent").get("link"), child,
                                origin(joint.find("origin")), np.fromstring(axis.get("xyz", "0 0 1"), sep=" ") if axis is not None else np.array([0., 0., 1.])))
        self.root_link = next(link for link in links if link not in children)
        self.ops = {}
        UsdGeom.Xform.Define(stage, path)
        source = workspace / "simulators/assets/quadruped-arena/go2/configuration/go2_description_base.usd"
        if kind == "go2w" or (kind == "go2" and not source.exists()):
            from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
            converted = UrdfConverter(UrdfConverterCfg(asset_path=str(urdf),
                usd_dir=str(workspace/"simulators/assets/quadruped-arena"/kind),
                usd_file_name=f"{kind}_description.usd", fix_base=False, merge_fixed_joints=False,
                joint_drive=UrdfConverterCfg.JointDriveCfg(gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.,damping=0.))))
            source = Path(converted.usd_dir)/f"configuration/{kind}_description_base.usd"
            if not source.exists(): source=Path(converted.usd_path)
        source_stage = Usd.Stage.Open(str(source)) if kind in ("go2","go2w") else None
        visual_paths = {str(p.GetPath()).split("/")[-2]: str(p.GetPath()) for p in source_stage.Traverse() if p.GetName() == "visuals"} if source_stage else {}
        colors = {m.get("name"): tuple(float(v) for v in m.find("color").get("rgba").split()[:3]) for m in root.findall("material") if m.find("color") is not None}
        self.mesh_count = 0
        for link in root.findall("link"):
            name = link.get("name")
            node = UsdGeom.Xform.Define(stage, f"{path}/{name}")
            self.ops[name] = node.AddTransformOp()
            if kind in ("go2","go2w"):
                if link.find("visual") is not None:
                    if name not in visual_paths:
                        raise RuntimeError(f"Missing converted Go2 visual: {name}; available={list(visual_paths)}")
                    visual = stage.DefinePrim(f"{path}/{name}/visuals", "Xform")
                    visual.GetReferences().AddReference(str(source), visual_paths[name])
                    if kind == "go2w":
                        # Imported mesh bindings point outside each link reference. Bind
                        # task-local materials strongly so wheels do not render white.
                        rubber=name.endswith("_foot")
                        light=name=="base"
                        color=(.022,.027,.032) if rubber else (.64,.69,.73) if light else (.10,.13,.16)
                        matname="Rubber" if rubber else "Shell" if light else "Chassis"
                        material=UsdShade.Material.Define(stage,f"{path}/Materials/{matname}")
                        shader=UsdShade.Shader.Define(stage,str(material.GetPath())+"/Shader")
                        shader.CreateIdAttr("UsdPreviewSurface")
                        shader.CreateInput("diffuseColor",Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
                        shader.CreateInput("roughness",Sdf.ValueTypeNames.Float).Set(.85 if rubber else .4)
                        shader.CreateInput("metallic",Sdf.ValueTypeNames.Float).Set(0. if rubber else .3)
                        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),"surface")
                        UsdShade.MaterialBindingAPI.Apply(visual).Bind(material,UsdShade.Tokens.strongerThanDescendants)
                    self.mesh_count += 1
                continue
            for index, visual in enumerate(link.findall("visual")):
                mesh_element = visual.find("geometry/mesh")
                if mesh_element is None:
                    continue
                filename = urdf.parent / mesh_element.get("filename")
                mesh = trimesh.load(str(filename), force="mesh", process=False)
                mesh.apply_scale(np.fromstring(mesh_element.get("scale", "1 1 1"), sep=" "))
                mesh.apply_transform(origin(visual.find("origin")))
                usd = UsdGeom.Mesh.Define(stage, f"{path}/{name}/visual{index}")
                usd.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(mesh.vertices, dtype=np.float32)))
                usd.CreateFaceVertexCountsAttr([3]*len(mesh.faces))
                usd.CreateFaceVertexIndicesAttr(mesh.faces.flatten().tolist())
                usd.CreateSubdivisionSchemeAttr("none")
                usd.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(mesh.vertex_normals, dtype=np.float32)))
                usd.SetNormalsInterpolation("vertex")
                material_element = visual.find("material")
                color = colors.get(material_element.get("name"), (.55, .58, .60)) if material_element is not None else (.55, .58, .60)
                material = UsdShade.Material.Define(stage, f"{path}/Materials/M{len(self.ops)}_{index}")
                shader = UsdShade.Shader.Define(stage, str(material.GetPath())+"/Shader")
                shader.CreateIdAttr("UsdPreviewSurface")
                shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
                shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(.55)
                material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
                UsdShade.MaterialBindingAPI.Apply(usd.GetPrim()).Bind(material)
                self.mesh_count += 1

    def update(self, xy, yaw, phase, moving, ground, wheel_angles=None):
        from scipy.spatial.transform import Rotation
        from pxr import Gf
        values = {}
        wave = math.sin(phase) * min(1., moving/.15)
        if self.kind == "go2w":
            for index,leg in enumerate(("FL","FR","RL","RR")):
                values[f"{leg}_hip_joint"] = 0.
                values[f"{leg}_thigh_joint"] = .8
                values[f"{leg}_calf_joint"] = -1.5
                values[f"{leg}_foot_joint"] = 0. if wheel_angles is None else wheel_angles[index]
            feet = [f"{leg}_foot" for leg in ("FL","FR","RL","RR")]
            sole = .089
        elif self.kind == "go2":
            for leg, sign in (("FL", 1), ("RR", 1), ("FR", -1), ("RL", -1)):
                values[f"{leg}_hip_joint"] = 0.
                values[f"{leg}_thigh_joint"] = .8 + .20*sign*wave
                values[f"{leg}_calf_joint"] = -1.6 - .16*sign*wave
            feet = [f"{leg}_foot" for leg in ("FL", "FR", "RL", "RR")]
            sole = .022
        else:
            for side, sign in (("left", 1), ("right", -1)):
                values[f"{side}_hip_pitch_joint"] = -.10 + .18*sign*wave
                values[f"{side}_knee_joint"] = .20 + .12*max(0., -sign*wave)
                values[f"{side}_ankle_pitch_joint"] = -.10 - .10*sign*wave
                values[f"{side}_shoulder_pitch_joint"] = -.10 - .20*sign*wave
                values[f"{side}_shoulder_roll_joint"] = .12*sign
                values[f"{side}_elbow_joint"] = .35
            feet = ["left_ankle_roll_link", "right_ankle_roll_link"]
            sole = .035
        matrices = {self.root_link: np.eye(4)}
        pending = self.joints.copy()
        while pending:
            progress = False
            for item in pending[:]:
                name, parent, child, base, axis = item
                if parent not in matrices:
                    continue
                motion = np.eye(4)
                if name in values:
                    motion[:3, :3] = Rotation.from_rotvec(axis*values[name]).as_matrix()
                matrices[child] = matrices[parent] @ base @ motion
                pending.remove(item)
                progress = True
            if not progress:
                raise RuntimeError("URDF tree disconnected")
        foot_z = [matrices[name][2, 3] for name in feet if name in matrices]
        if not foot_z:
            raise RuntimeError(f"No feet found for {self.kind}")
        body = transform((xy[0], xy[1], ground-min(foot_z)+sole), (0, 0, yaw))
        for name, op in self.ops.items():
            op.Set(Gf.Matrix4d((body @ matrices[name]).T.tolist()))
