"""Native Isaac map review: no match rollout or model calls."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import time
import traceback
import numpy as np
from isaaclab.app import AppLauncher

from arena_paths import PROJECT as project, workspace_root, rules_reference_path
workspace=workspace_root()
parser=argparse.ArgumentParser()
parser.add_argument('--dockyard',action='store_true')
parser.add_argument('--dockyard-config',default='dockyard_v1.json')
parser.add_argument('--preview',action='store_true')
parser.add_argument('--no-tour',action='store_true')
parser.add_argument('--interactive',action='store_true')
parser.add_argument('--run-dir',type=Path,required=True)
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args();args.enable_cameras=True
if args.interactive and args.headless:parser.error('--interactive conflicts with --headless')
app=AppLauncher(args).app
import carb
import omni.usd
import omni.replicator.core as rep
from pxr import Usd, UsdGeom, UsdPhysics, Gf
from PIL import Image
from arena_art import lighting
from dockyard_art import Dockyard
from dockyard_geometry import load_design,solids,validate_design
from dockyard_report import build_report
from demo_actors import Actor


def main():
    started=time.monotonic();d=load_design(args.dockyard_config);run=args.run_dir
    validation=validate_design(d)
    (run/'map_design.json').write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n')
    (run/'geometry_validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    rules_path=rules_reference_path()
    rules_bytes=rules_path.read_bytes();(run/'rules_reference.json').write_bytes(rules_bytes)
    scene_dir=workspace/'simulators/scenes/quadruped-arena'/run.name;scene_dir.mkdir(parents=True,exist_ok=True)
    omni.usd.get_context().new_stage();stage=omni.usd.get_context().get_stage()
    UsdGeom.SetStageMetersPerUnit(stage,1.);UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage,'/World').GetPrim())
    UsdPhysics.Scene.Define(stage,'/World/PhysicsScene').CreateGravityMagnitudeAttr(9.81)
    builder=Dockyard(stage,workspace/'simulators/assets/quadruped-arena/pbr',scene_dir/'labels')
    metadata=builder.build(d)
    samples=48 if args.preview else d['render']['samples'];lighting(stage,samples)
    a=Actor(stage,'/World/Actors/A','go2w',workspace);a.update(d['spawns']['A'],.55,0,False,0)
    b=Actor(stage,'/World/Actors/B','g1',workspace);b.update(d['spawns']['B'],-2.2,0,False,0)
    collider_checks={}
    for s in solids(d):
        prim=stage.GetPrimAtPath('/World/Geometry/'+s['name']+('/Collider' if s['shape']=='box' else ''))
        assert prim and prim.HasAPI(UsdPhysics.CollisionAPI),s['name']
        if s['shape']=='box':
            size=np.array(UsdGeom.Xformable(prim).GetOrderedXformOps()[0].Get())
            np.testing.assert_allclose(size,s['size']);collider_checks[s['name']]=size.tolist()
    collapse_wall=next(w for w in d['walls'] if w['name']==d['collapse']['wall'])
    root='/World/Geometry/'+collapse_wall['name'];wall=UsdGeom.Xformable(stage.GetPrimAtPath(root))
    wall.ClearXformOpOrder();wall_op=wall.AddTransformOp()
    center=np.array([*collapse_wall['center'],collapse_wall['size'][2]/2]);pivot=np.array(d['collapse']['pivot'])
    def collapse(fallen):
        angle=math.radians(d['collapse']['angle_degrees']) if fallen else 0
        rot=np.array([[1,0,0],[0,math.cos(angle),-math.sin(angle)],[0,math.sin(angle),math.cos(angle)]])
        matrix=np.eye(4);matrix[:3,:3]=rot;matrix[:3,3]=pivot+rot@(center-pivot)
        wall_op.Set(Gf.Matrix4d(matrix.T.tolist()))
        actual=UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(root+'/Collider'))
        corners=np.array([list(actual.Transform(Gf.Vec3d(*v))) for v in itertools.product((-.5,.5),repeat=3)])
        expected=d['collapse']['debris'] if fallen else dict(center=center,size=collapse_wall['size'])
        np.testing.assert_allclose(corners.mean(axis=0),expected['center'],atol=1e-6)
        np.testing.assert_allclose(np.ptp(corners,axis=0),expected['size'],atol=1e-6)
    def set_view(cam,op,view):
        cam.GetFocalLengthAttr().Set(view['focal'])
        op.Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*view['eye']),Gf.Vec3d(*view['target']),Gf.Vec3d(0,0,1)).GetInverse())
    W,H=[1120,700] if args.preview else d['render']['resolution']
    for name,view in d['cameras'].items():
        saved=UsdGeom.Camera.Define(stage,'/World/Cameras/'+name)
        saved.CreateHorizontalApertureAttr(20.955);saved.CreateVerticalApertureAttr(20.955*H/W)
        saved.CreateClippingRangeAttr(Gf.Vec2f(.04,1000));set_view(saved,saved.AddTransformOp(),view)
    collapse(False);stage.Export(str(scene_dir/'dockyard_before.usda'))
    collapse(True);stage.Export(str(scene_dir/'dockyard_after.usda'));collapse(False)
    camera=UsdGeom.Camera.Define(stage,'/World/CaptureCamera')
    camera.CreateHorizontalApertureAttr(20.955);camera.CreateVerticalApertureAttr(20.955*H/W)
    camera.CreateClippingRangeAttr(Gf.Vec2f(.04,1000));cam_op=camera.AddTransformOp()
    product=rep.create.render_product(str(camera.GetPath()),(W,H));rgb=rep.AnnotatorRegistry.get_annotator('rgb')
    rgb.attach([product]);rep.orchestrator.set_capture_on_play(False)
    outputs={}
    views=list(d['cameras'].items())+[('collapse_overview',d['cameras']['overview']),
        ('collapse_detail',dict(eye=[-5,1.8,3.8],target=[-.4,4.3,.6],focal=18))]
    try:
        for name,view in views:
            collapse(name.startswith('collapse_'));set_view(camera,cam_op,view)
            for _ in range(math.ceil(samples/8)+24):app.update()
            pixels=rgb.get_data();assert pixels.size and np.std(pixels[...,:3])>4,'blank render: '+name
            file=run/(name+'.png');Image.fromarray(pixels[...,:3]).save(file);outputs[name]=str(file)
            print(f'DOCKYARD_RENDER {name} {W}x{H} target={samples}spp elapsed={time.monotonic()-started:.1f}s',flush=True)
    finally:rgb.detach([product.path]);product.destroy()
    collapse(False)
    if not args.no_tour:
        import imageio.v2 as imageio
        width,height=d['render']['tour_resolution'];n=24 if args.preview else d['render']['tour_frames'];fps=d['render']['tour_fps']
        tspp=d['render']['tour_samples'];settings=carb.settings.get_settings()
        settings.set('/rtx/pathtracing/totalSpp',tspp);settings.set('/rtx/pathtracing/clampSpp',tspp)
        camera.GetVerticalApertureAttr().Set(20.955*height/width)
        product=rep.create.render_product(str(camera.GetPath()),(width,height));rgb.attach([product])
        frames=[];frame_dir=run/'tour_frames';frame_dir.mkdir(exist_ok=True)
        try:
            for i in range(n):
                angle=math.radians(-49)+2*math.pi*i/n
                view=dict(eye=[43*math.cos(angle),43*math.sin(angle),36 if d.get('facility_solids') else 30],target=[0,0,.4],focal=24 if d.get('facility_solids') else 22)
                set_view(camera,cam_op,view)
                for _ in range(math.ceil(tspp/8)+8):app.update()
                pixels=rgb.get_data()[...,:3].copy();assert pixels.size and np.std(pixels)>4
                Image.fromarray(pixels).save(frame_dir/f'{i:04}.png');frames.append(pixels)
                if i%8==0:print(f'DOCKYARD_TOUR {i+1}/{n} elapsed={time.monotonic()-started:.1f}s',flush=True)
        finally:rgb.detach([product.path]);product.destroy()
        imageio.mimwrite(run/'dockyard_tour.mp4',frames,fps=fps,codec='libx264',quality=8,macro_block_size=2)
        images=[Image.fromarray(f) for f in frames]
        # GIF timestamps use 10 ms ticks; alternating 120/130 preserves 8 fps.
        gif_delays=[10*(round((i+1)*100/fps)-round(i*100/fps)) for i in range(n)]
        images[0].save(run/'dockyard_tour.gif',save_all=True,append_images=images[1:],duration=gif_delays,loop=0,optimize=False)
        outputs.update(tour_gif=str(run/'dockyard_tour.gif'),tour_mp4=str(run/'dockyard_tour.mp4'))
    metadata.update(render_engine='Isaac Sim RTX PathTracing',target_spp=samples,resolution=[W,H],
        collider_sizes=collider_checks,collision_prim_count=sum(p.HasAPI(UsdPhysics.CollisionAPI) for p in stage.Traverse()),
        collapse_collider_bounds_verified=True,rules_sha256=hashlib.sha256(rules_bytes).hexdigest(),
        rules_source=str(rules_path),rules_byte_identical=True,
        tour=dict(kind='static_map_camera_orbit_not_match',resolution=d['render']['tour_resolution'],
            target_spp=d['render']['tour_samples'],frames=0 if args.no_tour else n,fps=d['render']['tour_fps']))
    (run/'art_manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    result=dict(status='passed',kind='dockyard_map_review',outputs=outputs,scene_directory=str(scene_dir),
        wall_seconds=time.monotonic()-started,validation=metadata,geometry_validation=validation)
    (run/'result.json').write_text(json.dumps(result,indent=2)+'\n');build_report(run,d,result)
    print('DOCKYARD_RESULT '+json.dumps({k:result[k] for k in ['status','kind','scene_directory','wall_seconds']}),flush=True)
    if args.interactive:
        from omni.kit.viewport.utility import get_active_viewport
        viewport=get_active_viewport()
        if viewport:viewport.camera_path='/World/Cameras/overview'
        while app.is_running():app.update()


try:main()
except BaseException:
    error=traceback.format_exc();(args.run_dir/'error.txt').write_text(error)
    (args.run_dir/'result.json').write_text(json.dumps(dict(status='failed',error=error))+'\n')
    print(error,flush=True);raise
finally:app.close(wait_for_replicator=False)
