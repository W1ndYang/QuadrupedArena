"""Native Isaac stills + reusable USD for the industrial courtyard art branch."""
import argparse
import json
import math
from pathlib import Path
import time
import traceback
import numpy as np
from isaaclab.app import AppLauncher

from arena_paths import PROJECT as project, workspace_root
workspace=workspace_root()
parser=argparse.ArgumentParser()
parser.add_argument('--art',action='store_true')
parser.add_argument('--preview',action='store_true')
parser.add_argument('--interactive',action='store_true')
parser.add_argument('--run-dir',type=Path,required=True)
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args();args.enable_cameras=True
if args.interactive and args.headless:parser.error('--interactive cannot be combined with --headless')
app=AppLauncher(args).app
import omni.usd
import omni.replicator.core as rep
from pxr import Usd, UsdGeom, UsdPhysics, Gf
from PIL import Image
from arena_art import Courtyard, lighting
from demo_actors import Actor


def main():
    started=time.monotonic()
    art=json.loads((project/'configs/art_v1.json').read_text())
    cfg=json.loads((project/'configs'/art['rules_config']).read_text())
    scene_dir=workspace/'simulators/scenes/quadruped-arena'/args.run_dir.name
    scene_dir.mkdir(parents=True,exist_ok=True)
    omni.usd.get_context().new_stage();stage=omni.usd.get_context().get_stage()
    UsdGeom.SetStageMetersPerUnit(stage,1.);UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage,'/World').GetPrim())
    builder=Courtyard(stage,workspace/'simulators/assets/quadruped-arena/pbr',scene_dir/'labels')
    metadata=builder.build(cfg)
    samples=64 if args.preview else art['samples']; lighting(stage,samples)
    # Native robot meshes are only scale references here, not a match rollout.
    a=Actor(stage,'/World/Actors/A','go2w',workspace); a.update([-4.05,-2.65],.55,0,0,0)
    b=Actor(stage,'/World/Actors/B','g1',workspace); b.update([4.2,.3],-2.2,0,0,0)
    checks={}
    for c in cfg['arena']['covers']:
        collider=stage.GetPrimAtPath('/World/Covers/'+c['name']+'/Collider')
        assert collider.HasAPI(UsdPhysics.CollisionAPI)
        size=np.asarray(UsdGeom.Xformable(collider).GetOrderedXformOps()[0].Get())
        assert np.allclose(size,c['size'])
        checks[c['name']]=size.tolist()
    assert not any(p.HasAPI(UsdPhysics.CollisionAPI) for p in Usd.PrimRange(stage.GetPrimAtPath('/World/Art')))
    # The decorated high wall remains one transform hierarchy for fixed collapse.
    wall=UsdGeom.Xformable(stage.GetPrimAtPath('/World/Covers/'+cfg['collapse']['wall']))
    wall.ClearXformOpOrder();wall_op=wall.AddTransformOp()
    original=next(c for c in cfg['arena']['covers'] if c['name']==cfg['collapse']['wall'])
    center=np.array([*original['center'],original['size'][2]/2]);pivot=np.array(cfg['collapse']['pivot'])
    def collapse(fallen):
        angle=math.radians(cfg['collapse']['angle_degrees']) if fallen else 0.
        rot=np.array([[1,0,0],[0,math.cos(angle),-math.sin(angle)],[0,math.sin(angle),math.cos(angle)]])
        m=np.eye(4);m[:3,:3]=rot;m[:3,3]=pivot+rot@(center-pivot)
        wall_op.Set(Gf.Matrix4d(m.T.tolist()))
        # Independently transform all eight original collider corners to world.
        import itertools
        collider=stage.GetPrimAtPath('/World/Covers/'+cfg['collapse']['wall']+'/Collider')
        matrix=UsdGeom.XformCache().GetLocalToWorldTransform(collider)
        corners=np.array([list(matrix.Transform(Gf.Vec3d(*p))) for p in itertools.product((-.5,.5),repeat=3)])
        expected=cfg['collapse']['debris'] if fallen else original
        np.testing.assert_allclose(np.ptp(corners,axis=0),expected['size'],atol=1e-6)
        np.testing.assert_allclose(corners.mean(axis=0),[*expected['center'],expected['size'][2]/2],atol=1e-6)
    collapse(False)
    W,H=([1120,700] if args.preview else art['resolution'])
    for name,view in art['cameras'].items():
        saved=UsdGeom.Camera.Define(stage,'/World/Cameras/'+name)
        saved.CreateFocalLengthAttr(view['focal']);saved.CreateHorizontalApertureAttr(20.955)
        saved.CreateVerticalApertureAttr(20.955*H/W)
        saved.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*view['eye']),Gf.Vec3d(*view['target']),Gf.Vec3d(0,0,1)).GetInverse())
    camera=UsdGeom.Camera.Define(stage,'/World/Camera'); camera.CreateHorizontalApertureAttr(20.955)
    camera.CreateVerticalApertureAttr(20.955*H/W);camera.CreateClippingRangeAttr(Gf.Vec2f(.05,1000))
    cam_op=camera.AddTransformOp()
    product=rep.create.render_product(str(camera.GetPath()),(W,H))
    rgb=rep.AnnotatorRegistry.get_annotator('rgb');rgb.attach([product]);rep.orchestrator.set_capture_on_play(False)
    outputs={}
    views=list(art['cameras'].items())+[('collapse_overview',art['cameras']['overview'])]
    try:
        for i,(name,view) in enumerate(views):
            collapse(name=='collapse_overview')
            camera.GetFocalLengthAttr().Set(view['focal'])
            cam_op.Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*view['eye']),Gf.Vec3d(*view['target']),Gf.Vec3d(0,0,1)).GetInverse())
            for _ in range(math.ceil(samples/8)+24):app.update()
            pixels=rgb.get_data()
            assert pixels.size and np.std(pixels[...,:3])>4, 'Blank render'
            file=args.run_dir/(name+'.png');Image.fromarray(pixels[...,:3]).save(file);outputs[name]=str(file)
            print(f'ART_RENDER {name} {W}x{H} target={samples}spp elapsed={time.monotonic()-started:.1f}s',flush=True)
            if name=='overview':stage.Export(str(scene_dir/'courtyard_before.usda'))
            if name=='collapse_overview':stage.Export(str(scene_dir/'courtyard_after.usda'))
    finally:
        rgb.detach([product.path]);product.destroy()
    metadata.update(render_engine='Isaac Sim RTX PathTracing',target_spp=samples,resolution=[W,H],collider_sizes=checks,
        training=False,match_simulated=False,dynamics_validated=False,asset_paths='absolute local paths; copy/repath assets for another machine')
    (args.run_dir/'art_manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    result=dict(status='passed',kind='industrial_courtyard_art',outputs=outputs,scene_directory=str(scene_dir),
        wall_seconds=time.monotonic()-started,validation=metadata)
    (args.run_dir/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print('ART_RESULT '+json.dumps(result),flush=True)
    if args.interactive:
        collapse(False)
        from omni.kit.viewport.utility import get_active_viewport
        viewport=get_active_viewport()
        if viewport:viewport.camera_path='/World/Cameras/courtyard'
        print('Interactive courtyard ready. Close Isaac window to exit.',flush=True)
        while app.is_running():app.update()


try:main()
except BaseException:
    error=traceback.format_exc();(args.run_dir/'error.txt').write_text(error)
    (args.run_dir/'result.json').write_text(json.dumps(dict(status='failed',error=error))+'\n')
    print(error,flush=True);raise
finally:app.close(wait_for_replicator=False)
