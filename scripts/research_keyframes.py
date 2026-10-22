"""Render saved exact states for a report, including explicitly incomplete runs.
No policy, referee advancement, model calls, new outcomes or dynamics.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from isaaclab.app import AppLauncher

p=argparse.ArgumentParser()
p.add_argument('--output',type=Path,required=True)
p.add_argument('--sources',nargs='+',type=Path,required=True)
AppLauncher.add_app_launcher_args(p)
a=p.parse_args();a.enable_cameras=True
app=AppLauncher(a).app
import omni.usd
import omni.replicator.core as rep
from pxr import UsdGeom,Gf
from PIL import Image
from arena_art import Courtyard,lighting
from demo_actors import Actor

def main():
    from arena_paths import workspace_root
    workspace=workspace_root()
    all_records=[]
    for source in a.sources:
        states=json.loads((source/'timeline.json').read_text())
        cfg=json.loads((source/'trial_config.json').read_text())
        out=a.output/source.name;out.mkdir(parents=True,exist_ok=False)
        omni.usd.get_context().new_stage();stage=omni.usd.get_context().get_stage()
        UsdGeom.SetStageMetersPerUnit(stage,1.);UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
        stage.SetDefaultPrim(UsdGeom.Xform.Define(stage,'/World').GetPrim())
        Courtyard(stage,workspace/'simulators/assets/quadruped-arena/pbr',out/'labels').build(cfg)
        lighting(stage,16)
        actors=[Actor(stage,'/World/Actors/A','go2w',workspace),Actor(stage,'/World/Actors/B','g1',workspace)]
        wall=UsdGeom.Xformable(stage.GetPrimAtPath('/World/Covers/'+cfg['collapse']['wall']))
        wall.ClearXformOpOrder();wallop=wall.AddTransformOp()
        orig=next(c for c in cfg['arena']['covers'] if c['name']==cfg['collapse']['wall'])
        center=np.array([*orig['center'],orig['size'][2]/2]);pivot=np.asarray(cfg['collapse']['pivot'])
        cam=UsdGeom.Camera.Define(stage,'/World/ResearchCamera');cam.CreateFocalLengthAttr(27)
        cam.CreateHorizontalApertureAttr(36);cam.CreateVerticalApertureAttr(36)
        cam.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(-12,-14,20),Gf.Vec3d(0,.8,.35),Gf.Vec3d(0,0,1)).GetInverse())
        product=rep.create.render_product(str(cam.GetPath()),(960,960))
        rgb=rep.AnnotatorRegistry.get_annotator('rgb');rgb.attach([product]);rep.orchestrator.set_capture_on_play(False)
        times=sorted(set([0.,8.,10.,12.,16.,min(20.5,states[-1]['t']),states[-1]['t']]))
        records=[]
        for j,t in enumerate(times):
            s=min(states,key=lambda s:abs(s['t']-t));assert abs(s['t']-t)<1e-7
            for i,actor in enumerate(actors):
                actor.update(s['xy'][i],s['yaw'][i],t*s['speed'][i]*2*math.pi/.65,s['speed'][i],0,s['wheels'] if i==0 else None)
                if i==1 and s['hp'][i]==0:UsdGeom.Imageable(stage.GetPrimAtPath('/World/Actors/B')).MakeInvisible()
            for i in range(2):
                for stem in ('Target','TargetLabel'):
                    item=UsdGeom.Imageable(stage.GetPrimAtPath('/World/Art/'+stem+str(i)))
                    item.MakeVisible() if i==s['active_target'] else item.MakeInvisible()
            theta=math.radians(cfg['collapse']['angle_degrees'])*float(s['collapsed'])
            rot=np.array([[1,0,0],[0,math.cos(theta),-math.sin(theta)],[0,math.sin(theta),math.cos(theta)]])
            matrix=np.eye(4);matrix[:3,:3]=rot;matrix[:3,3]=pivot+rot@(center-pivot)
            wallop.Set(Gf.Matrix4d(matrix.T.tolist()))
            for _ in range(16 if j==0 else 8):app.update()
            pixels=np.asarray(rgb.get_data())[...,:3];assert pixels.std()>1
            dest=out/f'{t:05.2f}.png';Image.fromarray(pixels).save(dest)
            records.append(dict(time=t,file=str(dest),sha256=hashlib.sha256(dest.read_bytes()).hexdigest()))
            print(f'RESEARCH_FRAME {source.name} t={t}',flush=True)
        rgb.detach([product.path]);product.destroy()
        all_records.append(dict(source=str(source),source_timeline_sha256=hashlib.sha256((source/'timeline.json').read_bytes()).hexdigest(),
            source_finished=states[-1]['finished'],observed_end=states[-1]['t'],policy_rerun=False,
            provenance='Native Isaac posthoc rendering of exact archived transforms; not model input; no continuation of incomplete match.',frames=records))
    (a.output/'manifest.json').write_text(json.dumps(all_records,ensure_ascii=False,indent=2))

try:main()
finally:app.close()
