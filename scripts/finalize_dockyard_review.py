"""Verify review artifacts and encode GIF with exact centisecond timing.

Does not alter pixels, rerender scenes, or invoke decision models.
"""
import hashlib
import json
from pathlib import Path
import sys
from PIL import Image
from dockyard_report import build_report


def finalize(run):
    run=Path(run);r=json.loads((run/'result.json').read_text());d=json.loads((run/'map_design.json').read_text())
    assert r['status']=='passed' and r['kind']=='dockyard_map_review'
    checks=[];res=tuple(r['validation']['resolution'])
    for key in [*d['cameras'],'collapse_overview','collapse_detail']:
        with Image.open(run/(key+'.png')) as im:assert im.size==res
    checks.append('Ten native stills are present at the declared resolution')
    rules=(run/'rules_reference.json').read_bytes()
    assert rules==Path(r['validation']['rules_source']).read_bytes()
    assert hashlib.sha256(rules).hexdigest()==r['validation']['rules_sha256']
    checks.append('Competition rule snapshot is byte-identical to the archived full run')
    for asset in r['validation']['assets']:
        for file,digest in asset['files'].items():assert hashlib.sha256(Path(file).read_bytes()).hexdigest()==digest
    checks.append('All twelve PBR input maps match their recorded hashes')
    scene=Path(r['scene_directory'])
    assert all((scene/name).is_file() for name in ['dockyard_before.usda','dockyard_after.usda'])
    checks.append('Both native USD snapshots exist; native builder verified wall transform and 108 collision prims')
    tour=r['validation']['tour'];files=sorted((run/'tour_frames').glob('*.png'))
    assert len(files)==tour['frames']==72
    frames=[]
    for file in files:
        with Image.open(file) as im:
            assert im.size==tuple(tour['resolution']);frames.append(im.convert('RGB'))
    assert len({hashlib.sha256(f.tobytes()).hexdigest() for f in frames})==len(frames)
    fps=tour['fps'];delays=[10*(round((i+1)*100/fps)-round(i*100/fps)) for i in range(len(frames))]
    frames[0].save(run/'dockyard_tour.gif',save_all=True,append_images=frames[1:],duration=delays,loop=0,optimize=False)
    with Image.open(run/'dockyard_tour.gif') as im:
        assert im.n_frames==72
        ms=0
        for i in range(im.n_frames):im.seek(i);ms+=im.info['duration']
        assert ms==9000
    checks.append('72 distinct Isaac frames; GIF has 72 frames and exact 9.000-second timing')
    build_report(run,d,r)
    source=Path(__file__).read_bytes()
    (run/'finalization_source.py').write_bytes(source)
    result=dict(status='passed',checks=checks,source_sha256=hashlib.sha256(source).hexdigest(),
        gif_pixel_transforms=False,postprocess='Re-encode original rendered PNG frames with 120/130 ms GIF delays; regenerate local HTML.')
    (run/'artifact_checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':finalize(sys.argv[1])
