"""Extract exact decision-boundary screenshots from a completed native replay.

No scene changes, image synthesis, cropping, policy execution or simulation.
The source video includes intro/collapse/outro frames, so wall-clock seek is
not equivalent to simulation time.
"""
import hashlib
import json
from pathlib import Path


def video_frame_index(sim_time,cfg):
    fps=cfg['fps'];tick=round(sim_time*fps)
    if abs(tick/fps-sim_time)>1e-7:
        raise ValueError('Requested simulation time is not an exact video frame')
    intro=round(cfg['intro_seconds']*fps)
    collapse=round(cfg['collapse']['display_seconds']*fps) if sim_time>=cfg['collapse']['time'] else 0
    return intro+tick+collapse


def extract_screenshots(directory,render_dir,summary,times):
    import imageio.v2 as imageio
    directory,render_dir=Path(directory),Path(render_dir)
    cfg=json.loads((directory/'trial_config.json').read_text())
    result=json.loads((render_dir/'result.json').read_text())
    if result['status']!='passed' or result['kind']!='r0_rules_trial':
        raise ValueError('Screenshots require a successful full native replay, not a sparse preview')
    if result['replay']['source_sha256']!=summary['replay']['source_sha256']:
        raise ValueError('Rendered trajectory differs from the decision log')
    expected=round(cfg['intro_seconds']*cfg['fps'])+round(result['match_seconds']*cfg['fps'])+1+round(cfg['outro_seconds']*cfg['fps'])
    if result['match_seconds']>=cfg['collapse']['time']:expected+=round(cfg['collapse']['display_seconds']*cfg['fps'])
    frame_map=None
    if (render_dir/'frame_index.json').exists():
        frame_map=json.loads((render_dir/'frame_index.json').read_text())
        if frame_map['frames']!=result['frames'] or frame_map['fps']!=cfg['fps']:raise ValueError('Invalid saved frame map')
    if (frame_map is None and result['frames']!=expected) or result['fps']!=cfg['fps']:
        raise ValueError('Video framing does not match the renderer timeline contract')
    video=render_dir/'demo.mp4';video_hash=hashlib.sha256(video.read_bytes()).hexdigest()
    dest=directory/'screenshots';dest.mkdir(exist_ok=True)
    def frame_for(t):
        if frame_map is None:return video_frame_index(t,cfg)
        matches=[r['video_frame_index'] for r in frame_map['regular_frames'] if abs(r['simulation_time']-t)<1e-7]
        if len(matches)!=1:raise ValueError('Missing or duplicate exact decision frame')
        return matches[0]
    records=[dict(observation_index=i,simulation_time=t,video_frame_index=frame_for(t),
        video_time_seconds=frame_for(t)/cfg['fps'],file=f'screenshots/{i:03}.png') for i,t in enumerate(times)]
    manifest_path=dest/'manifest.json'
    if manifest_path.exists():
        old=json.loads(manifest_path.read_text())
        if old.get('source_video_sha256')==video_hash and len(old.get('screenshots',[]))==len(records):
            valid=all(all(record[k]==saved.get(k) for k in record) and (directory/record['file']).is_file()
                and hashlib.sha256((directory/record['file']).read_bytes()).hexdigest()==saved.get('sha256')
                for record,saved in zip(records,old['screenshots']))
            if valid:return old
    indices={r['video_frame_index']:r for r in records}
    reader=imageio.get_reader(str(video));seen=set()
    try:
        for index,pixels in enumerate(reader):
            if index in indices:
                record=indices[index];path=directory/record['file']
                imageio.imwrite(path,pixels)
                record['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
                record['resolution']=[int(pixels.shape[1]),int(pixels.shape[0])];seen.add(index)
            if len(seen)==len(indices):break
    finally:reader.close()
    if seen!=set(indices):raise ValueError('Source video ended before all requested screenshots')
    manifest=dict(source_video=str(video),source_video_sha256=video_hash,
        provenance='Exact decoded frames from Isaac native replay; post-hoc presentation, not original planner image input',
        timing='Includes intro frame offset and completed collapse insert at/after collapse time; excludes cinematic intermediate wall poses',
        screenshots=records)
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    return manifest
