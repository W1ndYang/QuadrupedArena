"""Encode the exact Isaac onboard frames from a completed RL Skill match."""
import argparse
from contextlib import contextmanager
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile

import numpy as np
from PIL import Image


@contextmanager
def source_reader(run_dir=None, archive=None):
    if run_dir is not None:
        root=Path(run_dir).resolve()
        def read(name):return (root/name).read_bytes()
        yield read,str(root)
        return
    with tarfile.open(archive,'r:gz') as package:
        candidates=[m.name[:-len('/frame_index.json')] for m in package.getmembers()
                    if m.isfile() and m.name.count('/')==1 and m.name.endswith('/frame_index.json')]
        if len(candidates)!=1:raise ValueError('Expected one match frame_index.json in the archive')
        root=candidates[0]
        def read(name):
            member=package.getmember(root+'/'+name)
            if not member.isfile():raise ValueError('Frame archive member is not a regular file')
            return package.extractfile(member).read()
        yield read,str(Path(archive).resolve())+'#'+root


def valid_frame_path(name):
    if not isinstance(name,str):return False
    path=PurePosixPath(name)
    return not path.is_absolute() and len(path.parts)==2 and path.parts[0]=='frames' and path.suffix.lower() in ('.jpg','.jpeg','.png')


def ffmpeg_executable():
    executable=shutil.which('ffmpeg')
    if executable:return executable
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError,OSError) as exc:
        raise RuntimeError('ffmpeg or imageio-ffmpeg is required to encode MP4') from exc


def encode(read,source,output,fps):
    frames=json.loads(read('frame_index.json'))
    result=json.loads(read('result.json'))
    if result.get('status')!='passed' or result.get('mode')!='rl_skill_policy_isaac_trial':
        raise ValueError('Source is not a completed online RL Skill match')
    if not frames or len(frames)!=result['sensor_frames']:
        raise ValueError('Frame count does not match the online result')
    names=[frame['file'] for frame in frames]
    if len(set(names))!=len(names) or not all(map(valid_frame_path,names)):
        raise ValueError('Frame filenames are missing, unsafe or repeated')
    for index,frame in enumerate(frames):
        if abs(frame['simulation_seconds']-index/fps)>1e-6:
            raise ValueError(f'Frame {index} breaks the {fps} Hz simulation cadence')
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    error_log=output.with_suffix('.ffmpeg.log')
    command=[ffmpeg_executable(),'-nostdin','-y','-v','error','-f','rawvideo',
             '-pix_fmt','rgb24','-s:v','768x432','-r',str(fps),'-i','pipe:0',
             '-an','-c:v','libx264','-preset','medium','-crf','18',
             '-pix_fmt','yuv420p',str(output)]
    try:
        with error_log.open('wb') as error_stream:
            process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,
                                     stderr=error_stream)
            try:
                for index,name in enumerate(names):
                    with Image.open(BytesIO(read(name))) as picture:
                        if picture.size!=(768,432):raise ValueError(f'Unexpected image size at frame {index}')
                        process.stdin.write(np.asarray(picture.convert('RGB')).tobytes())
                process.stdin.close()
                code=process.wait()
                if code:raise RuntimeError(f'ffmpeg exited {code}; see {error_log}')
            except BaseException:
                process.stdin.close() if process.stdin and not process.stdin.closed else None
                process.kill();process.wait()
                raise
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    error_log.unlink(missing_ok=True)
    metadata=dict(source=source,video=str(output),fps=fps,frames=len(frames),
                  video_seconds=len(frames)/fps,first_simulation_seconds=frames[0]['simulation_seconds'],
                  last_simulation_seconds=frames[-1]['simulation_seconds'],
                  official_score=result['score'],finish=result['finish'],
                  view='Isaac onboard RGB captured during the live policy match; no policy rerun')
    output.with_suffix('.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(metadata,ensure_ascii=False,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser()
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--run-dir',type=Path)
    source.add_argument('--archive',type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--fps',type=int,default=10)
    args=parser.parse_args()
    if args.fps<=0:parser.error('--fps must be positive')
    with source_reader(args.run_dir,args.archive) as (read,label):
        encode(read,label,args.output,args.fps)


if __name__=='__main__':main()
