"""Create a portable result archive and SHA-256 manifest."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):value.update(chunk)
    return value.hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('paths',nargs='+',type=Path)
    parser.add_argument('--output',required=True,type=Path);args=parser.parse_args()
    roots=[p.resolve() for p in args.paths]
    for root in roots:
        if not root.exists():raise FileNotFoundError(root)
    output=args.output.resolve();output.parent.mkdir(parents=True,exist_ok=True)
    manifest=[]
    for root in roots:
        files=[root] if root.is_file() else sorted(p for p in root.rglob('*') if p.is_file())
        for path in files:manifest.append(dict(source=str(path),size=path.stat().st_size,sha256=digest(path)))
    manifest_path=output.with_suffix(output.suffix+'.sha256.json')
    manifest_path.write_text(json.dumps(dict(files=manifest),indent=2)+'\n')
    with tarfile.open(output,'w:gz') as archive:
        for root in roots:archive.add(root,arcname=root.name,recursive=True)
        archive.add(manifest_path,arcname=manifest_path.name)
    print(json.dumps(dict(archive=str(output),archive_sha256=digest(output),manifest=str(manifest_path),files=len(manifest)),indent=2))


if __name__=='__main__':main()
