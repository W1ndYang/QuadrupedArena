"""Inspect and explicitly approve a geometry-checked map semantic manifest."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from llm_semantic import build_map

MANIFEST=Path(__file__).with_name('llm_map_manifest.json')
GEOMETRY_FIELDS=('schema','nodes','edges','regions','obstacles','map_id')


def check(path=MANIFEST):
    manifest=json.loads(Path(path).read_text())
    expected=build_map()
    if any(manifest.get(k)!=expected[k] for k in GEOMETRY_FIELDS):
        raise ValueError('Map manifest differs from current geometry; regenerate and re-review')
    return manifest


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',type=Path,default=MANIFEST)
    parser.add_argument('--approve',action='store_true')
    parser.add_argument('--reviewer')
    args=parser.parse_args()
    manifest=check(args.manifest)
    if args.approve:
        if not args.reviewer or len(args.reviewer.strip())<2:
            raise ValueError('Provide a real reviewer name after inspecting the manifest')
        manifest['review']=dict(status='human_approved',
            reviewer=args.reviewer.strip(),
            reviewed_at_utc=datetime.now(timezone.utc).isoformat(),
            note='Reviewer inspected Node, Edge, Region and obstacle semantics')
        args.manifest.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(map_id=manifest['map_id'],
        nodes=len(manifest['nodes']),edges=len(manifest['edges']),
        regions=len(manifest['regions']),
        review=manifest['review']),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
