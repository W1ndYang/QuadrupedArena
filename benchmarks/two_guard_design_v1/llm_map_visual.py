"""Human-readable map review from the exact checked semantic manifest."""
from pathlib import Path


LANDMARK_NAMES = {
    'first_cover': 'Cover 1 / start',
    'first_south_gate': 'Gate 1 south',
    'first_north_gate': 'Gate 1 north',
    'second_cover': 'Cover 2',
    'second_south_gate': 'Gate 2 south',
    'second_north_gate': 'Gate 2 north',
    'goal': 'Goal',
}
REGION_NAMES = {'first': 'Guard 1 region', 'second': 'Guard 2 region'}
REGION_COLORS = {'first': '#b45309', 'second': '#7c3aed'}


def _draw_obstacles(ax, manifest, label=False):
    from matplotlib.patches import Rectangle

    for obstacle in manifest['obstacles']:
        x, y = obstacle['center'][:2]
        width, height, wall_height = obstacle['size']
        low = wall_height < 1.
        ax.add_patch(Rectangle((x-width/2, y-height/2), width, height,
                               facecolor='#9ca3af' if low else '#334155',
                               edgecolor='#0f172a', linewidth=.55,
                               alpha=.9 if low else 1., zorder=2))
        if label and (obstacle['name'].endswith('middle') or
                      obstacle['name'].startswith(('Central', 'Transfer'))):
            ax.annotate(obstacle['name'].replace('TransferBaffle', 'Baffle '),
                        (x, y), xytext=(0, 4), textcoords='offset points',
                        ha='center', va='bottom', fontsize=6.5, color='#0f172a',
                        bbox=dict(facecolor='white', alpha=.72, edgecolor='none', pad=.5),
                        zorder=7)


def _draw_regions(ax, manifest, labels=True):
    from matplotlib.patches import Polygon

    for region_id, region in manifest['regions'].items():
        color = REGION_COLORS[region_id]
        ax.add_patch(Polygon(region['polygon'], closed=True,
                             facecolor=color, edgecolor=color, linestyle='--',
                             linewidth=1.5, alpha=.13, zorder=1))
        if labels:
            xs = [xy[0] for xy in region['polygon']]
            ax.annotate(REGION_NAMES[region_id], (sum(xs)/len(xs), 3.55),
                        ha='center', va='bottom', fontsize=9, weight='bold',
                        color=color)


def _draw_overview(ax, manifest):
    _draw_regions(ax, manifest)
    for edge in manifest['edges'].values():
        if 'robot' not in edge['traversable_by']:
            continue
        path = edge['path']
        color = '#0891b2' if 'north' in edge['id'] else '#ea580c'
        ax.plot([xy[0] for xy in path], [xy[1] for xy in path],
                color=color, linewidth=2.1, alpha=.85, zorder=3)
    _draw_obstacles(ax, manifest, label=True)
    for node_id, label in LANDMARK_NAMES.items():
        x, y = manifest['nodes'][node_id]['xy']
        ax.scatter(x, y, s=55, facecolor='white', edgecolor='#111827',
                   linewidth=1.6, zorder=5)
        dy = -14 if 'south' in node_id else 8
        if node_id == 'goal':
            dy = -14
        ax.annotate(label, (x, y), xytext=(0, dy), textcoords='offset points',
                    fontsize=8, ha='center', va='top' if dy < 0 else 'bottom',
                    weight='bold', color='#0f172a',
                    bbox=dict(facecolor='white', alpha=.85, edgecolor='none', pad=1.),
                    zorder=8)
    ax.annotate('NORTH  +y', (-11.5, 7.25), color='#0891b2', fontsize=9,
                weight='bold')
    ax.annotate('SOUTH  -y', (-11.5, -7.35), color='#ea580c', fontsize=9,
                weight='bold')
    ax.set_xlim(-12., 12.)
    ax.set_ylim(-8., 8.)
    ax.set_aspect('equal')
    ax.set_title('Arena overview: obstacles, two guard regions, robot route landmarks',
                 fontsize=12, weight='bold')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.grid(color='#dbe3ec', linewidth=.6, zorder=0)


def _draw_region_detail(ax, manifest, region_id):
    region = manifest['regions'][region_id]
    nodes = manifest['nodes']
    _draw_regions(ax, manifest, labels=False)
    for edge in manifest['edges'].values():
        if (edge['start'] not in region['node_ids'] or
                edge['end'] not in region['node_ids']):
            continue
        start = nodes[edge['start']]['xy']
        end = nodes[edge['end']]['xy']
        ax.plot([start[0], end[0]], [start[1], end[1]],
                color=REGION_COLORS[region_id], alpha=.2, linewidth=1., zorder=2)
    entrances = set(region['entrances'])
    exits = set(region['exits'])
    for node_id in region['node_ids']:
        x, y = nodes[node_id]['xy']
        color = ('#0f766e' if node_id in entrances else
                 '#be123c' if node_id in exits else REGION_COLORS[region_id])
        marker = 's' if node_id in entrances or node_id in exits else 'o'
        ax.scatter(x, y, s=48 if marker == 's' else 23, marker=marker,
                   color=color, edgecolor='white', linewidth=.5, zorder=4)
        short = node_id.split('_x', 1)[1]
        ax.annotate(short, (x, y), xytext=(3, 3), textcoords='offset points',
                    fontsize=6.7, color='#111827', zorder=5)
    xs = [xy[0] for xy in region['polygon']]
    ax.set_xlim(min(xs)-.55, max(xs)+.55)
    ax.set_ylim(-3.85, 3.85)
    ax.set_aspect('equal')
    ax.set_title(f'{REGION_NAMES[region_id]}: 12 spatial landmarks', fontsize=10)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.grid(color='#e5e7eb', linewidth=.5)


def checklist(manifest):
    lines = [
        '# Map semantic review checklist',
        '',
        f"Map ID: `{manifest['map_id']}`. The drawing and this checklist come from the checked JSON manifest.",
        '',
        '- [ ] North is +y; south is -y. The overview matches the intended arena.',
        '- [ ] Dark walls are full-height (2.4 m); gray middle walls are low (0.6 m).',
        '- [ ] Cover 1 is at (-8, 0), Cover 2 at (3, 0), Goal at (10, 0).',
        '- [ ] Gate 1 north/south are (-5, +1.6) / (-5, -1.6).',
        '- [ ] Gate 2 north/south are (6, +1.6) / (6, -1.6).',
        '- [ ] Guard 1 region spans x=-3.9..-2.1, y=-3.35..3.35.',
        '- [ ] Guard 2 region spans x=7.1..8.9, y=-3.35..3.35.',
        '- [ ] Teal/red squares in detail views label west/east spatial boundary nodes only;',
        '      they are not fixed patrol endpoints or a hidden patrol route.',
        '- [ ] Thin lines in detail views indicate allowed map connections only;',
        '      actual guard waypoints and order remain unknown each episode.',
        '',
        'If any box is false or unclear, do not approve; report the mismatch.',
        'Approval records a human reviewer and time in `llm_map_manifest.json`.',
    ]
    return '\n'.join(lines)+'\n'


def render(manifest, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, overview = plt.subplots(figsize=(15, 8), layout='constrained')
    fig.patch.set_facecolor('#f8fafc')
    overview.set_facecolor('white')
    _draw_overview(overview, manifest)
    fig.suptitle('Two-guard arena | dark = full wall, gray = low wall',
                 fontsize=14, weight='bold')
    png = output_dir/'map_review.png'
    svg = output_dir/'map_review.svg'
    topology = output_dir/'map_review_topology.png'
    notes = output_dir/'map_review_checklist.md'
    fig.savefig(png, dpi=170)
    fig.savefig(svg)
    plt.close(fig)
    detail, (first, second) = plt.subplots(1, 2, figsize=(10, 9),
                                           layout='constrained')
    detail.patch.set_facecolor('#f8fafc')
    for ax in (first, second):
        ax.set_facecolor('white')
    _draw_region_detail(first, manifest, 'first')
    _draw_region_detail(second, manifest, 'second')
    detail.suptitle('Guard spatial landmarks and allowed connections',
                    fontsize=13, weight='bold')
    detail.savefig(topology, dpi=160)
    plt.close(detail)
    notes.write_text(checklist(manifest))
    return dict(png=str(png), svg=str(svg), topology=str(topology),
                checklist=str(notes))
