"""Render the Oracle region audit directly from config and Planner free nodes."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np

from adaptive_motion import load_config
from oracle_state_v1 import RegionCatalog
from trial_rules import covers_at


def draw(path: Path):
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs/trial_rotating.json")
    catalog = RegionCatalog.from_config(cfg)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)
    palette = {"target_zone": "#27ae60", "cover_zone": "#e3a83a", "open_zone": "#8db9e5"}
    roles = {r.region_id: r.role for r in catalog.regions}
    for axis, (phase, t, planner) in zip(axes, (("pre_collapse", 0, catalog.planners[0]),
                                               ("post_collapse", cfg["collapse"]["time"], catalog.planners[1]))):
        by_role = {role: [] for role in palette}
        for node in sorted(planner.free_nodes):
            xy = (node[0] * planner.grid, node[1] * planner.grid)
            by_role[roles[catalog.locate(xy, phase)]].append(xy)
        for role, points in by_role.items():
            if points:
                arr = np.asarray(points)
                axis.scatter(arr[:, 0], arr[:, 1], s=24, marker="s", color=palette[role],
                             linewidths=0, label=f"{role} free nodes")
        for cover in covers_at(cfg, t):
            x, y = cover["center"]
            width, height = cover["size"][:2]
            axis.add_patch(Rectangle((x-width/2, y-height/2), width, height,
                                     facecolor="#555555", edgecolor="#222222", alpha=.75))
            axis.text(x, y, cover["name"], fontsize=7, ha="center", va="center", color="white")
        for point in cfg["arena"]["task_points"]:
            x, y = point["center"]
            axis.add_patch(Circle((x, y), point["radius"], fill=False, color="#087a3e", lw=2))
            axis.text(x, y+.67, "target_"+point["name"], fontsize=8, ha="center")
        for key, color in (("start_a", "#c42635"), ("start_b", "#7327a2")):
            axis.scatter(*cfg[key], s=110, marker="*", color=color, edgecolor="black", zorder=6)
            axis.text(cfg[key][0]+.12, cfg[key][1]-.1, key, fontsize=8)
        for key, style in (("start_a", "-"), ("start_b", "--")):
            for point in cfg["arena"]["task_points"]:
                route = planner.route(np.asarray(cfg[key], float), np.asarray(point["center"], float))
                if route is None:
                    continue
                arr = np.asarray([cfg[key], *route])
                axis.plot(arr[:, 0], arr[:, 1], style, color="#c42635" if key == "start_a" else "#7327a2",
                          alpha=.55, linewidth=1)
        axis.set_title(f"{phase} (t={t:g}s)")
        axis.set_xlim(-6, 6)
        axis.set_ylim(-4, 4.4)
        axis.set_aspect("equal")
        axis.set_xlabel("x (m)")
        axis.grid(alpha=.15)
    axes[0].set_ylabel("y (m)")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Oracle v1.1 region audit: configured covers, target circles, Planner free nodes and routes")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    draw(parser.parse_args().output)
