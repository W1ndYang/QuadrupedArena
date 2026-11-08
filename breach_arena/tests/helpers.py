"""Small controllers and an empty test arena for rule-level tests."""
from breach_arena.config import GameMap


def arena(covers=(), quad=(-6.0, 0.0), human=(0.0, 0.0), goal=(10.0, 6.0), human_yaw=180, quad_yaw=0):
    return GameMap(dict(schema='breach-map-v1', id='TEST', title='test', arena=dict(size=[24, 16], covers=list(covers)),
                        goal=dict(center=list(goal), radius=1.0),
                        spawns=dict(quadruped=dict(xy=list(quad), yaw_degrees=quad_yaw),
                                    human=dict(xy=list(human), yaw_degrees=human_yaw)),
                        points=dict(A=dict(xy=[-6.0, 3.0], kind='lane')), plays={}, human_bots={}))


def wall(name, x0, x1, y0, y1, h=2.4):
    return dict(name=name, center=[(x0 + x1) / 2, (y0 + y1) / 2], size=[x1 - x0, y1 - y0, h])


class Script:
    """Controller from a function obs -> command."""

    def __init__(self, fn):
        self.fn = fn
        self.seen = []

    def decide(self, obs):
        self.seen.append(obs)
        return self.fn(obs)


def idle(obs):
    return dict(goal=None, look=None, fire=False)


def aim_at_visible(obs):
    o = obs['opponent']
    return dict(goal=None, look=o['xy'] if o else None, fire=o is not None)
