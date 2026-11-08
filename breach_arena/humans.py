"""Human-side controllers: scripted style bots and the live-player adapter (with aim assist).

Bots see only their own legal observation. Styles:
  hold      stand on the goal, face the last threat or sweep the doors
  standoff  same, from a stand-off point (e.g. S_NE)
  hunt      leave the goal, chase the last legal sighting, search, then return
"""
import math

from .geometry import bearing, wrap

LOOK_AT_LAST_S = 1.2


class _Base:
    def attach(self, game_map, rules):
        self.map, self.rules = game_map, rules
        self.contact_since = None
        self.watch = [game_map.resolve(p) for p in game_map.bots.get('watch', [])]
        self.watch_i, self.watch_until = 0, 0.0

    def engage(self, obs):
        """Shared reflex: stop and aim at a visible quadruped in range after the reaction delay."""
        o = obs['opponent']
        if o is None:
            self.contact_since = None
            return None
        if self.contact_since is None:
            self.contact_since = obs['t']
        ready = obs['t'] - self.contact_since >= self.rules.human.bot_reaction_s - 1e-9
        if ready and math.dist(o['xy'], obs['self']['xy']) <= self.rules.tag.range_m:
            return dict(goal=None, look=o['xy'], fire=True, mode='AIM')
        return None

    def scan_target(self, obs):
        m = obs['memory']
        if m is not None and obs['t'] - m['t'] <= 3.0:
            return m['xy']
        if not self.watch:
            return None
        if obs['t'] >= self.watch_until:
            self.watch_i = (self.watch_i + 1) % len(self.watch)
            self.watch_until = obs['t'] + 1.0
        return list(self.watch[self.watch_i])


class HoldBot(_Base):
    """Guard a fixed point ('GOAL' = on the goal, or a stand-off point name)."""

    def __init__(self, at='GOAL', style='hold'):
        self.at_spec, self.style = at, style

    def attach(self, game_map, rules):
        super().attach(game_map, rules)
        self.at = game_map.resolve(self.at_spec)

    def decide(self, obs):
        cmd = self.engage(obs)
        if cmd is not None:
            return cmd
        far = math.dist(obs['self']['xy'], self.at) > 0.1
        look = obs['opponent']['xy'] if obs['opponent'] else self.scan_target(obs)
        return dict(goal=list(self.at) if far else None, look=look, fire=False,
                    mode=('GO_' if far else '') + self.style.upper())


class HuntBot(_Base):
    """Leave the goal and hunt the last legal sighting; return after `return_after_s` without contact."""

    def __init__(self, return_after_s=10.0, route=None):
        self.return_after_s, self.route_spec = return_after_s, route

    def attach(self, game_map, rules):
        super().attach(game_map, rules)
        spec = self.route_spec or game_map.bots.get('hunt_route', [])
        self.route = [game_map.resolve(p) for p in spec]
        self.last_contact = None
        self.target = None
        self.returning = False

    def decide(self, obs):
        cmd = self.engage(obs)
        if cmd is not None:
            self.last_contact, self.returning = obs['t'], False
            return cmd
        o = obs['opponent']
        if o is not None:                              # visible but out of range: close in
            self.last_contact, self.returning = obs['t'], False
            self.target = [round(o['xy'][0] * 2) / 2, round(o['xy'][1] * 2) / 2]
            return dict(goal=self.target, look=o['xy'], fire=False, mode='CHASE')
        mem = obs['memory']
        if self.last_contact is not None and not self.returning:
            lost = obs['t'] - self.last_contact
            if lost >= self.return_after_s:
                self.returning = True
            elif lost < LOOK_AT_LAST_S and mem is not None:
                return dict(goal=None, look=mem['xy'], fire=False, mode='LOOK_LAST')
            else:
                tgt = self.target or (mem or {}).get('xy')
                if tgt is not None and math.dist(obs['self']['xy'], tgt) > 0.5:
                    return dict(goal=list(tgt), look=None, fire=False, mode='SEARCH')
                return dict(goal=None, look=self.scan_target(obs), fire=False, mode='SEARCH_SCAN')
        if self.returning:
            g = self.map.goal
            done = math.dist(obs['self']['xy'], g) <= 0.3
            return dict(goal=None if done else list(g), look=self.scan_target(obs), fire=False,
                        mode='HOLD_AFTER_HUNT' if done else 'RETURN')
        # No contact yet: walk the default hunt route out of the goal area.
        while self.route and math.dist(obs['self']['xy'], self.route[0]) <= 0.5:
            self.route.pop(0)
        if self.route:
            return dict(goal=list(self.route[-1]), via=[list(p) for p in self.route[:-1]], look=None,
                        fire=False, mode='HUNT')
        return dict(goal=None, look=self.scan_target(obs), fire=False, mode='HUNT_SCAN')


class PlayerInput:
    """Adapter for a live human. `read()` returns the latest input:
    {'move': [vx, vy] in -1..1 (world frame), 'aim': [x, y] world point, 'fire': bool}.

    Aim assist (rules.human.aim_assist_deg): while firing, if the quadruped is legally visible
    and within the assist cone of the player's aim direction, the body tracks it exactly.
    Movement is converted to a short look-ahead goal so the referee's planner keeps clearance.
    """

    def __init__(self, read):
        self.read = read

    def attach(self, game_map, rules):
        self.map, self.rules = game_map, rules

    def decide(self, obs):
        inp = self.read() or {}
        me = obs['self']['xy']
        aim = inp.get('aim')
        look = aim
        assisted = False
        o = obs['opponent']
        if o is not None and aim is not None:
            err = abs(wrap(bearing(me, o['xy']) - bearing(me, aim)))
            if err <= math.radians(self.rules.human.aim_assist_deg):
                look, assisted = o['xy'], True
        move = inp.get('move') or [0.0, 0.0]
        n = math.hypot(*move)
        goal = None
        if n > 0.05 and not inp.get('fire'):
            k = 0.6 / n
            goal = [me[0] + move[0] * k, me[1] + move[1] * k]
        mode = 'PLAYER_AIM' if inp.get('fire') else ('PLAYER_MOVE' if goal else 'PLAYER')
        return dict(goal=goal, look=look, fire=bool(inp.get('fire')), mode=mode + ('+ASSIST' if assisted else ''))


def make_bot(spec):
    """'hold' | 'standoff:S_NE' | 'hunt' | 'hunt:4' (return after 4 s without contact)."""
    kind, _, arg = spec.partition(':')
    if kind == 'hold':
        return HoldBot('GOAL', 'hold')
    if kind == 'standoff':
        if not arg:
            raise ValueError('standoff needs a point, e.g. standoff:S_NE')
        return HoldBot(arg, 'standoff')
    if kind == 'hunt':
        return HuntBot(float(arg) if arg else 10.0)
    raise ValueError(f'unknown human bot {spec!r}')
