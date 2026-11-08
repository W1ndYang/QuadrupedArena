"""Live human-vs-quadruped client (top-down, pygame). Optional dependency: pip install pygame.

The player controls the human: WASD / arrows to walk, mouse to aim, hold left button or SPACE to
fire. Aim assist: when firing, a legally visible quadruped within the assist cone of the mouse
direction is tracked automatically (you still have to stand still for 1 s). Only what the human
can legally see is drawn; the quadruped's plans and the referee state are never shown.
The match runs in real time; LLM decisions arrive asynchronously without pausing it.
"""
import math
import time

from .config import GameMap, load_rules
from .humans import PlayerInput
from .runner import run_round

COLORS = dict(bg=(246, 245, 242), floor=(255, 255, 255), grid=(232, 229, 222), high=(154, 165, 177),
              low=(217, 164, 65), goal=(58, 167, 109), quad=(23, 138, 107), human=(44, 79, 163),
              smoke=(150, 150, 150), text=(29, 35, 41), accent=(192, 57, 43), ghost=(160, 190, 180))


class LivePlayer(PlayerInput):
    label = 'player'

    def __init__(self):
        self.state = dict(move=[0.0, 0.0], aim=None, fire=False)
        super().__init__(lambda: self.state)
        self.last_obs = None

    def decide(self, obs):
        self.last_obs = obs
        return super().decide(obs)


class Screen:
    def __init__(self, game_map, rules, width=1200, headless=False):
        import os
        if headless:
            os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
        import pygame
        self.pg = pygame
        pygame.display.init()
        pygame.font.init()
        self.map, self.rules = game_map, rules
        W, H = game_map.size
        self.S = width / (W + 1.0)
        self.size = (width, int((H + 1.6) * self.S))
        self.win = pygame.display.set_mode(self.size)
        pygame.display.set_caption(f'Breach Arena · {game_map.id} {game_map.title} · you are the human')
        self.font = pygame.font.SysFont('notosanscjksc,notosanscjk,microsoftyahei,pingfangsc,arial', 16)

    def w2s(self, p):
        W, H = self.map.size
        return (int(self.size[0] / 2 + p[0] * self.S), int(self.S * 0.3 + (H / 2 - p[1]) * self.S))

    def s2w(self, q):
        W, H = self.map.size
        return [(q[0] - self.size[0] / 2) / self.S, H / 2 - (q[1] - self.S * 0.3) / self.S]

    def draw(self, match, player, banner=None):
        pg, win, S = self.pg, self.win, self.S
        win.fill(COLORS['bg'])
        W, H = self.map.size
        tl = self.w2s((-W / 2, H / 2))
        pg.draw.rect(win, COLORS['floor'], (*tl, int(W * S), int(H * S)))
        for c in self.map.covers:
            (cx, cy), (sx, sy, sz) = c['center'], c['size']
            p = self.w2s((cx - sx / 2, cy + sy / 2))
            pg.draw.rect(win, COLORS['low'] if sz < 1 else COLORS['high'], (*p, max(1, int(sx * S)), max(1, int(sy * S))))
        pg.draw.circle(win, COLORS['goal'], self.w2s(self.map.goal), int(self.map.goal_radius * S), 3)
        obs = player.last_obs
        if obs:
            for s in obs['smokes']:
                surf = pg.Surface(self.size, pg.SRCALPHA)
                alpha = 150 if obs['t'] >= s['active_at'] else 60
                pg.draw.circle(surf, (*COLORS['smoke'], alpha), self.w2s(s['center']), int(s['radius'] * S))
                win.blit(surf, (0, 0))
            mem = obs['memory']
            if obs['opponent'] is None and mem is not None:
                pg.draw.circle(win, COLORS['ghost'], self.w2s(mem['xy']), 8, 2)
            if obs['opponent'] is not None:
                q = obs['opponent']
                c = self.w2s(q['xy'])
                a = q['yaw']
                pts = [(c[0] + 14 * math.cos(a), c[1] - 14 * math.sin(a)),
                       (c[0] + 9 * math.cos(a + 2.5), c[1] - 9 * math.sin(a + 2.5)),
                       (c[0] + 9 * math.cos(a - 2.5), c[1] - 9 * math.sin(a - 2.5))]
                pg.draw.polygon(win, COLORS['quad'], pts)
        h = match.actors[1]
        hc = self.w2s(h.xy)
        pg.draw.circle(win, COLORS['human'], hc, 10)
        pg.draw.line(win, COLORS['human'], hc, (hc[0] + 22 * math.cos(h.yaw), hc[1] - 22 * math.sin(h.yaw)), 3)
        lock = match.lock_age[1]
        if lock > 0:
            pg.draw.rect(win, COLORS['accent'], (hc[0] - 20, hc[1] - 24, int(40 * min(1.0, lock)), 5))
        mode = match.modes[1]
        txt = (f"{match.rules.duration_s - match.t:5.1f} s left   lock {lock:.2f}/1.00 s"
               f"{'   [aim assist]' if 'ASSIST' in mode else ''}{'   [STUNNED]' if match.t < match.human_disabled_until else ''}"
               f"   smoke left (quadruped): {match.inventory}")
        win.blit(self.font.render(txt, True, COLORS['text']), (12, self.size[1] - 26))
        if banner:
            b = self.font.render(banner, True, COLORS['accent'])
            win.blit(b, (self.size[0] // 2 - b.get_width() // 2, 8))
        pg.display.flip()

    def read_input(self, player):
        pg = self.pg
        quit_ = False
        for e in pg.event.get():
            if e.type == pg.QUIT or (e.type == pg.KEYDOWN and e.key == pg.K_ESCAPE):
                quit_ = True
        k = pg.key.get_pressed()
        mx = (k[pg.K_d] or k[pg.K_RIGHT]) - (k[pg.K_a] or k[pg.K_LEFT])
        my = (k[pg.K_w] or k[pg.K_UP]) - (k[pg.K_s] or k[pg.K_DOWN])
        player.state['move'] = [float(mx), float(my)]
        player.state['aim'] = self.s2w(pg.mouse.get_pos())
        player.state['fire'] = bool(pg.mouse.get_pressed()[0] or k[pg.K_SPACE])
        return quit_


def play_live(map_id, quad='rule', rounds=2, out_dir=None, headless_seconds=None, script_input=None, **decider_kw):
    """Run `rounds` rounds against a live player. headless_seconds/script_input are for CI smoke tests."""
    game_map, rules = GameMap.load(map_id), load_rules()
    screen = Screen(game_map, rules, headless=headless_seconds is not None)
    history, results = [], []
    for r in range(1, rounds + 1):
        player = LivePlayer()
        state = dict(quit=False)
        started = time.perf_counter()

        def tick(match, player=player, state=state):
            if script_input is not None:
                script_input(match.t, player)
                if headless_seconds is not None and match.t >= headless_seconds:
                    state['quit'] = True
            else:
                state['quit'] = screen.read_input(player)
            if match.tick % 2 == 0:
                screen.draw(match, player, banner=f'Round {r}/{rounds}')
            if state['quit']:
                match.result = match.result or dict(winner='aborted', reason='player quit', t=round(match.t, 3))
                raise _Stop()

        try:
            replay, viewer = run_round(game_map, rules, quad, player, out_dir, r, history,
                                       pace=None if headless_seconds is not None else 1.0, on_tick=tick,
                                       timing=decider_kw.pop('timing', 'realtime'), **decider_kw)
        except _Stop:
            results.append(dict(round=r, result='aborted', wall=round(time.perf_counter() - started, 1)))
            break
        res = replay['summary']['result']
        results.append(dict(round=r, result=res, viewer=str(viewer) if viewer else None))
        history = replay['next_round_history']
        if headless_seconds is None:
            _wait_key(screen, f"Round {r}: {res['winner']} ({res['reason']}) at {res['t']} s — press Enter")
    screen.pg.quit()
    return results


class _Stop(Exception):
    pass


def _wait_key(screen, text):
    pg = screen.pg
    surf = screen.font.render(text, True, COLORS['accent'])
    screen.win.blit(surf, (screen.size[0] // 2 - surf.get_width() // 2, screen.size[1] // 2))
    pg.display.flip()
    while True:
        for e in pg.event.get():
            if e.type == pg.QUIT or (e.type == pg.KEYDOWN and e.key in (pg.K_RETURN, pg.K_KP_ENTER, pg.K_ESCAPE)):
                return
        time.sleep(0.02)
