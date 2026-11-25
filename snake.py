#!/usr/bin/env python3
"""
Snake AI - Hunter planner + neuroevolution.

Modes
-----
  python snake.py                      # watch the Hunter AI (search-based planner)
  python snake.py --mode neural        # watch the trained neural network
  python snake.py --mode human         # play it yourself (arrow keys / WASD)
  python snake.py --mode train         # evolve the neural network (headless, all CPU cores)
  python snake.py --mode benchmark     # headless stats for an AI (--agent hunter|neural)

Options: --no-hazards, --speed N, --seed N, --games N, --generations N, --agent NAME

In-game keys
------------
  SPACE pause   +/- speed   V toggle path overlay   M cycle mode
  H toggle hazards (next game)   R restart   ESC quit

The game
--------
Eat red apples (+1). Golden apples (+3) appear sometimes and expire quickly.
As the score rises, the arena fights back:
  magenta obstacles (permanent), yellow traps (temporary, fade out),
  orange enemies (wander and sometimes stalk the snake's head).
Walls, obstacles, traps, enemies and your own body are deadly.
Starving (no food for too long) also ends the game.
"""
import os

os.environ.setdefault("SDL_AUDIODRIVER", "dummy")  # avoid ALSA noise on WSL
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import argparse
import json
import math
import random
import sys
import time
from collections import deque, Counter

import numpy as np

# ===============================
# --- Configuration
# ===============================
COLS, ROWS = 40, 30
CELL = 20
PANEL_W = 270
BOARD_W, BOARD_H = COLS * CELL, ROWS * CELL

UP, RIGHT, DOWN, LEFT = (0, -1), (1, 0), (0, 1), (-1, 0)
CLOCKWISE = [UP, RIGHT, DOWN, LEFT]

GOLDEN_POINTS = 3
GOLDEN_CHANCE = 0.18
GOLDEN_TTL = 90
TRAP_TTL = 220
MAX_OBSTACLES = 30
MAX_TRAPS = 6
MAX_ENEMIES = 4
ENEMY_PERIOD = 2          # enemies move every N snake steps
ENEMY_STALK_CHANCE = 0.35
SPAWN_SAFE_DIST = 5       # hazards never spawn this close to the head

BRAIN_FILE = "neural_brain.npz"
BRAIN_STATE_FILE = "neural_state.json"

# Colors
BG = (16, 18, 24)
BG_ALT = (20, 23, 30)
PANEL_BG = (24, 27, 36)
TEXT = (230, 233, 240)
TEXT_DIM = (130, 138, 155)
HEAD_COLOR = (80, 255, 120)
TAIL_COLOR = (10, 70, 40)
APPLE_COLOR = (235, 50, 60)
GOLD_COLOR = (255, 205, 40)
OBSTACLE_COLOR = (200, 60, 220)
TRAP_COLOR = (255, 230, 60)
ENEMY_COLOR = (255, 150, 30)
PATH_COLOR = (80, 200, 255)
STATUS_COLORS = {
    "HUNTING": (90, 230, 120),
    "GOLD RUSH": GOLD_COLOR,
    "STALLING": (90, 170, 255),
    "ESCAPING": (255, 160, 60),
    "DESPERATE": (255, 90, 90),
    "TRAPPED": (255, 50, 50),
    "THINKING": (200, 140, 255),
    "PLAYER": TEXT,
}

DEATH_TEXT = {
    "wall": "hit the wall",
    "self": "bit itself",
    "obstacle": "crashed into an obstacle",
    "trap": "stepped on a trap",
    "enemy": "ran into an enemy",
    "starved": "starved",
    "won": "FILLED THE BOARD!",
    "timeout": "step limit reached",
}


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def is_reverse(a, b):
    return a[0] == -b[0] and a[1] == -b[1]


# ===============================
# --- Game (pure logic, no pygame)
# ===============================
class SnakeGame:
    def __init__(self, cols=COLS, rows=ROWS, hazards=True, seed=None, starve_limit=None):
        self.cols, self.rows = cols, rows
        self.n = cols * rows
        self.hazards = hazards
        self.rng = random.Random(seed)
        self.base_starve = starve_limit
        self.reset()

    # ---- setup -------------------------------------------------------
    def reset(self, seed=None):
        if seed is not None:
            self.rng.seed(seed)
        cx, cy = self.cols // 2, self.rows // 2
        self.body = deque([(cx, cy), (cx - 1, cy), (cx - 2, cy)])
        self.body_set = set(self.body)
        self.direction = RIGHT
        self.score = 0
        self.apples_eaten = 0
        self.steps = 0
        self.steps_since_food = 0
        self.obstacles = set()
        self.traps = {}           # pos -> ttl
        self.enemies = []
        self.golden = None
        self.golden_ttl = 0
        self.alive = True
        self.won = False
        self.death_reason = None
        self.apple = None
        self._place_apple()

    @property
    def head(self):
        return self.body[0]

    @property
    def length(self):
        return len(self.body)

    @property
    def starve_limit(self):
        if self.base_starve is not None:
            return self.base_starve
        # Generous enough for long tail-chasing stalls, but no infinite loops.
        return max(400, int(self.n * 1.2))

    def in_bounds(self, p):
        return 0 <= p[0] < self.cols and 0 <= p[1] < self.rows

    def is_static_hazard(self, p):
        return p in self.obstacles or p in self.traps

    def is_deadly(self, p, growing=False):
        """Would moving the head into p this step kill the snake?"""
        if not self.in_bounds(p) or p in self.obstacles or p in self.traps:
            return True
        if p in self.enemies:
            return True
        if p in self.body_set:
            return not (p == self.body[-1] and not growing)
        return False

    def _neighbors(self, p):
        x, y = p
        for dx, dy in CLOCKWISE:
            q = (x + dx, y + dy)
            if 0 <= q[0] < self.cols and 0 <= q[1] < self.rows:
                yield q

    def _static_reachable(self, start, extra_block=None):
        """Cells reachable from start ignoring the (moving) body and enemies."""
        seen = {start}
        q = deque([start])
        while q:
            c = q.popleft()
            for nb in self._neighbors(c):
                if nb in seen or nb in self.obstacles or nb in self.traps or nb == extra_block:
                    continue
                seen.add(nb)
                q.append(nb)
        return seen

    def _occupied(self):
        occ = set(self.body_set) | self.obstacles | set(self.traps) | set(self.enemies)
        if self.apple:
            occ.add(self.apple)
        if self.golden:
            occ.add(self.golden)
        return occ

    def _place_apple(self):
        self.apple = None
        reachable = self._static_reachable(self.head)
        occ = self._occupied()
        cands = [c for c in reachable if c not in occ]
        if not cands:
            cands = [(x, y) for x in range(self.cols) for y in range(self.rows) if (x, y) not in occ]
        if not cands:
            self.apple = None
            self.won = True
            self.alive = False
            self.death_reason = "won"
            return
        self.apple = self.rng.choice(cands)

    def _place_golden(self):
        reachable = self._static_reachable(self.head)
        occ = self._occupied()
        cands = [c for c in reachable if c not in occ and manhattan(c, self.head) >= 4]
        if cands:
            self.golden = self.rng.choice(cands)
            self.golden_ttl = GOLDEN_TTL

    def _hazard_spawn_cells(self, min_dist=SPAWN_SAFE_DIST):
        occ = self._occupied()
        hx, hy = self.head
        dx, dy = self.direction
        avoid = {(hx + dx * i, hy + dy * i) for i in range(1, 9)}
        for bx, by in self.body:          # never pinch the body's corridors
            avoid.update(((bx + 1, by), (bx - 1, by), (bx, by + 1), (bx, by - 1)))
        return [
            (x, y)
            for x in range(self.cols)
            for y in range(self.rows)
            if (x, y) not in occ and (x, y) not in avoid and manhattan((x, y), self.head) >= min_dist
        ]

    def _spawn_static(self, kind):
        """Spawn an obstacle/trap without ever sealing off part of the arena."""
        cands = self._hazard_spawn_cells()
        if not cands:
            return
        before = len(self._static_reachable(self.head))
        self.rng.shuffle(cands)
        for c in cands[:40]:
            if len(self._static_reachable(self.head, extra_block=c)) == before - 1:
                if kind == "obstacle":
                    self.obstacles.add(c)
                else:
                    self.traps[c] = TRAP_TTL
                return

    def _spawn_enemy(self):
        cands = self._hazard_spawn_cells(min_dist=10)
        if cands:
            self.enemies.append(self.rng.choice(cands))

    def _escalate(self):
        if not self.hazards:
            return
        a = self.apples_eaten
        while len(self.obstacles) < min(a // 3, MAX_OBSTACLES):
            n = len(self.obstacles)
            self._spawn_static("obstacle")
            if len(self.obstacles) == n:
                break
        if a % 5 == 0 and len(self.traps) < MAX_TRAPS:
            self._spawn_static("trap")
        while len(self.enemies) < min(a // 7, MAX_ENEMIES):
            n = len(self.enemies)
            self._spawn_enemy()
            if len(self.enemies) == n:
                break

    def _move_enemies(self):
        taken = set(self.enemies)
        for i, e in enumerate(self.enemies):
            taken.discard(e)
            opts = [
                q for q in self._neighbors(e)
                if q not in self.body_set and q not in self.obstacles and q not in self.traps
                and q not in taken and q != self.apple and q != self.golden
            ]
            if opts:
                if self.rng.random() < ENEMY_STALK_CHANCE:
                    nxt = min(opts, key=lambda q: manhattan(q, self.head))
                else:
                    nxt = self.rng.choice(opts + [e])
            else:
                nxt = e
            self.enemies[i] = nxt
            taken.add(nxt)

    # ---- simulation --------------------------------------------------
    def step(self, direction):
        """Advance one tick. Returns points gained this step."""
        if not self.alive:
            return 0
        if is_reverse(direction, self.direction):
            direction = self.direction
        self.direction = direction
        hx, hy = self.head
        nh = (hx + direction[0], hy + direction[1])
        self.steps += 1
        self.steps_since_food += 1

        ate_apple = nh == self.apple
        ate_golden = nh == self.golden
        growing = ate_apple or ate_golden

        if not self.in_bounds(nh):
            return self._die("wall")
        if nh in self.obstacles:
            return self._die("obstacle")
        if nh in self.traps:
            return self._die("trap")
        if nh in self.enemies:
            return self._die("enemy")
        if nh in self.body_set and not (nh == self.body[-1] and not growing):
            return self._die("self")

        if not growing:
            tail = self.body.pop()
            self.body_set.discard(tail)
        self.body.appendleft(nh)
        self.body_set.add(nh)

        gained = 0
        if ate_golden:
            gained += GOLDEN_POINTS
            self.golden = None
            self.golden_ttl = 0
        if ate_apple:
            gained += 1
            self.apples_eaten += 1
            self._place_apple()
            if not self.alive:      # board filled
                self.score += gained
                return gained
            self._escalate()
            if self.hazards and self.golden is None and self.rng.random() < GOLDEN_CHANCE:
                self._place_golden()
        if gained:
            self.score += gained
            self.steps_since_food = 0

        # timers
        if self.golden is not None:
            self.golden_ttl -= 1
            if self.golden_ttl <= 0:
                self.golden = None
        if self.traps:
            for p in list(self.traps):
                self.traps[p] -= 1
                if self.traps[p] <= 0:
                    del self.traps[p]
        if self.enemies and self.steps % ENEMY_PERIOD == 0:
            self._move_enemies()

        if self.steps_since_food > self.starve_limit:
            self._die("starved")
        return gained

    def _die(self, reason):
        self.alive = False
        self.death_reason = reason
        return 0


# ===============================
# --- Path planning toolkit (flat-index grid, time-aware)
# ===============================
INF_T = 1 << 30


class Planner:
    """
    Fast grid search helpers shared by the Hunter AI and the neural agent.

    The key trick is a *time-aware* BFS: each body segment is treated as an
    obstacle only until the tail has slid past it.  Segment i (head = 0) of a
    snake of length L vacates its cell after L - i moves, so a path may pass
    through a cell that is body *now* if it arrives there late enough.
    """

    def __init__(self, cols=COLS, rows=ROWS):
        self.cols, self.rows, self.n = cols, rows, cols * rows
        self.nbrs = []
        for i in range(self.n):
            x, y = i % cols, i // cols
            lst = []
            for dx, dy in CLOCKWISE:
                nx, ny = x + dx, y + dy
                if 0 <= nx < cols and 0 <= ny < rows:
                    lst.append(ny * cols + nx)
            self.nbrs.append(lst)

    def idx(self, p):
        return p[1] * self.cols + p[0]

    def pos(self, i):
        return (i % self.cols, i // self.cols)

    def static_times(self, game, inflate=False):
        """
        bt[c] = earliest step at which cell c may be entered (ignoring the body).
        With inflate=True, cells next to an enemy are also treated as walls,
        because the enemy may step there before we arrive.
        """
        bt = [0] * self.n
        for p in game.obstacles:
            bt[self.idx(p)] = INF_T
        for p, ttl in game.traps.items():
            bt[self.idx(p)] = ttl + 1          # traps vanish after their ttl
        for p in game.enemies:
            e = self.idx(p)
            bt[e] = INF_T
            if inflate:
                for nb in self.nbrs[e]:
                    if self.pos(nb) not in game.body_set:
                        bt[nb] = INF_T
        return bt

    @staticmethod
    def body_times(bt_static, body):
        bt = bt_static[:]
        L = len(body)
        for i, c in enumerate(body):
            t = L - i
            if t > bt[c]:
                bt[c] = t
        return bt

    def bfs(self, start, bt, target=-1):
        dist = [-1] * self.n
        parent = [-1] * self.n
        dist[start] = 0
        q = deque([start])
        nbrs = self.nbrs
        while q:
            c = q.popleft()
            d = dist[c] + 1
            for nb in nbrs[c]:
                if dist[nb] < 0 and bt[nb] <= d:
                    dist[nb] = d
                    parent[nb] = c
                    if nb == target:
                        return dist, parent
                    q.append(nb)
        return dist, parent

    def plain_dist(self, start, bt):
        """Plain BFS distance map from start; cells with bt > 1 are walls."""
        dist = [-1] * self.n
        dist[start] = 0
        q = deque([start])
        nbrs = self.nbrs
        while q:
            c = q.popleft()
            d = dist[c] + 1
            for nb in nbrs[c]:
                if dist[nb] < 0 and bt[nb] <= 1:
                    dist[nb] = d
                    q.append(nb)
        return dist

    @staticmethod
    def path_to(parent, start, target):
        path = []
        c = target
        while c != start and c != -1:
            path.append(c)
            c = parent[c]
        path.reverse()
        return path

    def body_after(self, path, body, grows=1):
        """Virtual body after following path (eating at the end)."""
        new_len = len(body) + grows
        return (path[::-1] + body)[:new_len]

    def tail_reachable(self, body, bt_static):
        bt = self.body_times(bt_static, body)
        tail = body[-1]
        dist, _ = self.bfs(body[0], bt, target=tail)
        return dist[tail] >= 0

    def evaluate(self, body, bt_static):
        """(distance from head to tail or -1, reachable area) for a virtual body."""
        bt = self.body_times(bt_static, body)
        dist, _ = self.bfs(body[0], bt)
        area = sum(1 for d in dist if d > 0)
        return dist[body[-1]], area


# ===============================
# --- Hunter AI (search-based)
# ===============================
class HunterAI:
    """
    1. HUNT: shortest time-aware path to the best food (golden apple if it can be
       reached before it expires). Before committing, simulate eating it and verify
       the snake can still reach its own tail afterwards -> it never paints itself
       into a corner.
    2. STALL: if hunting is unsafe, chase the tail along the longest route, which
       buys time for the body to clear while staying guaranteed-safe.
    3. ESCAPE: if even that fails, pick the move with the most room to breathe,
       keeping away from enemies.
    4. DESPERATE: when close to starving, accept riskier food paths that still
       leave enough open space.
    """

    name = "Hunter AI"

    def __init__(self, cols=COLS, rows=ROWS):
        self.P = Planner(cols, rows)
        self.status = "THINKING"
        self.path = []          # list of (x, y) for the overlay

    def decide(self, game):
        P = self.P
        self._head = game.head
        head = P.idx(game.head)
        body = [P.idx(p) for p in game.body]
        loose = P.static_times(game)
        strict = P.static_times(game, inflate=True) if game.enemies else loose

        # Plan with enemy danger zones first; relax only when that finds nothing.
        variants = [strict, loose] if strict is not loose else [loose]
        hungry = game.steps_since_food > game.starve_limit * 0.5
        for bt_static in variants:
            bt = P.body_times(bt_static, body)
            dist, parent = P.bfs(head, bt)
            paths = []
            for t, label in self._targets(game, dist):
                path = P.path_to(parent, head, t)
                paths.append(path)
                if P.tail_reachable(P.body_after(path, body), strict):
                    return self._go(path, label)
            if hungry:
                # Close to starving: accept a path that leaves room for our whole body.
                for path in paths:
                    vb = P.body_after(path, body)
                    _, area = P.evaluate(vb, loose)
                    if area >= len(vb) or game.steps_since_food > game.starve_limit * 0.85:
                        return self._go(path, "DESPERATE")

        return self._survive(game, body, strict, loose)

    def _targets(self, game, dist):
        P = self.P
        out = []
        if game.golden is not None:
            g = P.idx(game.golden)
            if 0 < dist[g] <= game.golden_ttl:
                out.append((g, "GOLD RUSH"))
        if game.apple is not None:
            a = P.idx(game.apple)
            if dist[a] > 0:
                out.append((a, "HUNTING"))
        return out

    def _go(self, path, label):
        self.status = label
        self.path = [self.P.pos(c) for c in path]
        nx, ny = self.path[0]
        return (nx - self._head[0], ny - self._head[1])

    def _survive(self, game, body, strict, loose):
        P = self.P
        apple = P.idx(game.apple) if game.apple else -1
        golden = P.idx(game.golden) if game.golden else -1
        bt_now = P.body_times(loose, body)
        best_key, best_dir = None, game.direction
        hx, hy = game.head
        for d in CLOCKWISE:
            if is_reverse(d, game.direction):
                continue
            nx, ny = hx + d[0], hy + d[1]
            if not (0 <= nx < P.cols and 0 <= ny < P.rows):
                continue
            c = ny * P.cols + nx
            if bt_now[c] > 1:
                continue                      # instant death
            grows = c == apple or c == golden
            vb = [c] + (body if grows else body[:-1])
            tail_d, area = P.evaluate(vb, loose)
            if strict is not loose:
                s_tail, s_area = P.evaluate(vb, strict)
            else:
                s_tail, s_area = tail_d, area
            enemy_near = sum(1 for e in game.enemies if manhattan(e, (nx, ny)) <= 2)
            key = (
                s_tail >= 0,                   # safe even if enemies close in
                tail_d >= 0,                   # safe with enemies where they are
                -enemy_near,
                tail_d if tail_d >= 0 else 0,  # longest tail chase buys most time
                s_area,
                area,
            )
            if best_key is None or key > best_key:
                best_key, best_dir = key, d
        if best_key is None:
            self.status = "TRAPPED"
        else:
            self.status = "STALLING" if best_key[1] else "ESCAPING"
        self.path = []
        return best_dir


# ===============================
# --- Neural agent (evolved MLP over planner-derived features)
# ===============================
N_INPUTS = 25
N_HIDDEN = 24
N_OUTPUTS = 3
GENOME_SIZE = N_INPUTS * N_HIDDEN + N_HIDDEN + N_HIDDEN * N_OUTPUTS + N_OUTPUTS


class NeuralNet:
    """Tiny numpy MLP (25 -> 24 tanh -> 3). Weights live in one flat genome."""

    def __init__(self, genome=None, rng=None):
        if genome is None:
            rng = rng or np.random.default_rng()
            genome = rng.normal(0, 0.5, GENOME_SIZE)
        self.genome = np.asarray(genome, dtype=np.float64)
        g, o = self.genome, 0
        self.w1 = g[o:o + N_INPUTS * N_HIDDEN].reshape(N_HIDDEN, N_INPUTS); o += N_INPUTS * N_HIDDEN
        self.b1 = g[o:o + N_HIDDEN]; o += N_HIDDEN
        self.w2 = g[o:o + N_HIDDEN * N_OUTPUTS].reshape(N_OUTPUTS, N_HIDDEN); o += N_HIDDEN * N_OUTPUTS
        self.b2 = g[o:o + N_OUTPUTS]

    def forward(self, x):
        return self.w2 @ np.tanh(self.w1 @ x + self.b1) + self.b2

    def save(self, path=BRAIN_FILE, meta=None):
        np.savez(path, genome=self.genome, n_inputs=N_INPUTS, n_hidden=N_HIDDEN,
                 meta=json.dumps(meta or {}))

    @staticmethod
    def load(path=BRAIN_FILE):
        if not os.path.exists(path):
            return None, None
        try:
            data = np.load(path, allow_pickle=False)
            if int(data["n_inputs"]) != N_INPUTS or int(data["n_hidden"]) != N_HIDDEN:
                print(f"⚠️  {path} has an incompatible architecture; ignoring it.")
                return None, None
            return NeuralNet(data["genome"]), json.loads(str(data["meta"]))
        except Exception as e:  # corrupt file etc.
            print(f"⚠️  Could not load {path}: {e}")
            return None, None


class NeuralAgent:
    """
    The network chooses between straight / right / left.  Instead of raw pixels
    it sees what a good player thinks about for each option:
      deadly?  how much room afterwards?  can I still reach my tail?
      is it the shortest way to the apple / golden apple?  enemies nearby?
    """

    name = "Neural AI"

    def __init__(self, net=None, cols=COLS, rows=ROWS):
        self.net = net or NeuralNet()
        self.P = Planner(cols, rows)
        self.status = "THINKING"
        self.path = []

    def relative_dirs(self, direction):
        i = CLOCKWISE.index(direction)
        return [CLOCKWISE[i], CLOCKWISE[(i + 1) % 4], CLOCKWISE[(i - 1) % 4]]

    def features(self, game):
        P = self.P
        body = [P.idx(p) for p in game.body]
        bt_static = P.static_times(game)
        bt = P.body_times(bt_static, body)
        L = len(body)
        apple = P.idx(game.apple) if game.apple else -1
        golden = P.idx(game.golden) if game.golden else -1
        fd = P.plain_dist(apple, bt) if apple >= 0 else None
        gd = P.plain_dist(golden, bt) if golden >= 0 else None
        hx, hy = game.head
        scale = float(P.cols + P.rows)

        per_move = []
        for d in self.relative_dirs(game.direction):
            nx, ny = hx + d[0], hy + d[1]
            if not (0 <= nx < P.cols and 0 <= ny < P.rows) or bt[ny * P.cols + nx] > 1:
                per_move.append(None)
                continue
            c = ny * P.cols + nx
            grows = c == apple or c == golden
            vb = [c] + (body if grows else body[:-1])
            tail_d, area = P.evaluate(vb, bt_static)
            f_dist = 0 if c == apple else (fd[c] if fd and fd[c] >= 0 else -1)
            g_dist = 0 if c == golden else (gd[c] if gd and gd[c] >= 0 else -1)
            enemy = sum(1 for e in game.enemies if manhattan(e, (nx, ny)) <= 2)
            per_move.append((area, tail_d, f_dist, g_dist, enemy))

        f_best = min((m[2] for m in per_move if m and m[2] >= 0), default=-1)
        g_best = min((m[3] for m in per_move if m and m[3] >= 0), default=-1)
        x = []
        for m in per_move:
            if m is None:
                x += [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
                continue
            area, tail_d, f_dist, g_dist, enemy = m
            x += [
                0.0,
                min(area / (2.0 * L + 10.0), 1.0),
                1.0 if tail_d >= 0 else 0.0,
                1.0 if f_dist >= 0 and f_dist == f_best else 0.0,
                min(f_dist / scale, 1.0) if f_dist >= 0 else 1.0,
                1.0 if g_dist >= 0 and g_dist == g_best else 0.0,
                min(enemy / 2.0, 1.0),
            ]
        if game.apple:
            ang = math.atan2(game.apple[1] - hy, game.apple[0] - hx) - math.atan2(game.direction[1], game.direction[0])
        else:
            ang = 0.0
        x += [L / P.n, game.steps_since_food / game.starve_limit, math.sin(ang), math.cos(ang)]
        return np.array(x, dtype=np.float64)

    def decide(self, game):
        x = self.features(game)
        choice = int(np.argmax(self.net.forward(x)))
        f = x[choice * 7:(choice + 1) * 7]
        if f[0]:
            self.status = "TRAPPED"
        elif f[5]:
            self.status = "GOLD RUSH"
        elif f[3]:
            self.status = "HUNTING"
        elif f[2]:
            self.status = "STALLING"
        else:
            self.status = "ESCAPING"
        return self.relative_dirs(game.direction)[choice]


class HumanAgent:
    name = "Human"

    def __init__(self):
        self.status = "PLAYER"
        self.path = []
        self.queue = deque()

    def push(self, d):
        if len(self.queue) < 3:
            self.queue.append(d)

    def decide(self, game):
        while self.queue:
            d = self.queue.popleft()
            if not is_reverse(d, game.direction) and d != game.direction:
                return d
        return game.direction


def make_agent(kind, quiet=False):
    if kind == "hunter":
        return HunterAI()
    if kind == "neural":
        net, meta = NeuralNet.load()
        if net is None:
            if not quiet:
                print(f"🧠 No trained brain found ({BRAIN_FILE}). Run: python snake.py --mode train")
            return None
        if not quiet and meta:
            print(f"🧠 Loaded neural brain: generation {meta.get('generation')}, "
                  f"avg score {meta.get('avg_score', 0):.1f}")
        return NeuralAgent(net)
    if kind == "human":
        return HumanAgent()
    raise ValueError(kind)


def play_episode(game, agent, max_steps=None):
    while game.alive and (max_steps is None or game.steps < max_steps):
        game.step(agent.decide(game))
    if game.alive:
        game.death_reason = "timeout"
    return game.score, game.steps, game.death_reason


# ===============================
# --- Neuroevolution trainer (headless, multi-core)
# ===============================
TRAIN_POP = 64
TRAIN_ELITES = 6
TRAIN_EPISODES = 3
TRAIN_MAX_STEPS = 1500
TRAIN_STARVE = 300


def _evaluate_genome(args):
    genome, seeds, hazards = args
    agent = NeuralAgent(NeuralNet(genome))
    game = SnakeGame(hazards=hazards, starve_limit=TRAIN_STARVE)
    fitness, scores = 0.0, []
    for s in seeds:
        game.reset(seed=s)
        score, steps, reason = play_episode(game, agent, TRAIN_MAX_STEPS)
        # Food is what matters; survival breaks ties; crashing is worse than starving.
        fitness += score * 100.0 + min(steps, 400) * 0.1
        if reason in ("wall", "self", "obstacle", "trap", "enemy"):
            fitness -= 25.0
        scores.append(score)
    return fitness / len(seeds), float(np.mean(scores))


def train(generations=None, hazards=True, seed=None):
    from multiprocessing import Pool, cpu_count

    rng = np.random.default_rng(seed)
    state = {}
    if os.path.exists(BRAIN_STATE_FILE):
        try:
            with open(BRAIN_STATE_FILE) as f:
                state = json.load(f)
        except Exception:
            state = {}
    best_net, meta = NeuralNet.load()
    gen = int(state.get("generation", 0)) if best_net is not None else 0
    best_fit = float(state.get("best_fitness", -1e9)) if best_net is not None else -1e9

    population = [NeuralNet(rng=rng).genome for _ in range(TRAIN_POP)]
    if best_net is not None:
        print(f"🔁 Resuming from generation {gen} (best fitness {best_fit:.1f})")
        population[0] = best_net.genome.copy()
        for i in range(1, TRAIN_POP // 3):
            population[i] = best_net.genome + rng.normal(0, 0.1, GENOME_SIZE)
    else:
        print("🆕 Starting a fresh population")

    sigma = 0.25
    workers = max(1, cpu_count())
    print(f"⚙️  Population {TRAIN_POP}, {TRAIN_EPISODES} episodes each, {workers} worker processes.")
    print("   Ctrl+C to stop; the best brain is saved after each generation.\n")

    def save(net, gen, fit, avg):
        net.save(BRAIN_FILE, {"generation": gen, "fitness": fit, "avg_score": avg, "time": time.time()})
        with open(BRAIN_STATE_FILE, "w") as f:
            json.dump({"version": 2, "generation": gen, "best_fitness": fit, "avg_score": avg}, f, indent=2)

    try:
        with Pool(workers) as pool:
            stop_at = None if generations is None else gen + generations
            while stop_at is None or gen < stop_at:
                gen += 1
                t0 = time.time()
                # Same seeds for everyone this generation = fair comparison.
                seeds = [int(x) for x in rng.integers(0, 2**31 - 1, TRAIN_EPISODES)]
                results = pool.map(_evaluate_genome, [(g, seeds, hazards) for g in population])
                fits = np.array([r[0] for r in results])
                scores = np.array([r[1] for r in results])
                order = np.argsort(-fits)

                champ = order[0]
                # Re-check the champion on fresh seeds before trusting it (luck filter).
                val_seeds = [int(x) for x in rng.integers(0, 2**31 - 1, 6)]
                val_fit, val_score = _evaluate_genome((population[champ], val_seeds, hazards))
                improved = val_fit > best_fit
                if improved:
                    best_fit = val_fit
                    save(NeuralNet(population[champ]), gen, val_fit, val_score)

                print(f"Gen {gen:4d} | best {scores[champ]:6.1f} | mean {scores.mean():6.1f} | "
                      f"validated {val_score:6.1f} | σ {sigma:.3f} | {time.time() - t0:5.1f}s"
                      + ("  💾 new best" if improved else ""))

                # Adaptive mutation strength: shrink when improving, grow when stuck.
                sigma = max(0.03, sigma * 0.95) if improved else min(0.5, sigma * 1.03)

                elites = [population[i] for i in order[:TRAIN_ELITES]]
                new_pop = [e.copy() for e in elites]
                ranked = [population[i] for i in order]
                while len(new_pop) < TRAIN_POP:
                    a = ranked[min(rng.integers(0, TRAIN_POP, 4))]   # tournament of 4
                    b = ranked[min(rng.integers(0, TRAIN_POP, 4))]
                    mask = rng.random(GENOME_SIZE) < 0.5
                    child = np.where(mask, a, b)
                    mut = rng.random(GENOME_SIZE) < 0.15
                    child = child + mut * rng.normal(0, sigma, GENOME_SIZE)
                    new_pop.append(child)
                population = new_pop
    except KeyboardInterrupt:
        print("\n🛑 Training stopped.")
    if os.path.exists(BRAIN_STATE_FILE):   # remember how far we got for the next resume
        try:
            with open(BRAIN_STATE_FILE) as f:
                st = json.load(f)
            st["generation"] = gen
            with open(BRAIN_STATE_FILE, "w") as f:
                json.dump(st, f, indent=2)
        except Exception:
            pass
    print(f"✅ Best brain saved to {BRAIN_FILE} (fitness {best_fit:.1f})")


# ===============================
# --- Benchmark
# ===============================
def _bench_one(args):
    kind, seed, hazards = args
    agent = make_agent(kind, quiet=True)
    game = SnakeGame(hazards=hazards, seed=seed)
    t0 = time.time()
    score, steps, reason = play_episode(game, agent, max_steps=50000)
    return score, game.length, steps, reason, time.time() - t0


def benchmark(kind, games, hazards, seed=None):
    from multiprocessing import Pool, cpu_count

    if make_agent(kind) is None:
        return
    base = seed if seed is not None else random.randrange(1 << 30)
    print(f"📊 Benchmarking {kind} over {games} games (hazards {'on' if hazards else 'off'})...")
    t0 = time.time()
    with Pool(min(cpu_count(), games)) as pool:
        res = pool.map(_bench_one, [(kind, base + i, hazards) for i in range(games)])
    scores = [r[0] for r in res]
    lengths = [r[1] for r in res]
    print(f"   score  avg {np.mean(scores):7.1f}  median {np.median(scores):7.1f}  "
          f"min {min(scores)}  max {max(scores)}")
    print(f"   length avg {np.mean(lengths):7.1f}  max {max(lengths)}  (board has {COLS * ROWS} cells)")
    print(f"   steps  avg {np.mean([r[2] for r in res]):7.0f}")
    print(f"   deaths {dict(Counter(r[3] for r in res))}")
    print(f"   {time.time() - t0:.1f}s wall time")


# ===============================
# --- Visual front-end (pygame)
# ===============================
SPEEDS = [4, 8, 12, 16, 20, 30, 45, 60, 90, 120, 180, 300, 600, 1200]
MODES = ["hunter", "neural", "human"]


class App:
    def __init__(self, mode="hunter", hazards=True, speed=None, seed=None):
        import pygame
        self.pg = pygame
        pygame.init()
        self.screen = pygame.display.set_mode((BOARD_W + PANEL_W, BOARD_H))
        pygame.display.set_caption("Snake AI")
        self.clock = pygame.time.Clock()
        self.font_big = pygame.font.Font(None, 44)
        self.font = pygame.font.Font(None, 28)
        self.font_small = pygame.font.Font(None, 21)
        self.hazards = hazards
        self.game = SnakeGame(hazards=hazards, seed=seed)
        self.show_path = True
        self.paused = False
        self.stats = {m: {"games": 0, "best": 0, "recent": deque(maxlen=50)} for m in MODES}
        self.deaths = Counter()
        self.death_timer = 0
        self.frame = 0
        self._board_bg = self._make_board_bg()
        self.set_mode(mode)
        if speed is not None:
            self.speed_idx = min(range(len(SPEEDS)), key=lambda i: abs(SPEEDS[i] - speed))

    # ---- mode / speed ------------------------------------------------
    def set_mode(self, mode):
        agent = make_agent(mode)
        if agent is None:           # neural requested but untrained
            mode, agent = "hunter", make_agent("hunter")
        self.mode, self.agent = mode, agent
        self.speed_idx = SPEEDS.index(12) if mode == "human" else SPEEDS.index(30)
        self.game.reset()
        self.death_timer = 0

    @property
    def speed(self):
        return SPEEDS[self.speed_idx]

    # ---- main loop ---------------------------------------------------
    def run(self):
        pg = self.pg
        acc = 0.0
        while True:
            dt = self.clock.tick(60) / 1000.0
            self.frame += 1
            for ev in pg.event.get():
                if ev.type == pg.QUIT:
                    return
                if ev.type == pg.KEYDOWN:
                    if not self.on_key(ev.key):
                        return
            if self.death_timer > 0:
                self.death_timer -= 1
                if self.death_timer == 0:
                    self.game.reset()
            elif not self.paused:
                acc += dt * self.speed
                n = int(acc)
                acc -= n
                for _ in range(min(n, 400)):
                    self.game.step(self.agent.decide(self.game))
                    if not self.game.alive:
                        self.on_death()
                        acc = 0
                        break
            self.draw()

    def on_key(self, k):
        pg = self.pg
        if k == pg.K_ESCAPE or k == pg.K_q:
            return False
        if self.mode == "human":
            keymap = {pg.K_UP: UP, pg.K_w: UP, pg.K_DOWN: DOWN, pg.K_s: DOWN,
                      pg.K_LEFT: LEFT, pg.K_a: LEFT, pg.K_RIGHT: RIGHT, pg.K_d: RIGHT}
            if k in keymap:
                self.agent.push(keymap[k])
                return True
        if k == pg.K_SPACE or k == pg.K_p:
            self.paused = not self.paused
        elif k in (pg.K_PLUS, pg.K_EQUALS, pg.K_KP_PLUS):
            self.speed_idx = min(self.speed_idx + 1, len(SPEEDS) - 1)
        elif k in (pg.K_MINUS, pg.K_KP_MINUS):
            self.speed_idx = max(self.speed_idx - 1, 0)
        elif k == pg.K_v:
            self.show_path = not self.show_path
        elif k == pg.K_m:
            self.set_mode(MODES[(MODES.index(self.mode) + 1) % len(MODES)])
        elif k == pg.K_h:
            self.hazards = not self.hazards
            self.game.hazards = self.hazards
            self.game.reset()
        elif k == pg.K_r:
            self.game.reset()
            self.death_timer = 0
        return True

    def on_death(self):
        st = self.stats[self.mode]
        st["games"] += 1
        st["best"] = max(st["best"], self.game.score)
        st["recent"].append(self.game.score)
        self.deaths[self.game.death_reason] += 1
        self.death_timer = 150 if self.game.won else 75
        print(f"[{self.mode}] game {st['games']}: score {self.game.score}, length {self.game.length}, "
              f"{DEATH_TEXT.get(self.game.death_reason, self.game.death_reason)}")

    # ---- drawing -----------------------------------------------------
    def _make_board_bg(self):
        surf = self.pg.Surface((BOARD_W, BOARD_H))
        surf.fill(BG)
        for x in range(COLS):
            for y in range(ROWS):
                if (x + y) % 2:
                    surf.fill(BG_ALT, (x * CELL, y * CELL, CELL, CELL))
        return surf

    def cell_rect(self, p, inset=0):
        return self.pg.Rect(p[0] * CELL + inset, p[1] * CELL + inset, CELL - 2 * inset, CELL - 2 * inset)

    def center(self, p):
        return (p[0] * CELL + CELL // 2, p[1] * CELL + CELL // 2)

    def draw(self):
        pg, g, scr = self.pg, self.game, self.screen
        scr.blit(self._board_bg, (0, 0))
        pulse = 0.5 + 0.5 * math.sin(self.frame * 0.15)

        # hazards
        for p in g.obstacles:
            r = self.cell_rect(p, 1)
            pg.draw.rect(scr, (110, 30, 120), r, border_radius=4)
            pg.draw.rect(scr, OBSTACLE_COLOR, r.inflate(-4, -4), border_radius=3)
        if g.traps:
            overlay = pg.Surface((CELL, CELL), pg.SRCALPHA)
            for p, ttl in g.traps.items():
                overlay.fill((0, 0, 0, 0))
                a = int(80 + 175 * min(1.0, ttl / TRAP_TTL))
                pg.draw.rect(overlay, (*TRAP_COLOR, a // 3), (1, 1, CELL - 2, CELL - 2), border_radius=4)
                pg.draw.line(overlay, (*TRAP_COLOR, a), (4, 4), (CELL - 5, CELL - 5), 3)
                pg.draw.line(overlay, (*TRAP_COLOR, a), (CELL - 5, 4), (4, CELL - 5), 3)
                scr.blit(overlay, (p[0] * CELL, p[1] * CELL))
        for e in g.enemies:
            c = self.center(e)
            pg.draw.circle(scr, ENEMY_COLOR, c, CELL // 2 - 1)
            for sx in (-4, 4):
                pg.draw.circle(scr, (40, 10, 0), (c[0] + sx, c[1] - 2), 2)
            pg.draw.line(scr, (40, 10, 0), (c[0] - 6, c[1] - 7), (c[0] - 2, c[1] - 5), 2)
            pg.draw.line(scr, (40, 10, 0), (c[0] + 6, c[1] - 7), (c[0] + 2, c[1] - 5), 2)

        # planned path
        if self.show_path and len(self.agent.path) > 0 and self.death_timer == 0:
            pts = [self.center(g.head)] + [self.center(p) for p in self.agent.path]
            surf = pg.Surface((BOARD_W, BOARD_H), pg.SRCALPHA)
            col = STATUS_COLORS.get(self.agent.status, PATH_COLOR)
            if len(pts) > 1:
                pg.draw.lines(surf, (*col, 170), False, pts, 3)
            pg.draw.circle(surf, (*col, 200), pts[-1], 5, 2)
            scr.blit(surf, (0, 0))

        # food
        if g.apple:
            c = self.center(g.apple)
            glow = pg.Surface((CELL * 2, CELL * 2), pg.SRCALPHA)
            pg.draw.circle(glow, (*APPLE_COLOR, int(40 + 40 * pulse)), (CELL, CELL), CELL - 2)
            scr.blit(glow, (c[0] - CELL, c[1] - CELL))
            pg.draw.circle(scr, APPLE_COLOR, (c[0], c[1] + 1), CELL // 2 - 2)
            pg.draw.circle(scr, (255, 150, 150), (c[0] - 3, c[1] - 2), 2)
            pg.draw.line(scr, (120, 70, 20), (c[0], c[1] - 6), (c[0] + 1, c[1] - 9), 2)
            pg.draw.ellipse(scr, (70, 200, 80), (c[0] + 1, c[1] - 10, 6, 4))
        if g.golden:
            c = self.center(g.golden)
            glow = pg.Surface((CELL * 3, CELL * 3), pg.SRCALPHA)
            pg.draw.circle(glow, (*GOLD_COLOR, int(50 + 60 * pulse)), (CELL * 3 // 2,) * 2, CELL + 2)
            scr.blit(glow, (c[0] - CELL * 3 // 2, c[1] - CELL * 3 // 2))
            pg.draw.circle(scr, GOLD_COLOR, c, CELL // 2 - 1)
            pg.draw.circle(scr, (255, 250, 200), (c[0] - 3, c[1] - 3), 3)
            frac = g.golden_ttl / GOLDEN_TTL
            rect = pg.Rect(0, 0, CELL + 8, CELL + 8)
            rect.center = c
            pg.draw.arc(scr, GOLD_COLOR, rect, math.pi / 2, math.pi / 2 + 2 * math.pi * frac, 2)

        self._draw_snake()

        if self.death_timer > 0:
            fade = pg.Surface((BOARD_W, BOARD_H), pg.SRCALPHA)
            color = (40, 200, 90) if g.won else (200, 30, 30)
            fade.fill((*color, min(110, self.death_timer * 3)))
            scr.blit(fade, (0, 0))
            msg = DEATH_TEXT.get(g.death_reason, g.death_reason or "")
            self._text_center(("🏆 " if g.won else "") + f"Snake {msg}!", BOARD_H // 2 - 20, self.font_big)
            self._text_center(f"Score {g.score}  •  Length {g.length}", BOARD_H // 2 + 20, self.font)
        elif self.paused:
            self._text_center("PAUSED", BOARD_H // 2, self.font_big)

        self._draw_panel()
        pg.display.flip()

    def _text_center(self, text, y, font):
        surf = font.render(text, True, TEXT)
        shadow = font.render(text, True, (0, 0, 0))
        r = surf.get_rect(center=(BOARD_W // 2, y))
        self.screen.blit(shadow, r.move(2, 2))
        self.screen.blit(surf, r)

    def _draw_snake(self):
        pg, g, scr = self.pg, self.game, self.screen
        body = list(g.body)
        n = len(body)
        for i in range(n - 1, -1, -1):
            t = i / max(1, n - 1)
            col = tuple(int(HEAD_COLOR[k] * (1 - t) + TAIL_COLOR[k] * t) for k in range(3))
            r = self.cell_rect(body[i], 2)
            pg.draw.rect(scr, col, r, border_radius=6)
            if i + 1 < n:  # bridge to the next segment so the body looks continuous
                a, b = body[i], body[i + 1]
                br = self.cell_rect(a, 2).union(self.cell_rect(b, 2))
                if a[0] == b[0]:
                    br.inflate_ip(0, -CELL + 4)
                else:
                    br.inflate_ip(-CELL + 4, 0)
                pg.draw.rect(scr, col, br)
        # head + eyes
        hx, hy = self.center(g.head)
        dx, dy = g.direction
        px, py = -dy, dx
        for s in (-1, 1):
            ex, ey = hx + dx * 3 + px * 4 * s, hy + dy * 3 + py * 4 * s
            pg.draw.circle(scr, (255, 255, 255), (ex, ey), 3)
            pg.draw.circle(scr, (0, 0, 0), (ex + dx, ey + dy), 2)

    def _draw_panel(self):
        pg, g, scr = self.pg, self.game, self.screen
        x0 = BOARD_W
        pg.draw.rect(scr, PANEL_BG, (x0, 0, PANEL_W, BOARD_H))
        pg.draw.line(scr, (50, 55, 70), (x0, 0), (x0, BOARD_H), 2)
        x = x0 + 18
        y = 16
        scr.blit(self.font_big.render("SNAKE AI", True, HEAD_COLOR), (x, y)); y += 40
        scr.blit(self.font.render(self.agent.name, True, TEXT), (x, y)); y += 28
        status = self.agent.status
        scr.blit(self.font.render(status, True, STATUS_COLORS.get(status, TEXT)), (x, y)); y += 38

        st = self.stats[self.mode]
        recent = st["recent"]
        avg = sum(recent) / len(recent) if recent else 0
        rows = [
            ("Score", g.score),
            ("Length", g.length),
            ("Steps", g.steps),
            ("Best", st["best"]),
            ("Avg (last 50)", f"{avg:.1f}"),
            ("Games", st["games"]),
            ("Speed", f"{self.speed} steps/s"),
        ]
        for label, val in rows:
            scr.blit(self.font_small.render(label, True, TEXT_DIM), (x, y))
            v = self.font.render(str(val), True, TEXT)
            scr.blit(v, (x0 + PANEL_W - 18 - v.get_width(), y - 3))
            y += 24

        # hunger bar
        y += 6
        scr.blit(self.font_small.render("Hunger", True, TEXT_DIM), (x, y)); y += 18
        frac = min(1.0, g.steps_since_food / g.starve_limit)
        bar_w = PANEL_W - 36
        pg.draw.rect(scr, (45, 50, 62), (x, y, bar_w, 10), border_radius=5)
        hc = (int(80 + 175 * frac), int(220 - 170 * frac), 80)
        pg.draw.rect(scr, hc, (x, y, max(4, int(bar_w * frac)), 10), border_radius=5)
        y += 24

        # hazards
        haz = f"{len(g.obstacles)} obst • {len(g.traps)} traps • {len(g.enemies)} enemies" if self.hazards else "hazards off"
        scr.blit(self.font_small.render(haz, True, TEXT_DIM), (x, y)); y += 22
        if self.deaths:
            top = ", ".join(f"{k} {v}" for k, v in self.deaths.most_common(3))
            scr.blit(self.font_small.render("Deaths: " + top, True, TEXT_DIM), (x, y))
        y += 26

        # legend
        for color, label in ((APPLE_COLOR, "apple +1"), (GOLD_COLOR, f"golden +{GOLDEN_POINTS} (expires)"),
                             (OBSTACLE_COLOR, "obstacle"), (TRAP_COLOR, "trap (fades)"),
                             (ENEMY_COLOR, "enemy (stalks you)")):
            pg.draw.rect(scr, color, (x, y + 2, 12, 12), border_radius=3)
            scr.blit(self.font_small.render(label, True, TEXT_DIM), (x + 20, y)); y += 18

        # controls
        controls = ["SPACE pause   +/- speed", "V path   R restart   M mode", "H hazards   ESC quit"]
        if self.mode == "human":
            controls.insert(0, "Arrows / WASD to steer")
        y = BOARD_H - 10 - 18 * len(controls)
        pg.draw.line(scr, (50, 55, 70), (x, y - 8), (x0 + PANEL_W - 18, y - 8), 1)
        for line in controls:
            scr.blit(self.font_small.render(line, True, TEXT_DIM), (x, y)); y += 18


# ===============================
# --- Entry point
# ===============================
def main():
    ap = argparse.ArgumentParser(description="Snake AI: Hunter planner + neuroevolution")
    ap.add_argument("--mode", choices=["hunter", "neural", "human", "train", "benchmark"], default="hunter")
    ap.add_argument("--agent", choices=["hunter", "neural"], default="hunter", help="agent for --mode benchmark")
    ap.add_argument("--games", type=int, default=32, help="games for --mode benchmark")
    ap.add_argument("--generations", type=int, default=None, help="generations for --mode train (default: until Ctrl+C)")
    ap.add_argument("--no-hazards", action="store_true", help="classic snake: no obstacles/traps/enemies/golden")
    ap.add_argument("--speed", type=int, default=None, help="initial steps per second")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    hazards = not args.no_hazards

    if args.mode == "train":
        train(args.generations, hazards, args.seed)
    elif args.mode == "benchmark":
        benchmark(args.agent, args.games, hazards, args.seed)
    else:
        App(args.mode, hazards, args.speed, args.seed).run()
        import pygame
        pygame.quit()


if __name__ == "__main__":
    main()
