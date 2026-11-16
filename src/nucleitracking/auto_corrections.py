"""
Automated post-tracking lineage corrections.

Detects the tracking errors that were most common in the manual corrections of
20250131 and repairs them, emitting the same ``Action`` objects the interactive
``correct-embryo`` tool records.  The result is saved as a regular
``corrections_<timestamp>.pkl`` so the Napari session loads it as the starting
point for manual review, and every automatic edit can be inspected or undone.

Error classes handled (in the order they are applied):

1. **Zipper shifts** — a row of nuclei is each linked to its neighbour's
   descendant, leaving a terminal at one end and an orphan at the other.  Links
   that break constant-velocity motion (scaled by the actual frame interval, so
   acquisition pauses are handled) and terminal/orphan pairs seed regions that
   are re-linked by a local assignment, accepted only if it lowers the motion
   cost without breaking links.
2. **Missed divisions** — at mitosis, the mother keeps one daughter and the
   second daughter starts as an orphan track.  Orphans are assigned to mothers
   so that the mother sits at the midpoint of the two daughters.
3. **Stolen daughters** — a one-frame spurious spot next to the mother in
   anaphase captures a real daughter.  Orphan chains may enter the lineage one
   node after their head, discarding the spurious head.
4. **Residual-spot continuations** — the mother is linked to a short-lived spot
   (< ``min_branch_len``) while both real daughters are orphans.  Such stub
   children are ignored when looking for mothers and unlinked when a real
   daughter is attached.
5. **Short gaps** — a terminal nucleus re-appears as an orphan a few frames
   later.  Linked with flow-compensated distance up to ``max_gap`` frames.

Finally, remaining short terminal branches are marked ``ISSUE`` and terminals
at the posterior pole ``POLE_TERMINAL`` (the same rules the GUI applies when
stepping through terminals), leaving only the ambiguous terminals for review.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import KDTree

POSITION_COLUMNS = ["x", "y", "z"]


@dataclass
class AutoCorrectParams:
    # shared
    min_branch_len: int = 4  # terminal branches shorter than this are spurious
    pole_ap: float = 0.95  # terminals beyond this AP are pole terminals
    flow_neighbors: int = 12  # neighbours used to estimate local flow
    # mitosis windows (fraction of nuclei dividing per frame, smoothed)
    mitosis_div_fraction: float = 0.004
    mitosis_dilation: int = 4
    # zipper repair
    zipper_residual: float = 2.5  # um, constant-velocity residual that flags a link
    zipper_radius: float = 8.0  # um, neighbourhood re-linked around flags
    zipper_no_link_cost: float = 4.0
    zipper_min_gain: float = 1.0  # um, required total cost reduction
    # division / gap rescue
    max_gap: int = 3  # frames between parent and orphan (longer gaps were
    #                   mostly spurious on 20250131, so they are left for review)
    max_division_gap: int = 2  # frames between mother and late daughter
    link_radius: float = 9.0  # um, search radius for candidate parents
    max_midpoint_error: float = 3.0  # um, mother vs. daughter midpoint
    min_sibling_dist: float = 2.0  # um
    max_sibling_dist: float = 12.0  # um
    max_continuation_error: float = 4.0  # um, after flow compensation
    min_mother_age: int = 10  # frames since the mother's lineage last divided
    min_parent_history: int = 3  # parents must have >= this many ancestors
    min_orphan_len: int = 4  # shorter orphan chains are spurious, not rescued
    gap_penalty: float = 0.1  # um per extra frame of gap
    entry_penalty: float = 0.5  # um for skipping a spurious orphan head
    no_link_cost: float = 4.0
    n_rounds: int = 4


@dataclass
class LogEntry:
    step: str
    kind: str
    parent: int
    child: int
    frame: int
    cost: float = np.nan
    note: str = ""


@dataclass
class LineageGraph:
    """
    Mutable parent/child graph over a prepared spots DataFrame.

    Rows are positional indices into ``df``; ``ids`` maps them back to the
    DataFrame index, which is what correction ``Action`` objects refer to.
    """

    df: pd.DataFrame
    ids: np.ndarray = field(init=False)
    frame: np.ndarray = field(init=False)
    pos: np.ndarray = field(init=False)
    ap: np.ndarray = field(init=False)
    parent: np.ndarray = field(init=False)
    children: list[list[int]] = field(init=False)

    def __post_init__(self):
        df = self.df
        self.ids = df.index.to_numpy()
        self.row_of = {nid: r for r, nid in enumerate(self.ids)}
        self.frame = df["frame"].to_numpy().astype(int)
        self.pos = df[POSITION_COLUMNS].to_numpy().astype(float)
        self.ap = df["AP"].to_numpy().astype(float)

        parent_ids = df["parent_id"].to_numpy()
        self.parent = np.array(
            [self.row_of.get(p, -1) if p != -1 else -1 for p in parent_ids], dtype=int
        )
        self.children = [[] for _ in range(len(df))]
        for r, p in enumerate(self.parent):
            if p != -1:
                self.children[p].append(r)

        self.first_frame = int(self.frame.min())
        self.last_frame = int(self.frame.max())
        order = np.argsort(self.frame, kind="stable")
        bounds = np.searchsorted(
            self.frame[order], np.arange(self.first_frame, self.last_frame + 2)
        )
        self.rows_at = {
            f: order[bounds[i] : bounds[i + 1]]
            for i, f in enumerate(range(self.first_frame, self.last_frame + 1))
        }
        self.trees = {f: KDTree(self.pos[rows]) for f, rows in self.rows_at.items()}

        # interval from frame f to f+1, relative to the median interval; used to
        # scale velocities across acquisition pauses
        frames = np.arange(self.first_frame, self.last_frame + 1)
        if "time" in df.columns:
            t = df.groupby("frame")["time"].first().reindex(frames).to_numpy()
            dt = np.diff(t)
            dt = dt / np.nanmedian(dt)
            dt = np.where(np.isfinite(dt) & (dt > 0), dt, 1.0)
        else:
            dt = np.ones(len(frames) - 1)
        self.dt = dict(zip(frames[:-1].tolist(), dt.tolist(), strict=True))

    # -------------------------------------------------------------- editing
    def link(self, p: int, c: int):
        if self.parent[c] != -1:
            self.unlink(c)
        self.parent[c] = p
        self.children[p].append(c)

    def unlink(self, c: int):
        p = self.parent[c]
        if p == -1:
            return
        self.children[p].remove(c)
        self.parent[c] = -1

    # -------------------------------------------------------------- queries
    def near(self, f: int, point: np.ndarray, radius: float) -> np.ndarray:
        if f not in self.trees:
            return np.empty(0, dtype=int)
        hits = self.trees[f].query_ball_point(point, radius)
        return self.rows_at[f][hits]

    def chain_down(self, r: int, max_len: int = 10_000) -> tuple[int, int, bool]:
        """Follow single children from r. Returns (n_nodes, last_row, ends_at_terminal)."""
        n = 1
        while n < max_len:
            kids = self.children[r]
            if len(kids) != 1:
                return n, r, len(kids) == 0
            r = kids[0]
            n += 1
        return n, r, False

    def is_stub(self, r: int, min_len: int) -> bool:
        """A short-lived spot: its chain dies (not divides) within min_len nodes."""
        n, last, terminal = self.chain_down(r, max_len=min_len)
        return terminal and n < min_len and self.frame[last] < self.last_frame

    def real_children(self, r: int, min_len: int) -> tuple[list[int], list[int]]:
        real, stubs = [], []
        for c in self.children[r]:
            (stubs if self.is_stub(c, min_len) else real).append(c)
        return real, stubs

    def history(self, r: int, cap: int) -> tuple[int, int]:
        """
        Walk up at most ``cap`` ancestors from r. Returns (n_ancestors,
        frames_since_division), where frames_since_division is measured to the
        last ancestor with two children, or to the furthest ancestor visited.
        """
        n = 0
        cur = r
        while self.parent[cur] != -1 and n < cap:
            p = self.parent[cur]
            n += 1
            if len(self.children[p]) >= 2:
                return n, int(self.frame[r] - self.frame[p])
            cur = p
        return n, int(self.frame[r] - self.frame[cur])

    def divides_up(self, r: int, lo: int, hi: int) -> bool:
        """Does any ancestor of r (excluding r) divide within frames [lo, hi]?"""
        a = self.parent[r]
        while a != -1 and self.frame[a] >= lo:
            if len(self.children[a]) >= 2 and self.frame[a] <= hi:
                return True
            a = self.parent[a]
        return False

    def divides_down(self, r: int, lo: int, hi: int) -> bool:
        """Does the chain starting at r (including r) divide within [lo, hi]?"""
        while self.frame[r] <= hi:
            kids = self.children[r]
            if len(kids) >= 2:
                return self.frame[r] >= lo
            if not kids:
                return False
            r = kids[0]
        return False

    def branch_len(self, r: int) -> int:
        """Nodes from r back to (excluding) the last dividing ancestor or root."""
        n = 1
        p = self.parent[r]
        while p != -1 and len(self.children[p]) < 2:
            n += 1
            p = self.parent[p]
        return n


class Flow:
    """
    Local nucleus flow between consecutive frames, estimated from current
    non-dividing single-frame links as the median displacement of neighbours.
    """

    def __init__(self, g: LineageGraph, k: int):
        self.k = k
        self.g = g
        self.by_frame: dict[int, tuple[KDTree, np.ndarray]] = {}
        self.rebuild()

    def rebuild(self):
        g = self.g
        child = np.arange(len(g.parent))
        par = g.parent
        has = par != -1
        child, par = child[has], par[has]
        single = np.array([len(g.children[p]) == 1 for p in par], dtype=bool)
        step = g.frame[child] - g.frame[par] == 1
        child, par = child[single & step], par[single & step]
        disp = g.pos[child] - g.pos[par]
        self.by_frame = {}
        for f in np.unique(g.frame[par]):
            m = g.frame[par] == f
            if m.sum() >= self.k:
                self.by_frame[int(f)] = (KDTree(g.pos[par[m]]), disp[m])

    def at(self, f: int, point: np.ndarray, exclude_self: bool = False) -> np.ndarray:
        """Median displacement from frame f to f+1 around point."""
        if f not in self.by_frame:
            return np.zeros(3)
        tree, disp = self.by_frame[f]
        k = self.k + 1 if exclude_self else self.k
        _, nb = tree.query(point, k=min(k, len(disp)))
        nb = np.atleast_1d(nb)
        if exclude_self:
            nb = nb[1:]
        return np.median(disp[nb], axis=0)

    def propagate(self, f: int, point: np.ndarray, n_frames: int) -> np.ndarray:
        p = point.copy()
        for i in range(n_frames):
            p = p + self.at(f + i, p)
        return p


def mitosis_frames(g: LineageGraph, params: AutoCorrectParams) -> set[int]:
    frames = np.arange(g.first_frame, g.last_frame + 1)
    n_div = np.array(
        [sum(len(g.children[r]) >= 2 for r in g.rows_at[f]) for f in frames]
    )
    n_tot = np.array([len(g.rows_at[f]) for f in frames])
    frac = pd.Series(n_div / n_tot).rolling(5, center=True, min_periods=1).mean()
    hot = frac.to_numpy() > params.mitosis_div_fraction
    d = params.mitosis_dilation
    out = set()
    for i in np.where(hot)[0]:
        out.update(int(f) for f in frames[max(0, i - d) : i + d + 1])
    return out


class MitosisWindows:
    """
    Contiguous runs of mitosis frames.  Every frame belongs to the cycle span of
    its nearest window, and a lineage may divide at most once per span; this is
    what keeps a nucleus from dividing twice in one cycle.  Spans rather than
    windows, because tracking can place a division in the tail of the wave, just
    outside the detected window.
    """

    def __init__(self, frames: set[int]):
        self.bounds: list[tuple[int, int]] = []
        for f in sorted(frames):
            if self.bounds and f == self.bounds[-1][1] + 1:
                self.bounds[-1] = (self.bounds[-1][0], f)
            else:
                self.bounds.append((f, f))
        self.window_of = {
            f: k for k, (lo, hi) in enumerate(self.bounds) for f in range(lo, hi + 1)
        }

    def __contains__(self, frame) -> bool:
        return int(frame) in self.window_of

    def get(self, frame) -> tuple[int, int] | None:
        k = self.window_of.get(int(frame))
        return None if k is None else self.bounds[k]

    def span(self, frame) -> tuple[int, int]:
        """Frames closer to this frame's mitosis window than to any other."""
        frame = int(frame)
        edges = [
            (hi + next_lo) // 2
            for (_, hi), (next_lo, _) in zip(self.bounds, self.bounds[1:], strict=False)
        ]
        k = int(np.searchsorted(edges, frame, side="left"))
        lo = edges[k - 1] + 1 if k > 0 else -(10**9)
        hi = edges[k] if k < len(edges) else 10**9
        return lo, hi


# ---------------------------------------------------------------------------
# Step 1 — zipper shifts
# ---------------------------------------------------------------------------
def _velocity_in(g: LineageGraph, flow: Flow, r: int) -> np.ndarray:
    """Expected displacement of r over the next frame, from its own past motion."""
    f = g.frame[r]
    p = g.parent[r]
    if p != -1 and f - g.frame[p] == 1:
        v = g.pos[r] - g.pos[p]
    else:
        v = flow.at(f - 1, g.pos[r])
    return v * g.dt.get(f, 1.0) / g.dt.get(f - 1, 1.0)


def _velocity_out(g: LineageGraph, flow: Flow, r: int) -> np.ndarray:
    """Expected displacement into r over the previous frame, from its future motion."""
    f = g.frame[r]
    kids = g.children[r]
    if len(kids) == 1 and g.frame[kids[0]] - f == 1:
        v = g.pos[kids[0]] - g.pos[r]
    else:
        v = flow.at(f, g.pos[r])
    return v * g.dt.get(f - 1, 1.0) / g.dt.get(f, 1.0)


def _link_costs(
    g: LineageGraph, flow: Flow, parents: list[int], kids: list[int]
) -> np.ndarray:
    """
    Constant-velocity cost of linking each parent to each kid: how far the
    link's displacement is from the parent's incoming and the kid's outgoing
    velocity.  A row shifted by one neighbour breaks this badly, while sisters
    separating after mitosis (where neighbour flow is meaningless) do not.
    """
    disp = g.pos[kids][None, :, :] - g.pos[parents][:, None, :]
    v_in = np.array([_velocity_in(g, flow, p) for p in parents])[:, None, :]
    v_out = np.array([_velocity_out(g, flow, c) for c in kids])[None, :, :]
    return 0.5 * (
        np.linalg.norm(disp - v_in, axis=-1) + np.linalg.norm(disp - v_out, axis=-1)
    )


def fix_zippers(
    g: LineageGraph, flow: Flow, mitosis: MitosisWindows, params: AutoCorrectParams
) -> list[LogEntry]:
    log = []
    for f in range(g.first_frame + 1, g.last_frame):
        links = [
            (p, g.children[p][0])
            for p in g.rows_at[f]
            if len(g.children[p]) == 1 and g.frame[g.children[p][0]] == f + 1
        ]
        if not links:
            continue
        ps, cs = map(list, zip(*links, strict=True))
        disp = g.pos[cs] - g.pos[ps]
        v_in = np.array([_velocity_in(g, flow, p) for p in ps])
        v_out = np.array([_velocity_out(g, flow, c) for c in cs])
        res = 0.5 * (
            np.linalg.norm(disp - v_in, axis=1) + np.linalg.norm(disp - v_out, axis=1)
        )
        flagged = g.pos[np.array(ps)[res > params.zipper_residual]]
        # every shifted row ends in a terminal at f and an orphan at f+1
        terminals = [r for r in g.rows_at[f] if not g.children[r]]
        orphans = [r for r in g.rows_at[f + 1] if g.parent[r] == -1]
        seeds = [flagged]
        if terminals and orphans:
            seeds += [g.pos[terminals], g.pos[orphans]]
        seeds = np.concatenate(seeds)
        if len(seeds) == 0:
            continue

        # group seeds into spatial clusters and re-link each region
        tree = KDTree(seeds)
        seen = set()
        for i in range(len(seeds)):
            if i in seen:
                continue
            cluster, frontier = {i}, [i]
            while frontier:
                j = frontier.pop()
                for k in tree.query_ball_point(seeds[j], 2 * params.zipper_radius):
                    if k not in cluster:
                        cluster.add(k)
                        frontier.append(k)
            seen |= cluster
            log += _relink_region(g, flow, mitosis, f, seeds[sorted(cluster)], params)
    return log


def _relink_region(
    g: LineageGraph,
    flow: Flow,
    mitosis: MitosisWindows,
    f: int,
    seeds: np.ndarray,
    params: AutoCorrectParams,
) -> list[LogEntry]:
    parents = set()
    for s in seeds:
        parents.update(g.near(f, s, params.zipper_radius).tolist())
    # only non-dividing parents take part
    parents = [p for p in parents if len(g.children[p]) <= 1]
    kids = set()
    for p in parents:
        kids.update(c for c in g.children[p] if g.frame[c] == f + 1)
        for c in g.near(f + 1, g.pos[p], params.zipper_radius):
            if g.parent[c] == -1 or g.parent[c] in parents:
                kids.add(int(c))
    kids = [c for c in kids if g.parent[c] == -1 or g.parent[c] in parents]
    if not parents or not kids:
        return []

    cost = _link_costs(g, flow, parents, kids)
    nl = params.zipper_no_link_cost
    n_p, n_c = cost.shape
    big = np.full((n_p + n_c, n_c + n_p), 1e6)
    big[:n_p, :n_c] = np.where(cost < nl, cost, 1e6)
    big[:n_p, n_c:] = np.where(np.eye(n_p), nl, 1e6)
    big[n_p:, :n_c] = np.where(np.eye(n_c), nl, 1e6)
    big[n_p:, n_c:] = 0  # dummy-dummy
    ri, ci = linear_sum_assignment(big)

    new_parent = dict.fromkeys(kids, -1)
    new_cost = 0.0
    for r, c in zip(ri, ci, strict=True):
        if r < n_p and c < n_c:
            new_parent[kids[c]] = parents[r]
            new_cost += cost[r, c]
        elif r < n_p or c < n_c:
            new_cost += nl
    p_index = {p: i for i, p in enumerate(parents)}
    old_cost = 0.0
    for j, c in enumerate(kids):
        p = g.parent[c]
        old_cost += min(cost[p_index[p], j], 2 * nl) if p != -1 else nl
    old_cost += nl * sum(1 for p in parents if not any(g.parent[c] == p for c in kids))
    if old_cost - new_cost < params.zipper_min_gain:
        return []
    # A genuine shift is repaired by re-pairing, never by breaking links.
    n_old = sum(g.parent[c] != -1 for c in kids)
    n_new = sum(p != -1 for p in new_parent.values())
    if n_new < n_old:
        return []
    # Re-pairing moves whole subtrees between lineages: reject the region if that
    # would give a lineage two divisions in one cycle.
    span = mitosis.span(f)
    for c, p in new_parent.items():
        if (
            p not in (-1, g.parent[c])
            and g.divides_up(p, *span)
            and g.divides_down(c, *span)
        ):
            return []

    log = []
    for c in kids:
        if new_parent[c] != g.parent[c]:
            old = g.parent[c]
            if old != -1:
                g.unlink(c)
                log.append(LogEntry("zipper", "unlink", int(old), int(c), f + 1))
    for j, c in enumerate(kids):
        p = new_parent[c]
        if p != -1 and g.parent[c] != p:
            g.link(p, c)
            log.append(
                LogEntry(
                    "zipper", "link", int(p), int(c), f + 1, float(cost[p_index[p], j])
                )
            )
    return log


# ---------------------------------------------------------------------------
# Steps 2-5 — orphan rescue (divisions, stolen daughters, stubs, gaps)
# ---------------------------------------------------------------------------
def _candidate_cost(
    g: LineageGraph,
    flow: Flow,
    mitosis: MitosisWindows,
    p: int,
    e: int,
    params: AutoCorrectParams,
) -> tuple[float, str] | None:
    gap = g.frame[e] - g.frame[p]
    real, _stubs = g.real_children(p, params.min_branch_len)
    if len(real) >= 2:
        return None
    n_anc, since_div = g.history(
        p, cap=max(params.min_parent_history, params.min_mother_age)
    )
    if n_anc < params.min_parent_history:
        return None
    pred = flow.propagate(g.frame[p], g.pos[p], gap)

    if len(real) == 0:
        err = float(np.linalg.norm(g.pos[e] - pred))
        if err > params.max_continuation_error:
            return None
        # joining p's lineage to e's chain must not put two divisions in one
        # cycle (p divided early in the span and e divides later)
        for span in {mitosis.span(g.frame[p]), mitosis.span(g.frame[e])}:
            if g.divides_up(p, *span) and g.divides_down(e, *span):
                return None
        return err + params.gap_penalty * (gap - 1), "continuation"

    # division hypothesis: p keeps `real[0]`, e becomes the second daughter
    if gap > params.max_division_gap or g.frame[p] not in mitosis:
        return None
    if since_div < params.min_mother_age:
        return None
    sib = real[0]
    # one division per lineage per cycle: the mother's lineage must not already
    # have divided in this span, and neither daughter chain may divide in it
    # later (that division would be the real one)
    span = mitosis.span(g.frame[p])
    if g.divides_up(p, *span) or g.divides_down(sib, *span) or g.divides_down(e, *span):
        return None
    # position of the sibling lineage at the orphan's frame
    r = sib
    while g.frame[r] < g.frame[e] and len(g.children[r]) == 1:
        r = g.children[r][0]
    if g.frame[r] == g.frame[e]:
        sib_pos = g.pos[r]
    else:
        sib_pos = flow.propagate(g.frame[r], g.pos[r], max(0, g.frame[e] - g.frame[r]))
    sib_dist = float(np.linalg.norm(sib_pos - g.pos[e]))
    if not (params.min_sibling_dist <= sib_dist <= params.max_sibling_dist):
        return None
    err = float(np.linalg.norm((sib_pos + g.pos[e]) / 2 - pred))
    if err > params.max_midpoint_error:
        return None
    return err + params.gap_penalty * (gap - 1), "division"


def rescue_orphans(
    g: LineageGraph, flow: Flow, mitosis: MitosisWindows, params: AutoCorrectParams
) -> list[LogEntry]:
    log = []
    for round_ in range(params.n_rounds):
        heads = np.where((g.parent == -1) & (g.frame > g.first_frame))[0]

        # (head, parent) -> best (cost, entry, skip, kind)
        best: dict[tuple[int, int], tuple[float, int, int, str]] = {}
        for h in heads:
            entries = [(h, 0)]
            if len(g.children[h]) == 1:
                entries.append((g.children[h][0], 1))
            for e, skip in entries:
                if g.chain_down(e, params.min_orphan_len)[0] < params.min_orphan_len:
                    continue
                for gap in range(1, params.max_gap + 1):
                    fp = g.frame[e] - gap
                    if fp < g.first_frame:
                        break
                    for p in g.near(fp, g.pos[e], params.link_radius):
                        if p == h:
                            continue
                        res = _candidate_cost(g, flow, mitosis, p, e, params)
                        if res is None:
                            continue
                        cost, kind = res
                        cost += params.entry_penalty * skip
                        key = (int(h), int(p))
                        if key not in best or cost < best[key][0]:
                            best[key] = (cost, int(e), skip, kind)

        if not best:
            break

        h_list = sorted({h for h, _ in best})
        p_list = sorted({p for _, p in best})
        h_ix = {h: i for i, h in enumerate(h_list)}
        p_ix = {p: i for i, p in enumerate(p_list)}
        n_h, n_p = len(h_list), len(p_list)
        cost = np.full((n_h, n_p + n_h), 1e6)
        for (h, p), (c, *_rest) in best.items():
            cost[h_ix[h], p_ix[p]] = c
        cost[np.arange(n_h), n_p + np.arange(n_h)] = params.no_link_cost
        ri, ci = linear_sum_assignment(cost)

        # Candidates were scored before any of this round's links existed, so
        # re-check each against the links already applied (cheapest first):
        # two assignments in one round can otherwise stack two divisions on
        # one lineage.  Rejected ones are reconsidered next round.
        chosen = sorted(
            (cost[i, j], h_list[i], p_list[j])
            for i, j in zip(ri, ci, strict=True)
            if j < n_p and cost[i, j] < 1e6
        )
        n_new = 0
        for _c, h, p in chosen:
            c, e, skip, kind = best[(h, p)]
            res = _candidate_cost(g, flow, mitosis, p, e, params)
            if res is None or res[1] != kind:
                continue
            _, stubs = g.real_children(p, params.min_branch_len)
            for s in stubs:
                g.unlink(s)
                log.append(
                    LogEntry(
                        "rescue",
                        "unlink",
                        int(p),
                        int(s),
                        int(g.frame[s]),
                        note="stub child of mother",
                    )
                )
            if skip:
                g.unlink(e)
                log.append(
                    LogEntry(
                        "rescue",
                        "unlink",
                        int(h),
                        int(e),
                        int(g.frame[e]),
                        note="spurious orphan head",
                    )
                )
            g.link(p, e)
            log.append(
                LogEntry(
                    "rescue",
                    "link",
                    int(p),
                    int(e),
                    int(g.frame[e]),
                    c,
                    note=f"{kind} (round {round_ + 1}, gap {g.frame[e] - g.frame[p]})",
                )
            )
            n_new += 1
        if n_new == 0:
            break
    return log


# ---------------------------------------------------------------------------
# Step 6 — mark remaining terminals
# ---------------------------------------------------------------------------
def mark_terminals(g: LineageGraph, params: AutoCorrectParams) -> list[LogEntry]:
    log = []
    terminals = np.where(
        np.array([len(k) == 0 for k in g.children]) & (g.frame < g.last_frame)
    )[0]
    for t in terminals:
        if g.branch_len(t) < params.min_branch_len:
            log.append(LogEntry("mark", "issue", -1, int(t), int(g.frame[t])))
        elif g.ap[t] > params.pole_ap:
            log.append(LogEntry("mark", "pole", -1, int(t), int(g.frame[t])))
    return log


# ---------------------------------------------------------------------------
# Validation — one division per lineage per mitosis window
# ---------------------------------------------------------------------------
def check_cycles(g: LineageGraph, mitosis: MitosisWindows) -> list[LogEntry]:
    """
    Report (not fix) divisions that violate the cell cycle: a division outside
    any mitosis window, or two divisions on one lineage within one cycle span.
    These can come from the tracking itself; they are logged for review.
    """
    log = []
    for d in np.where(np.array([len(k) >= 2 for k in g.children]))[0]:
        if g.frame[d] not in mitosis:
            log.append(
                LogEntry(
                    "check", "division_outside_mitosis", -1, int(d), int(g.frame[d])
                )
            )
        a = g.parent[d]
        while a != -1 and len(g.children[a]) < 2:
            a = g.parent[a]
        span = mitosis.span(g.frame[d])
        if a != -1 and span[0] <= g.frame[a] <= span[1]:
            log.append(
                LogEntry(
                    "check",
                    "double_division",
                    int(a),
                    int(d),
                    int(g.frame[d]),
                    note=f"also divided at frame {g.frame[a]}, same cycle",
                )
            )
    return log


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def auto_correct(
    df: pd.DataFrame, params: AutoCorrectParams | None = None, verbose: bool = True
) -> tuple[list, pd.DataFrame]:
    """
    Run all automatic corrections on a prepared spots DataFrame (as returned by
    ``corrections.prepare_spots``, optionally with earlier actions replayed).

    Returns (actions, log) where ``actions`` is a list of correction ``Action``
    objects ready to be pickled for ``correct-embryo`` and ``log`` describes
    every edit in terms of DataFrame ids.
    """
    from .corrections import (
        LinkNucleiAction,
        MarkIssueAction,
        MarkNucleusAction,
        Status,
        UnlinkFromParentAction,
    )

    params = params or AutoCorrectParams()
    g = LineageGraph(df)
    mitosis = MitosisWindows(mitosis_frames(g, params))
    flow = Flow(g, params.flow_neighbors)
    if verbose:
        print(f"  mitosis frames: {_ranges(sorted(mitosis.window_of))}")
    n_before = len(check_cycles(g, mitosis))

    log = fix_zippers(g, flow, mitosis, params)
    if verbose:
        print(f"  zipper repair: {sum(e.kind == 'link' for e in log)} links changed")
    flow.rebuild()

    n = len(log)
    log += rescue_orphans(g, flow, mitosis, params)
    if verbose:
        kinds = pd.Series([e.note.split(" ")[0] for e in log[n:] if e.kind == "link"])
        print(f"  orphan rescue: {kinds.value_counts().to_dict()}")

    n = len(log)
    log += mark_terminals(g, params)
    if verbose:
        kinds = pd.Series([e.kind for e in log[n:]])
        print(f"  terminal marks: {kinds.value_counts().to_dict()}")

    checks = check_cycles(g, mitosis)
    log += checks
    if verbose:
        print(
            f"  cell-cycle check: {len(checks)} violations logged for review "
            f"({n_before} before auto-correction)"
        )

    actions = []
    for e in log:
        child = g.ids[e.child]
        if e.kind == "unlink":
            a = UnlinkFromParentAction(child)
        elif e.kind == "link":
            a = LinkNucleiAction(g.ids[e.parent], child)
        elif e.kind == "issue":
            a = MarkIssueAction(child)
        elif e.kind == "pole":
            a = MarkNucleusAction(child, Status.POLE_TERMINAL)
        else:
            continue
        a.auto_reason = f"{e.step}:{e.kind} {e.note}".strip()
        actions.append(a)

    log_df = pd.DataFrame([e.__dict__ for e in log])
    if not log_df.empty:
        log_df["parent"] = [g.ids[p] if p != -1 else -1 for p in log_df["parent"]]
        log_df["child"] = g.ids[log_df["child"].to_numpy()]
        log_df["AP"] = df.loc[log_df["child"], "AP"].to_numpy()
        for col in ("x", "y", "z"):
            log_df[col] = df.loc[log_df["child"], col].to_numpy()

    return actions, log_df


def _ranges(frames: list[int]) -> str:
    if not frames:
        return "none"
    out, start, prev = [], frames[0], frames[0]
    for f in frames[1:]:
        if f != prev + 1:
            out.append(f"{start}-{prev}")
            start = f
        prev = f
    out.append(f"{start}-{prev}")
    return ", ".join(out)
