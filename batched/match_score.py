"""Score 3D masks against spots by one-to-one matching instead of point-in-mask.

Spots and mask centroids are matched 1:1 (Hungarian on distance, max --r voxels). Leftovers are
classified so that gray areas are separated from likely errors:
  matched          spot <-> mask within r
  merge            unmatched spot that sits inside (or within r of) a mask already matched to
                   another spot: 2 spots / 1 mask (e.g. a dividing pair kept together) - gray area
  split            unmatched mask whose nearest spot (within 2r) is matched to another mask:
                   1 spot / 2 masks (e.g. tracker failed to separate, or a spot averaged between
                   a fallen-out nucleus and its neighbour) - often a spot error, not a mask error
  missed           unmatched spot with no mask nearby -> candidate real miss
  extra            unmatched mask with no spot within 2r -> fallen-out nucleus, pole cell, or
                   false positive (check by eye)
"""

import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree


def match_score(masks, P, r=6.0):
    labels = np.unique(masks)
    labels = labels[labels > 0]
    C = (
        np.array(ndi.center_of_mass(masks > 0, masks, labels))
        if len(labels)
        else np.zeros((0, 3))
    )
    n, m = len(P), len(C)
    sp_match = np.full(n, -1)
    mk_match = np.full(m, -1)
    if n and m:
        tree = cKDTree(C)
        pairs = tree.query_ball_point(P, r)
        rows, cols, d = [], [], []
        for i, js in enumerate(pairs):
            for j in js:
                rows.append(i)
                cols.append(j)
                d.append(np.linalg.norm(P[i] - C[j]))
        if rows:
            # solve on the connected sub-problem only (dense cost over spots/masks that have candidates)
            ui, ri = np.unique(rows, return_inverse=True)
            uj, cj = np.unique(cols, return_inverse=True)
            cost = np.full((len(ui), len(uj)), 1e6, np.float32)
            cost[ri, cj] = d
            a, b = linear_sum_assignment(cost)
            ok = cost[a, b] <= r
            sp_match[ui[a[ok]]] = uj[b[ok]]
            mk_match[uj[b[ok]]] = ui[a[ok]]
    p_int = np.clip(np.round(P).astype(int), 0, np.array(masks.shape) - 1)
    inside = masks[tuple(p_int.T)] if n else np.zeros(0, int)
    lab_index = {l: k for k, l in enumerate(labels)}
    cat_sp = np.array(["matched"] * n, dtype=object)
    for i in np.where(sp_match < 0)[0]:
        host = lab_index.get(inside[i], -1)
        if host < 0 and m:
            dd, host = cKDTree(C).query(P[i])
            host = host if dd <= r else -1
        cat_sp[i] = "merge" if host >= 0 and mk_match[host] >= 0 else "missed"
    cat_mk = np.array(["matched"] * m, dtype=object)
    if n:
        dd, nn = cKDTree(P).query(C) if m else (np.zeros(0), np.zeros(0, int))
        for j in np.where(mk_match < 0)[0]:
            cat_mk[j] = "split" if dd[j] <= 2 * r and sp_match[nn[j]] >= 0 else "extra"
    stats = {
        "n_spots": n,
        "n_masks": m,
        "matched": int((sp_match >= 0).sum()),
        "merge": int((cat_sp == "merge").sum()),
        "missed": int((cat_sp == "missed").sum()),
        "split": int((cat_mk == "split").sum()),
        "extra": int((cat_mk == "extra").sum()),
    }
    stats["recall_matched"] = stats["matched"] / n if n else np.nan
    stats["recall_incl_merge"] = (
        (stats["matched"] + stats["merge"]) / n if n else np.nan
    )
    return stats, cat_sp, cat_mk, C, labels
