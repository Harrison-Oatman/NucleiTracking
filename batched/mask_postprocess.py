"""Per-label clean-up of 3D instance masks.

For each label: keep its largest 6-connected component (other fragments -> background),
fill holes (3D cavities, and holes closed within any z/y/x plane), then drop labels below
min_vol voxels. Filled voxels only take background, never another label's voxels.

    from mask_postprocess import postprocess, mask_stats
    clean, info = postprocess(masks, min_vol=30)
"""

import numpy as np
from scipy import ndimage as ndi


def _per_label(masks):
    labels = np.unique(masks)
    labels = labels[labels > 0]
    objs = ndi.find_objects(masks)
    for lab in labels:
        sl = objs[lab - 1]
        if sl is not None:
            yield lab, sl


def mask_stats(masks):
    """Per label: volume, number of 6-connected components, voxels added by hole filling."""
    rows = []
    for lab, sl in _per_label(masks):
        m = masks[sl] == lab
        _, ncomp = ndi.label(m)
        filled = ndi.binary_fill_holes(m)
        rows.append((lab, int(m.sum()), ncomp, int(filled.sum() - m.sum())))
    return np.array(rows, dtype=np.int64).reshape(-1, 4)  # label, vol, ncomp, holes


def trim_streaks(masks, vol, max_elong=3.0, axis=2, max_angle_deg=30.0, frac=0.5):
    """Trim masks stretched along the illumination streak axis (default x): for labels whose
    long/short principal-axis ratio > max_elong and whose long axis is within max_angle_deg of
    `axis`, keep only voxels above frac x (peak - mask min) + mask min, connected to the peak.
    Returns (masks, n_trimmed). vol must be the raw volume (not normalized)."""
    out = masks.copy()
    n = 0
    cos_max = np.cos(np.deg2rad(max_angle_deg))
    for lab, sl in _per_label(masks):
        m = out[sl] == lab
        pts = np.argwhere(m)
        if len(pts) < 30:
            continue
        ev, evec = np.linalg.eigh(np.cov(pts.T))
        if (
            np.sqrt(ev[2] / max(ev[0], 1e-6)) <= max_elong
            or abs(evec[axis, 2]) < cos_max
        ):
            continue
        v = ndi.gaussian_filter(vol[sl].astype(np.float32), 1.0)
        vals = v[m]
        thr = vals.min() + frac * (vals.max() - vals.min())
        keep = m & (v >= thr)
        cc, _ = ndi.label(keep)
        peak = np.unravel_index(np.argmax(np.where(m, v, -np.inf)), m.shape)
        keep = cc == cc[peak] if cc[peak] else keep
        sub = out[sl]
        sub[m & ~keep] = 0
        n += 1
    return out, n


def postprocess(masks, min_vol=30):
    out = masks.copy()
    n_frag = n_filled = n_drop = 0
    for lab, sl in _per_label(masks):
        sub = out[sl]
        m = sub == lab
        cc, ncomp = ndi.label(m)
        if ncomp > 1:
            keep = np.argmax(np.bincount(cc.ravel())[1:]) + 1
            sub[m & (cc != keep)] = 0
            m = cc == keep
            n_frag += 1
        filled = ndi.binary_fill_holes(m)
        for ax in range(
            3
        ):  # also holes that are only closed within a plane (2D rings, cups)
            f2 = np.stack([ndi.binary_fill_holes(p) for p in np.moveaxis(m, ax, 0)])
            filled |= np.moveaxis(f2, 0, ax)
        add = filled & ~m & (sub == 0)
        if add.any():
            sub[add] = lab
            n_filled += 1
            m = m | add
        if m.sum() < min_vol:
            sub[m] = 0
            n_drop += 1
    return out, {
        "labels_fragmented": n_frag,
        "labels_filled": n_filled,
        "labels_dropped": n_drop,
    }
