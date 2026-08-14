"""Two anatomical sanity rules applied to the network's masks.

Optional, but the segmentations in the study were cleaned with these before
anything was measured from them. Both target things that are wrong on anatomy
alone, not things that happen to disagree with an expert; nothing here is
fitted to an outcome.

**Rule 1 — one object.** The LV is a single connected structure. Where a slice
contains more than one component of (myocardium + blood pool), only the one
nearest the LV long axis is kept.

**Rule 2 — in the right place.** The LV sits on a consistent axis through the
stack. If *every* component in a slice is further than ``max_dist_px`` from
that axis, the network has not found the ventricle at all and the slice is
cleared.

The axis is the mean cavity centroid over slices carrying at least 30 % of that
stack's largest blood pool — i.e. defined only from slices where the ventricle
is unambiguous, so a bad end slice cannot drag it.

Why 30 px: interior slices sit a median 1.5 px from the axis (p90 = 4.0), and
the genuine failures at 75-87 px. Anything from ~25 to ~50 px separates them
identically, so the threshold sits in a gap in the data rather than on a fitted
boundary.

Usage:

    from postprocess import clean_volume
    cleaned, n_cleared, n_deblobbed = clean_volume(vol)   # (x, y, slice, phase)
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

MAX_DIST_PX = 30.0      # see module docstring: 25-50 all behave identically
CAVITY_FRAC = 0.30      # slices defining the axis need this much of the max cavity


def lv_axis(vol: np.ndarray, phase: int) -> np.ndarray | None:
    """Mean blood-pool centroid over slices where the ventricle is unambiguous."""
    cav = np.array([(vol[:, :, s, phase] == 2).sum() for s in range(vol.shape[2])], float)
    if cav.max() <= 0:
        return None
    good = [s for s in range(vol.shape[2]) if cav[s] >= CAVITY_FRAC * cav.max()]
    if not good:
        return None
    return np.array([ndi.center_of_mass(vol[:, :, s, phase] == 2) for s in good]).mean(axis=0)


def clean_phase(labels: np.ndarray, center: np.ndarray | None,
                max_dist_px: float = MAX_DIST_PX) -> tuple[np.ndarray, int, int]:
    """Apply both rules to one ``(x, y, slice)`` label slab."""
    out = labels.copy()
    if center is None:
        return out, 0, 0
    n_cleared = n_deblobbed = 0
    for s in range(out.shape[2]):
        mask = out[:, :, s] > 0
        if not mask.any():
            continue
        lab, n = ndi.label(mask)
        cents = [ndi.center_of_mass(lab == i) for i in range(1, n + 1)]
        dist = [float(np.hypot(c[0] - center[0], c[1] - center[1])) for c in cents]
        k = int(np.argmin(dist))
        if dist[k] > max_dist_px:               # rule 2: nothing here is the LV
            out[:, :, s] = 0
            n_cleared += 1
        elif n > 1:                             # rule 1: keep the LV component only
            out[:, :, s] = np.where(lab == k + 1, out[:, :, s], 0)
            n_deblobbed += 1
    return out, n_cleared, n_deblobbed


def clean_volume(vol: np.ndarray,
                 max_dist_px: float = MAX_DIST_PX) -> tuple[np.ndarray, int, int]:
    """Clean every phase of a ``(x, y, slice, phase)`` label volume.

    The axis is recomputed per phase: the heart moves between ED and ES, and
    borrowing one phase's center for the other would introduce exactly the kind
    of drift the rule is meant to catch.
    """
    out = vol.copy()
    cleared = deblobbed = 0
    for p in range(vol.shape[3]):
        center = lv_axis(vol, p)
        out[:, :, :, p], c, d = clean_phase(vol[:, :, :, p], center, max_dist_px)
        cleared += c
        deblobbed += d
    return out, cleared, deblobbed
