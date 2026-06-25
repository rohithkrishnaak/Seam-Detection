#!/usr/bin/env python3
"""
seam_detector.py
================

Detects the seam (gap/joint) between two pieces of metal by finding the point
where a straight laser line, projected across both pieces, visibly "bends"
(jumps vertically) due to a misalignment or step between the two surfaces.

This is an "industrial-grade" sub-pixel pipeline built on two classical
machine-vision techniques:

  1. STEGER'S ALGORITHM for sub-pixel line/ridge extraction. Instead of
     thresholding the laser into a binary blob (which only gives pixel-level,
     i.e. +/-1px, accuracy) we model the laser stripe as a bright RIDGE in the
     image intensity surface and locate its centerline to sub-pixel accuracy
     using the local 2nd-order (Hessian) structure of the image.

  2. TRIANGULAR FUZZY NUMBERS (TFN) for robustly pairing the two candidate
     line segments (left piece / right piece) that belong to the same
     physical laser line, out of many candidate segments in the scene
     (background laser dots/lines, reflections, wires, etc.). Instead of
     hard if/else pixel-distance cutoffs, each candidate pair gets a smooth
     fuzzy-membership score in [0, 1] for "is this gap seam-like?" and "is
     this vertical alignment seam-like?", which are combined into one
     confidence score. The best-scoring pair wins.

HOW IT WORKS (pipeline)
------------------------
 1. (Optional) Ask the user for a Region Of Interest (ROI) to restrict the
    search -- this is the single biggest accuracy/robustness lever, since it
    removes background laser segments, reflections, and skin tone from
    consideration before any math is done.
 2. STEGER SUB-PIXEL RIDGE EXTRACTION (`_steger_ridge_points`):
      a. Build a "redness" response channel emphasising the red laser
         against the (mostly grey/skin) background: R - 0.5G - 0.5B.
      b. Convolve with separable Gaussian-derivative kernels (order 0, 1, 2)
         at scale sigma to obtain Ix, Iy, Ixx, Iyy, Ixy in one consistent,
         properly-scaled differentiation -- this combines Gaussian smoothing
         and differentiation into a single, numerically well-behaved step
         (the standard "scale-space derivative" trick), which is far more
         stable than smoothing then running a raw Sobel kernel.
      c. At every pixel, build the local 2x2 Hessian matrix:
                H = [ Ixx  Ixy ]
                    [ Ixy  Iyy ]
         and solve for its eigenvalues/eigenvectors IN CLOSED FORM (2x2
         symmetric matrices have an analytic eigen-solution, so this is
         fully vectorised across the whole image with NumPy -- no per-pixel
         Python loop, no scipy needed).
         The eigenvector for the most-negative eigenvalue points in the
         direction PERPENDICULAR to the ridge (across the laser stripe);
         this is the direction the ridge is "thinnest" in.
      d. Sub-pixel localisation: along that perpendicular (normal) direction,
         Steger's method finds where the FIRST directional derivative of the
         intensity surface crosses zero (the true peak of the ridge profile)
         using one Newton step:
                t = -(grad . n) / (n^T H n)
         where n is the unit normal eigenvector. t is the sub-pixel offset,
         in pixels, from the integer pixel center to the true ridge peak
         along the normal direction. A point is accepted only if this
         Newton step stays inside the pixel (i.e. the quadratic Taylor model
         was trustworthy) AND the ridge curvature there is strong enough to
         be a real stripe and not sensor/JPEG noise.
 3. GROUPING (`_group_into_segments`): the accepted sub-pixel points are
    grouped into spatially contiguous runs (8-connectivity on their rounded
    pixel coordinates), giving one or more raw ridge "polylines". Because the
    seam sometimes does NOT create a visible pixel gap in the ridge (the
    laser can still register continuously across a small step), each raw
    polyline is additionally tested for an internal KINK: a piecewise
    two-line least-squares fit is swept across the polyline and compared to
    a single-line fit. A large reduction in fit error indicates a genuine
    bend, and the polyline is split there into a left and right segment.
 4. TFN PAIRING (`_best_segment_pair`): every pair of resulting segments is
    scored with a fuzzy-logic confidence in [0, 1]:
       - mu_gap(gap)     : triangular membership favouring a small horizontal
                            gap (segments that are adjacent, as the two
                            halves of a once-continuous laser line would be),
                            gracefully degrading for larger or negative
                            (overlapping) gaps instead of a hard cutoff.
       - mu_ydiff(ydiff) : triangular membership favouring near-zero vertical
                            misalignment AT THE SEAM BOUNDARY (evaluated from
                            each segment's own local line fit, not a crude
                            whole-segment mean -- this stays correct even for
                            long, sloped, or curved laser segments).
       - confidence(n)   : a length-based reliability weight in [0, 1] so
                            short, noisy fragments (a handful of stray
                            sub-pixel points) can't out-score a long,
                            well-supported ridge segment merely by chance
                            geometry.
    The overall score is the fuzzy AND (product t-norm) of all three, and the
    highest-scoring pair is selected as the (left piece, right piece) laser
    segments.
 5. The existing least-squares `_fit_line` regression is then run on each
    segment's sub-pixel (xs, ys) arrays exactly as before, the seam x is the
    midpoint of the gap between the two segments, and each fitted line is
    evaluated there to get y_left / y_right -- the bend point.
 6. `draw_seam` marks that bend point with a vertical green line + dot, same
    as before.

USAGE
-----
    python3 seam_detector.py image1.jpg [image2.jpg ...] [options]

    python3 seam_detector.py photo.jpg
    python3 seam_detector.py photo.jpg --roi 200 400 900 1100
    python3 seam_detector.py photo.jpg --ask-roi      # interactively select ROI
    python3 seam_detector.py *.jpeg --outdir results --line-length 400
    python3 seam_detector.py photo.jpg --debug        # also saves the ridge mask

OPTIONS
-------
    --outdir DIR           Output directory for annotated images (default: "seam_output")
    --line-length PX        Half-length (in px, above & below the bend) of the
                             drawn green seam line (default: 250)
    --thickness PX           Thickness of the drawn green line (default: 4)
    --debug                   Also save the intermediate Steger ridge mask for
                             troubleshooting, as "<name>_ridge_mask.png"
    --roi X0 Y0 X1 Y1          Restrict detection to this pixel rectangle
                             (x0 y0 x1 y1), applied to every input image.
    --ask-roi                  Interactively ask for an ROI per image via an
                             OpenCV window (click-drag a rectangle, press
                             ENTER/SPACE to confirm, 'c' to use the full image).
                             Ignored in non-interactive / headless environments.
    --sigma F                   Gaussian smoothing scale for Steger's algorithm,
                             in pixels (default: 2.0). Roughly: should be on
                             the order of the laser stripe's half-width.
    --lambda-thresh F            Hessian eigenvalue threshold controlling ridge
                             strength sensitivity (default: -2.0). More
                             negative = stricter / fewer false positives.

REQUIREMENTS
------------
    pip install opencv-python numpy

OUTPUT
------
For each input "photo.jpg", an annotated copy is written to
"<outdir>/photo_seam.jpg" with a green vertical line + dot marking the
detected seam. A summary is printed to the console for every image,
including the pixel seam location and the TFN pairing confidence. If no seam
can be found, that is reported and the script moves on to the next image
(exit code reflects whether any failures occurred).
"""

import argparse
import os
import sys
from itertools import combinations

import cv2
import numpy as np


# ==========================================================================
# 1. STEGER'S ALGORITHM -- sub-pixel ridge (laser-line) extraction
# ==========================================================================

def _gaussian_derivative_kernels(sigma, ksize=None):
    """Build 1D Gaussian derivative kernels of order 0, 1, and 2 at scale
    `sigma`. These let us compute image derivatives Ix, Iy, Ixx, Iyy, Ixy via
    SEPARABLE convolution while simultaneously performing the Gaussian
    smoothing -- i.e. "scale-space derivatives" -- which is the numerically
    correct way to differentiate a noisy discrete image (as opposed to
    smoothing first and then applying a raw/unscaled Sobel operator).

    For a 1D Gaussian g0(x) = exp(-x^2 / 2*sigma^2) (normalised to sum to 1):
        g1(x) = d/dx g0(x)   = -(x / sigma^2)              * g0(x)
        g2(x) = d2/dx2 g0(x) = ((x^2 - sigma^2) / sigma^4)  * g0(x)
    These are the analytic derivatives of the Gaussian kernel itself.
    """
    if ksize is None:
        # Cover +/- 3 sigma, rounded to the nearest odd integer.
        ksize = int(2 * round(3 * sigma) + 1)
    half = ksize // 2
    x = np.arange(-half, half + 1, dtype=np.float64)

    g0 = np.exp(-(x ** 2) / (2 * sigma ** 2))
    g0 /= g0.sum()  # normalise so the 0th-order kernel is a pure smoother
    g1 = -(x / (sigma ** 2)) * g0
    g2 = ((x ** 2 - sigma ** 2) / (sigma ** 4)) * g0
    return g0, g1, g2


def _separable_filter(img, kx, ky):
    """Apply a separable 2D filter built from 1D kernels kx (horizontal) and
    ky (vertical). E.g. Ixy = _separable_filter(img, g1, g1) applies the
    1st-derivative kernel along both x and y."""
    return cv2.sepFilter2D(img, cv2.CV_64F, kx.reshape(1, -1), ky.reshape(-1, 1))


def _redness_channel(img):
    """Build a scalar "redness" response image that emphasises the red laser
    stripe over background clutter (skin tone, grey walls, metal).

    R - 0.5*G - 0.5*B is a simple, fast opponent-color projection: pure red
    pixels score high, while neutral/grey/skin pixels (where R, G, B are all
    similar) score close to zero, and blue/green dominated pixels go
    negative (clipped to 0). This is far cleaner than raw HSV thresholding
    for feeding a gradient-based ridge detector, since it stays continuous
    (no hard hue-wraparound boundary at 0/180 to fight with).
    """
    b, g, r = cv2.split(img.astype(np.float64))
    redness = r - 0.5 * g - 0.5 * b
    return np.clip(redness, 0, 255)


def _steger_ridge_points(img, sigma=2.0, lambda_thresh=-2.0, roi=None,
                          angle_filter_deg=None):
    """Core of Steger's algorithm: extract sub-pixel ridge (laser centerline)
    points from `img`.

    Parameters
    ----------
    img : np.ndarray (BGR)
    sigma : float
        Gaussian smoothing scale, in pixels. Should be on the order of the
        laser stripe's half-width so the ridge model fits.
    lambda_thresh : float
        Eigenvalue threshold. A pixel is only kept as a ridge candidate if
        its dominant Hessian eigenvalue is more negative than this (i.e. the
        intensity surface curves sharply downward away from a bright ridge
        there). More negative => stricter => fewer false positives.
    roi : (x0, y0, x1, y1) or None
        Optional pixel rectangle to restrict the search to.
    angle_filter_deg : float or None
        If set, only keep ridge points whose tangent direction (the direction
        ALONG the ridge, perpendicular to the Hessian normal) is within
        +/- angle_filter_deg degrees of horizontal. This suppresses the
        diagonal/vertical component in X-pattern lasers. Typical value: 45.

    Returns
    -------
    subx, suby : 1D np.ndarray of float
        Sub-pixel (x, y) image coordinates of every accepted ridge point.

    --- THE MATH ---
    At every pixel we have the local 2nd-order Taylor model of the intensity
    surface I(x, y), captured by the Hessian matrix:

        H = [ Ixx  Ixy ]
            [ Ixy  Iyy ]

    For a SYMMETRIC 2x2 matrix, the eigenvalues have a closed form (no
    iterative solver needed, so this is fully vectorisable in NumPy):

        trace = Ixx + Iyy
        diff  = Ixx - Iyy
        common = sqrt( (diff/2)^2 + Ixy^2 )
        lambda1 = trace/2 + common
        lambda2 = trace/2 - common

    For a bright ridge on a dark/neutral background, the cross-ridge profile
    is concave-down at the ridge peak, so the eigenvalue associated with the
    cross-ridge direction is strongly NEGATIVE; we pick whichever of
    lambda1/lambda2 is more negative as `lambda_main` -- this is the
    curvature across the stripe, and its magnitude measures how sharp /
    well-defined the ridge is at that pixel.

    The corresponding eigenVECTOR (nx, ny) -- for a symmetric matrix it is
    proportional to (Ixy, lambda_main - Ixx) -- points in the direction
    PERPENDICULAR to the ridge (i.e. "across" the laser stripe). This is the
    direction along which we search for the true sub-pixel ridge peak.

    Steger's localisation step: along the line through the pixel center in
    direction (nx, ny), the directional first derivative of I is

        f(t) = grad(I) . (nx, ny)   evaluated at offset t along the normal

    Its zero-crossing (the true ridge peak) is found with a single Newton
    step, using the directional second derivative as the local curvature:

        t = - (Ix*nx + Iy*ny) / (Ixx*nx^2 + 2*Ixy*nx*ny + Iyy*ny^2)

    t is the sub-pixel offset (in pixels) from the pixel center to the ridge
    peak, measured along (nx, ny). The final sub-pixel ridge coordinate is

        (x_subpixel, y_subpixel) = (x + t*nx, y + t*ny)

    A point is only accepted if |t*nx| <= ~0.71 and |t*ny| <= ~0.71 (i.e. the
    correction stays within the pixel, so the local quadratic model can be
    trusted), and if lambda_main < lambda_thresh (a sufficiently sharp ridge,
    not noise).
    """
    if roi is not None:
        x0, y0, x1, y1 = roi
        work = img[y0:y1, x0:x1]
        offset_x, offset_y = x0, y0
    else:
        work = img
        offset_x, offset_y = 0, 0

    redness = _redness_channel(work)

    g0, g1, g2 = _gaussian_derivative_kernels(sigma)

    Ix = _separable_filter(redness, g1, g0)   # d/dx
    Iy = _separable_filter(redness, g0, g1)   # d/dy
    Ixx = _separable_filter(redness, g2, g0)  # d2/dx2
    Iyy = _separable_filter(redness, g0, g2)  # d2/dy2
    Ixy = _separable_filter(redness, g1, g1)  # d2/dxdy

    # --- Closed-form eigen-decomposition of the 2x2 Hessian, vectorised ---
    trace = Ixx + Iyy
    diff = Ixx - Iyy
    common = np.sqrt((diff / 2.0) ** 2 + Ixy ** 2)
    lambda1 = trace / 2.0 + common
    lambda2 = trace / 2.0 - common

    # Pick the eigenvalue with the larger (more negative) magnitude on the
    # negative side -- the cross-ridge curvature.
    pick_lambda1 = lambda1 < lambda2
    lambda_main = np.where(pick_lambda1, lambda1, lambda2)

    # Corresponding eigenvector (unnormalised): e = (Ixy, lambda_main - Ixx).
    # Degenerate case (Ixy == 0 and lambda_main == Ixx, i.e. the Hessian is
    # already diagonal) falls back to the vertical normal (0, 1), which is
    # the correct "across-the-stripe" direction for a near-horizontal laser
    # line -- the expected geometry in this application.
    ex = Ixy
    ey = np.where(pick_lambda1, lambda1 - Ixx, lambda2 - Ixx)
    norm = np.sqrt(ex ** 2 + ey ** 2)
    safe = norm > 1e-9
    nx = np.where(safe, ex / np.maximum(norm, 1e-9), 0.0)
    ny = np.where(safe, ey / np.maximum(norm, 1e-9), 1.0)

    # --- Newton step: sub-pixel zero-crossing of the directional 1st derivative ---
    numerator = Ix * nx + Iy * ny
    denominator = Ixx * nx ** 2 + 2 * Ixy * nx * ny + Iyy * ny ** 2
    denom_safe = np.where(np.abs(denominator) > 1e-9, denominator, 1e-9)
    t = -numerator / denom_safe

    px_offset = t * nx
    py_offset = t * ny

    valid = (
        (lambda_main < lambda_thresh)
        & (np.abs(px_offset) <= 0.71)
        & (np.abs(py_offset) <= 0.71)
    )

    # --- Change 4: orientation filtering for X-pattern lasers ---
    # The normal (nx, ny) is perpendicular to the ridge. The ridge tangent is
    # therefore (-ny, nx). We want ridges whose tangent is roughly horizontal,
    # i.e. the tangent angle w.r.t. the x-axis is small.
    if angle_filter_deg is not None:
        # tangent direction = (-ny, nx);  angle = atan2(nx, -ny)  (from x-axis)
        # We want |angle| <= angle_filter_deg
        tangent_angle = np.abs(np.degrees(np.arctan2(nx, np.where(safe, -ny, 0.0))))
        # Angles > 90 are the same ridge direction flipped, normalise to [0,90]
        tangent_angle = np.minimum(tangent_angle, 180.0 - tangent_angle)
        valid = valid & (tangent_angle <= angle_filter_deg)

    ys, xs = np.where(valid)
    subx = xs + px_offset[ys, xs] + offset_x
    suby = ys + py_offset[ys, xs] + offset_y
    return subx, suby


def _multiscale_steger_ridge_points(img, sigma=2.0, lambda_thresh=-2.0,
                                     roi=None, angle_filter_deg=45.0):
    """Run Steger's ridge extraction at MULTIPLE scales and merge the
    resulting sub-pixel point clouds (Change 3: multi-scale consensus).

    Scales used: sigma * 0.75, sigma, sigma * 1.5.  The true laser ridge
    will be detected at all scales (reinforced); random noise peaks will
    typically only appear at one scale (not reinforced, and filtered out
    naturally by the subsequent connected-component grouping, which requires
    min_pts spatially contiguous points).

    Duplicate sub-pixel points (from multiple scales landing on the same
    integer pixel) are deduplicated by keeping the point from the scale
    with the strongest ridge response (most-negative lambda_main), but in
    practice the simpler approach of just taking the unique rounded-pixel
    positions -- keeping one sub-pixel offset per pixel -- works well and
    is far cheaper.
    """
    sigmas = [sigma * 0.75, sigma, sigma * 1.5]
    all_subx, all_suby = [], []
    for s in sigmas:
        sx, sy = _steger_ridge_points(img, sigma=s, lambda_thresh=lambda_thresh,
                                       roi=roi, angle_filter_deg=angle_filter_deg)
        all_subx.append(sx)
        all_suby.append(sy)

    # Concatenate all points, then deduplicate by rounded pixel coordinate,
    # keeping the first occurrence (from the finest scale, which has the
    # best localisation).
    subx = np.concatenate(all_subx)
    suby = np.concatenate(all_suby)

    if len(subx) == 0:
        return subx, suby

    # Deduplicate: one sub-pixel point per integer pixel
    xi = np.round(subx).astype(int)
    yi = np.round(suby).astype(int)
    # Use a dictionary keyed by (xi, yi) to keep the first occurrence
    seen = {}
    keep = []
    for idx in range(len(subx)):
        key = (xi[idx], yi[idx])
        if key not in seen:
            seen[key] = True
            keep.append(idx)
    keep = np.array(keep, dtype=int)
    return subx[keep], suby[keep]


# ==========================================================================
# 2. Segment grouping (with internal kink-splitting)
# ==========================================================================

def _fit_line(xs, ys):
    """Least-squares fit of y = m*x + b. Returns (m, b)."""
    A = np.vstack([xs, np.ones(len(xs))]).T
    m, b = np.linalg.lstsq(A, ys, rcond=None)[0]
    return float(m), float(b)


def _make_segment(xs, ys):
    return {
        "xs": xs,
        "ys": ys,
        "x0": float(xs.min()),
        "x1": float(xs.max()),
        "y0": float(ys.min()),
        "y1": float(ys.max()),
        "ymean": float(ys.mean()),
    }


def _maybe_split_kink(sx, sy, improve_ratio_thresh=0.2, min_side=8):
    """Given an ordered-by-x polyline of sub-pixel ridge points, test whether
    it contains a strong internal kink.
    
    This sweeps a candidate split index k across the polyline, fits an
    independent line to each side, and finds the k that minimises the
    COMBINED sum-of-squared-residuals (SSE). If that two-line fit is a large
    improvement over a single straight line, the kink is real.
    """
    n = len(sx)
    if n <= 2 * min_side:
        return [_make_segment(sx, sy)]

    A = np.vstack([sx, np.ones(n)]).T
    m, b = np.linalg.lstsq(A, sy, rcond=None)[0]
    sse_single = np.sum((sy - (m * sx + b)) ** 2)
    if sse_single <= 1e-9:
        return [_make_segment(sx, sy)]

    best_k, best_sse = None, None
    for k in range(min_side, n - min_side):
        xL, yL = sx[:k], sy[:k]
        xR, yR = sx[k:], sy[k:]
        AL = np.vstack([xL, np.ones(len(xL))]).T
        mL, bL = np.linalg.lstsq(AL, yL, rcond=None)[0]
        AR = np.vstack([xR, np.ones(len(xR))]).T
        mR, bR = np.linalg.lstsq(AR, yR, rcond=None)[0]
        sse = np.sum((yL - (mL * xL + bL)) ** 2) + np.sum((yR - (mR * xR + bR)) ** 2)
        if best_sse is None or sse < best_sse:
            best_sse, best_k = sse, k

    improvement = (sse_single - best_sse) / sse_single
    if improvement >= improve_ratio_thresh:
        return [_make_segment(sx[:best_k], sy[:best_k]),
                _make_segment(sx[best_k:], sy[best_k:])]
    return [_make_segment(sx, sy)]


def _group_into_segments(subx, suby, img_shape, min_pts=15, dilate_iter=1,
                          kink_improve_ratio=0.3):
    """Group sub-pixel ridge points (from Steger's algorithm) into spatially
    contiguous segments, splitting any segment that contains a strong
    internal kink (a candidate seam location).

    Grouping uses ordinary 8-connectivity connected-components on a binary
    mask built from the ROUNDED sub-pixel coordinates (a light dilation
    bridges the 1-2 px gaps that naturally occur between adjacent sub-pixel
    samples). This is what separates, e.g., the laser segment lying on the
    metal pieces from an unrelated laser segment on the background wall or a
    reflection on nearby wires.
    """
    h, w = img_shape[:2]
    mask = np.zeros((h, w), np.uint8)
    xi = np.clip(np.round(subx).astype(int), 0, w - 1)
    yi = np.clip(np.round(suby).astype(int), 0, h - 1)
    mask[yi, xi] = 255
    mask_dilated = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=dilate_iter)

    num_labels, labels = cv2.connectedComponents(mask_dilated, connectivity=8)
    comp_id = labels[yi, xi]

    segments = []
    for lbl in range(1, num_labels):
        sel = comp_id == lbl
        if sel.sum() < min_pts:
            continue
        sx, sy = subx[sel], suby[sel]
        order = np.argsort(sx)
        sx, sy = sx[order], sy[order]
        segments.extend(_maybe_split_kink(sx, sy, kink_improve_ratio))

    return segments


# ==========================================================================
# 3. TRIANGULAR FUZZY NUMBERS (TFN) -- robust segment-pair scoring
# ==========================================================================

def _triangular_membership(value, a, b, c):
    """Standard Triangular Fuzzy Number (TFN) membership function, defined
    by the triple (a, b, c) with a <= b <= c:

              0                       , value <= a
        mu = (value - a) / (b - a)     , a < value <= b   (rising edge)
             (c - value) / (c - b)     , b < value <  c   (falling edge)
              0                       , value >= c

    This gives a smooth, graceful "how typical is this value of the ideal
    case b?" score in [0, 1], instead of a brittle hard threshold. `a` and
    `c` are the points where membership has fully degraded to zero; `b` is
    the single most-representative ("ideal") value, with membership 1.0.
    """
    if value <= a or value >= c:
        return 0.0
    if value == b:
        return 1.0
    if value < b:
        return (value - a) / (b - a)
    return (c - value) / (c - b)


def _gap_membership(gap):
    """Fuzzy membership for the horizontal gap between two candidate
    segments. The two halves of a laser line interrupted by a seam are
    expected to be nearly touching (small positive gap), with the seam
    itself typically only a few to a few tens of pixels wide.

    TFN(a=-15, b=2, c=45):
      - Peaks at gap=2px (segments almost touching -- the ideal "split right
        at the seam" case).
      - Tolerates a slight NEGATIVE gap (a=-15) to gracefully accommodate
        segments that overlap a little at their rounded sub-pixel boundary
        (rather than a hard `if gap < -5: reject`).
      - Gracefully decays out to c=45px, so a genuinely wider seam gap is
        downweighted smoothly rather than being flatly rejected at a rigid
        "max_gap=60" cutoff.
    """
    return _triangular_membership(gap, a=-15, b=2, c=45)


def _ydiff_membership(y_diff):
    """Fuzzy membership for the vertical misalignment between two candidate
    segments, evaluated at the seam boundary (see `_best_segment_pair`).

    TFN(a=-5, b=0, c=20):
      - Peaks at y_diff=0 (perfectly continuous laser line through the
        seam region -- the laser itself doesn't perfectly align across a
        physical step, but should be close).
      - The left foot at a=-5 simply keeps the function well-defined since
        y_diff is an absolute value (always >= 0); it does not allow
        negative y_diff, it just keeps the rising edge from being a step
        function right at zero.
      - Decays to zero by c=20px, heavily penalising large vertical jumps,
        which indicate two segments are NOT part of the same physical laser
        line (e.g. one is on the workpiece, the other is a reflection
        elsewhere in the scene).
    """
    return _triangular_membership(y_diff, a=-5, b=0, c=20)


def _slope_collinearity_membership(slope_diff):
    """Fuzzy membership for the slope difference between two candidate
    segments (Change 5). Two segments that belong to the same physical laser
    line (split at the seam) should have approximately the same slope, since
    the laser plane is flat.

    TFN(a=-0.02, b=0, c=0.5):
      - Peaks at slope_diff=0 (perfectly parallel segments).
      - Gracefully tolerates small slope differences (the metal step
        introduces a slight slope change).
      - Decays to zero by c=0.5 (about 27 degrees of slope difference),
        heavily penalising segment pairs from completely different laser
        lines (e.g. one horizontal, one diagonal from an X-pattern).
    """
    return _triangular_membership(slope_diff, a=-0.02, b=0.0, c=0.5)


def _length_confidence(n_points, saturation_point=120):
    """Reliability weight in [0, 1] based on how many sub-pixel points
    support a segment's line fit. A segment with very few points (e.g. a
    stray noise fragment) can have a deceptively perfect gap/y_diff purely
    by chance geometry; this factor down-weights such fragments relative to
    long, well-supported ridge segments, whose line fits are statistically
    much more trustworthy. Saturates to 1.0 once a segment has "enough"
    points (>= saturation_point).
    """
    return min(1.0, n_points / float(saturation_point))


def _best_segment_pair(segments):
    """Find the pair of segments most likely to be the two halves of the
    SAME laser line, split by the seam, using a combined Triangular Fuzzy
    Number (TFN) confidence score instead of brittle hard pixel cutoffs.

    For every candidate pair (left, right) -- left being whichever segment
    starts at a smaller x -- we compute:

      gap     = right.x0 - left.x1
                (horizontal pixel gap between the two segments)

      y_diff  = | left_line(seam_x_est) - right_line(seam_x_est) |
                where seam_x_est is the midpoint of the gap, and
                left_line / right_line are each segment's OWN least-squares
                line fit evaluated there. Using each segment's local line
                fit (rather than a crude whole-segment mean y) keeps this
                metric correct even when a segment is long, sloped, or
                slightly curved -- a whole-segment mean can be tens of
                pixels away from the true y at the seam boundary for such
                segments.

      score   = mu_gap(gap) * mu_ydiff(y_diff)
                * confidence(len(left)) * confidence(len(right))

    i.e. the fuzzy AND (product t-norm) of "gap looks seam-like", "vertical
    alignment looks seam-like", and "both segments are well-supported
    enough to trust". The pair with the highest overall score wins.

    Returns
    -------
    (left, right, gap, y_diff, score) or None if fewer than 2 segments exist.
    """
    if len(segments) < 2:
        return None

    # Precompute each segment's own line fit once.
    fits = [_fit_line(s["xs"], s["ys"]) for s in segments]

    best = None
    for i, j in combinations(range(len(segments)), 2):
        a, b = segments[i], segments[j]
        if a["x0"] < b["x0"]:
            left, li, right, ri = a, i, b, j
        else:
            left, li, right, ri = b, j, a, i

        gap = right["x0"] - left["x1"]

        seam_x_est = (left["x1"] + right["x0"]) / 2.0
        mL, bL = fits[li]
        mR, bR = fits[ri]
        y_diff = abs((mL * seam_x_est + bL) - (mR * seam_x_est + bR))

        mu_gap = _gap_membership(gap)
        mu_ydiff = _ydiff_membership(y_diff)
        conf = _length_confidence(len(left["xs"])) * _length_confidence(len(right["xs"]))

        score = mu_gap * mu_ydiff * conf

        if best is None or score > best[4]:
            best = (left, right, gap, y_diff, score)

    return best


# ==========================================================================
# Top-level seam detection
# ==========================================================================

def detect_seam(img, roi=None, sigma=2.0, lambda_thresh=-2.0,
                 min_score=1e-6):
    """Detect the seam point between two metal pieces from a laser-line photo.

    Parameters
    ----------
    img : np.ndarray
        BGR image (as loaded by cv2.imread).
    roi : (x0, y0, x1, y1) or None
        Optional region of interest to restrict the search to.
    sigma : float
        Gaussian smoothing scale for Steger's algorithm (px).
    lambda_thresh : float
        Hessian eigenvalue threshold for ridge-strength sensitivity.
    min_score : float
        Minimum acceptable TFN pairing score; pairs scoring below this are
        treated as "no plausible seam found".

    Returns
    -------
    dict or None
        On success, a dict with keys:
            seam_x      : x pixel coordinate of the seam
            y_left      : y coordinate of the left segment's fitted line at seam_x
            y_right     : y coordinate of the right segment's fitted line at seam_x
            y_bend      : average of y_left/y_right -- the point to mark
            left_seg    : dict describing the left laser-line segment
            right_seg   : dict describing the right laser-line segment
            gap_px      : horizontal pixel gap between the two segments
            y_diff_px   : vertical misalignment at the seam boundary
            tfn_score   : combined fuzzy pairing confidence in [0, 1]
        Returns None if no plausible seam could be found.
    """
    # Adaptive two-pass strategy:
    #   Pass 1: multi-scale extraction WITH angle filtering (suppresses X-pattern diagonals)
    #   Pass 2: multi-scale extraction WITHOUT angle filtering (original behaviour)
    # Pick whichever pass yields the best TFN score.
    
    # Pass 1 (angle-filtered): helps with X-pattern lasers
    result1 = _run_detection_pipeline(img, sigma, lambda_thresh, roi, angle_filter_deg=45.0)
    # Pass 2 (unfiltered): original behaviour, preferred for most images
    result2 = _run_detection_pipeline(img, sigma, lambda_thresh, roi, angle_filter_deg=None)

    # Prefer the unfiltered pass (result2) as the primary result.
    # Only use the filtered pass (result1) when:
    #   (a) result2 failed entirely, OR
    #   (b) result1's score is significantly higher (>50%), indicating the
    #       angle filter genuinely helped (e.g. X-pattern images).
    r1_ok = result1 is not None and result1["tfn_score"] >= min_score
    r2_ok = result2 is not None and result2["tfn_score"] >= min_score

    if r2_ok and r1_ok:
        if result1["tfn_score"] > result2["tfn_score"] * 1.5:
            return result1
        return result2
    elif r2_ok:
        return result2
    elif r1_ok:
        return result1
    return None


def _run_detection_pipeline(img, sigma, lambda_thresh, roi, angle_filter_deg):
    """Run one pass of the full detection pipeline (ridge extraction ->
    segment grouping -> TFN pairing -> bend-point localisation).

    This is the inner workhorse called by detect_seam for each pass of
    its adaptive strategy.
    """
    subx, suby = _multiscale_steger_ridge_points(
        img, sigma=sigma, lambda_thresh=lambda_thresh,
        roi=roi, angle_filter_deg=angle_filter_deg)

    if len(subx) < 2:
        return None

    segments = _group_into_segments(subx, suby, img.shape)
    if len(segments) < 2:
        return None

    pair = _best_segment_pair(segments)
    if pair is None:
        return None

    left, right, gap, y_diff, score = pair

    m1, b1 = _fit_line(left["xs"], left["ys"])
    m2, b2 = _fit_line(right["xs"], right["ys"])

    # Change 1: line-intersection bend point
    gap_mid_x = (left["x1"] + right["x0"]) / 2.0
    slope_diff_val = m1 - m2
    if abs(slope_diff_val) > 1e-6:
        intersect_x = (b2 - b1) / slope_diff_val
        gap_half = max(abs(gap) / 2.0, 15.0)
        clamp_lo = left["x1"] - gap_half
        clamp_hi = right["x0"] + gap_half
        intersect_x = max(clamp_lo, min(clamp_hi, intersect_x))
        seam_x = intersect_x
    else:
        seam_x = gap_mid_x

    y_left = m1 * seam_x + b1
    y_right = m2 * seam_x + b2

    return {
        "seam_x": seam_x,
        "y_left": y_left,
        "y_right": y_right,
        "y_bend": (y_left + y_right) / 2.0,
        "left_seg": left,
        "right_seg": right,
        "gap_px": gap,
        "y_diff_px": y_diff,
        "tfn_score": score,
    }


# --------------------------------------------------------------------------
# Drawing / annotation
# --------------------------------------------------------------------------

def draw_seam(img, result, line_length=250, thickness=4,
              color=(0, 255, 0), dot_radius=8):
    """Return a copy of img with the detected seam drawn as a vertical green
    line (plus a dot at the precise bend point)."""
    out = img.copy()
    h, w = out.shape[:2]

    seam_x = int(round(result["seam_x"]))
    y_bend = int(round(result["y_bend"]))

    y_top = max(y_bend - line_length, 0)
    y_bot = min(y_bend + line_length, h - 1)

    cv2.line(out, (seam_x, y_top), (seam_x, y_bot), color, thickness)
    cv2.circle(out, (seam_x, y_bend), dot_radius, color, -1)

    return out


# --------------------------------------------------------------------------
# Optional interactive ROI selection
# --------------------------------------------------------------------------

def _ask_roi_interactive(img, window_name="Select ROI (drag a box, ENTER to confirm, 'c' = full image)"):
    """Open an OpenCV window letting the user drag-select a rectangular ROI.
    Returns (x0, y0, x1, y1) or None if the user pressed 'c' / cancelled, in
    which case the caller should fall back to using the full image.

    Falls back gracefully (returns None) if no display is available, e.g. in
    a headless/server environment where cv2.imshow would raise.
    """
    try:
        r = cv2.selectROI(window_name, img, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow(window_name)
    except cv2.error:
        print("         (no display available -- skipping interactive ROI, using full image)")
        return None

    x, y, w, h = r
    if w == 0 or h == 0:
        return None  # user cancelled / pressed 'c' without dragging
    return (int(x), int(y), int(x + w), int(y + h))


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def process_image(path, outdir, line_length, thickness, debug,
                   roi=None, ask_roi=False, sigma=2.0, lambda_thresh=-2.0):
    img = cv2.imread(path)
    if img is None:
        print(f"[FAIL] {path}: could not read image")
        return False

    base = os.path.splitext(os.path.basename(path))[0]

    effective_roi = roi
    if ask_roi:
        print(f"         select an ROI for {path} (or press 'c' for full image)...")
        effective_roi = _ask_roi_interactive(img)

    result = detect_seam(img, roi=effective_roi, sigma=sigma, lambda_thresh=lambda_thresh)

    if debug:
        subx, suby = _steger_ridge_points(img, sigma=sigma, lambda_thresh=lambda_thresh, roi=effective_roi)
        ridge_mask = np.zeros(img.shape[:2], np.uint8)
        xi = np.clip(np.round(subx).astype(int), 0, img.shape[1] - 1)
        yi = np.clip(np.round(suby).astype(int), 0, img.shape[0] - 1)
        ridge_mask[yi, xi] = 255
        mask_path = os.path.join(outdir, f"{base}_ridge_mask.png")
        cv2.imwrite(mask_path, ridge_mask)
        print(f"         debug ridge mask saved -> {mask_path}")

    if result is None:
        print(f"[FAIL] {path}: no seam detected "
              f"(could not find two matching laser-line segments)")
        return False

    annotated = draw_seam(img, result, line_length=line_length, thickness=thickness)
    out_path = os.path.join(outdir, f"{base}_seam.jpg")
    cv2.imwrite(out_path, annotated)

    print(f"[ OK ] {path}")
    print(f"         seam x       = {result['seam_x']:.2f} px")
    print(f"         y (left)     = {result['y_left']:.2f} px")
    print(f"         y (right)    = {result['y_right']:.2f} px")
    print(f"         bend point   = ({result['seam_x']:.2f}, {result['y_bend']:.2f})")
    print(f"         gap / y_diff = {result['gap_px']:.2f}px / {result['y_diff_px']:.2f}px")
    print(f"         TFN score    = {result['tfn_score']:.3f}")
    print(f"         saved        -> {out_path}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Detect the seam between two metal pieces by finding "
                    "where a laser line crossing them bends (sub-pixel, via "
                    "Steger's algorithm + fuzzy segment pairing), and mark "
                    "it with a green line.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("images", nargs="+", help="Path(s) to input image(s)")
    parser.add_argument("--outdir", default="seam_output",
                         help="Output directory (default: seam_output)")
    parser.add_argument("--line-length", type=int, default=250,
                         help="Half-length in px of the drawn seam line (default: 250)")
    parser.add_argument("--thickness", type=int, default=4,
                         help="Thickness in px of the drawn seam line (default: 4)")
    parser.add_argument("--debug", action="store_true",
                         help="Also save the intermediate Steger ridge mask")
    parser.add_argument("--roi", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"),
                         default=None,
                         help="Restrict detection to this pixel rectangle, "
                              "applied to every input image.")
    parser.add_argument("--ask-roi", action="store_true",
                         help="Interactively select an ROI per image (requires a display).")
    parser.add_argument("--sigma", type=float, default=2.0,
                         help="Gaussian smoothing scale in px for Steger's algorithm (default: 2.0)")
    parser.add_argument("--lambda-thresh", type=float, default=-2.0,
                         help="Hessian eigenvalue threshold for ridge sensitivity (default: -2.0)")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    all_ok = True
    for path in args.images:
        ok = process_image(
            path,
            outdir=args.outdir,
            line_length=args.line_length,
            thickness=args.thickness,
            debug=args.debug,
            roi=tuple(args.roi) if args.roi else None,
            ask_roi=args.ask_roi,
            sigma=args.sigma,
            lambda_thresh=args.lambda_thresh,
        )
        all_ok = all_ok and ok

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
