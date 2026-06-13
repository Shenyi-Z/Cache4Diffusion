"""Dynamic Mode Decomposition (Prony) forecast utilities — an EXPONENTIAL-basis
upgrade for the TaylorSeer cache.

Note: "DMD" here is Dynamic Mode Decomposition (Schmid 2010), the SVD-regularised
multivariate generalisation of Prony's method (1795) — NOT Distribution Matching
Distillation.

Why: across denoising steps each cached feature evolves under a (slowly varying)
near-linear operator, so locally it is a sum of damped / oscillatory EXPONENTIALS
— the exact solution class of a linear feature-ODE. The Taylor expansion is only a
local polynomial truncation of that class and diverges as the fresh interval grows;
the exponential basis is exact on the class, so it keeps quality at larger
``fresh_threshold``.

Storage convention mirrors taylorseer_utils: snapshots are kept per
(stream, layer, module) in ``cache_dic['dmd_snapshots']``, recorded at 'full'
steps and consumed by ``dmd_formula`` at 'Taylor' (forecast) steps. Below the
4-snapshot identifiability floor (or on a non-uniform compute cadence) the caller
falls back to ``taylor_formula`` automatically.

The horizon-free DMD fit (the SVD/eig) is cached per stream in
``cache_dic['dmd_fit']`` and recomputed only when a new snapshot shifts the
window, so a long fresh interval does not re-decompose the same snapshots on
every skipped step.
"""
from typing import Dict, List, Optional

import torch


def _dmd_fit_one(traj: torch.Tensor, rank: int = 0, ridge: float = 1e-8):
    """Fit the DMD eigendecomposition for ONE ``[d, n]`` trajectory (one batch
    item's snapshot history, columns OLDEST..NEWEST): identify the propagator A
    (``Y_{t+1} ~= A Y_t``) via one economy SVD and eigendecompose it once. The
    fit is horizon-free, so the caller caches it and advances it cheaply.

    :param rank: SVD truncation rank (0 = pick from the spectrum; rejects noise)
    :param ridge: Tikhonov term added to the inverted singular values
    :returns: ``(Phi, evals, b)`` or ``None`` on a degenerate fit.
    """
    X, Xp = traj[:, :-1], traj[:, 1:]
    try:
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    except Exception:
        return None
    r = rank
    if r <= 0:
        r = int((S > S[0] * 1e-4).sum().clamp(min=1).item())
    r = max(1, min(r, S.numel()))
    Ur, Sr, Vr = U[:, :r], S[:r], Vh[:r].mH
    Sinv = (1.0 / (Sr + ridge)).to(torch.complex128)
    Atil = (Ur.mH @ Xp @ Vr).to(torch.complex128) * Sinv.unsqueeze(0)
    try:
        evals, W = torch.linalg.eig(Atil)
        Phi = ((Xp @ Vr).to(torch.complex128) * Sinv.unsqueeze(0)) @ W
        b = torch.linalg.lstsq(
            Phi, traj[:, -1].to(torch.complex128).unsqueeze(1)
        ).solution.squeeze(1)
    except Exception:
        return None
    return (Phi, evals, b)


def _dmd_fit(snapshots: List[torch.Tensor], rank: int = 0, ridge: float = 1e-8):
    """Fit DMD once for a window of >= 4 same-shape snapshots, INDEPENDENTLY per
    batch item (axis 0).

    Flattening the whole tensor into a single state (the pre-fix behaviour)
    folds the batch dimension into one DMD fit, so one prompt's forecast would
    depend on the other prompts in the same batch — unlike the elementwise
    Taylor path. Fitting per batch item keeps them independent. The fit is
    horizon-free, so ``dmd_formula`` caches it per stream and reuses it for every
    skip step (one SVD/eig per window, not per skipped step).

    :returns: ``(per_item_fits, shape, dtype)`` or ``None`` when too short;
        ``per_item_fits[i]`` is ``None`` for a degenerate batch item.
    """
    if len(snapshots) < 4:
        return None
    newest = snapshots[-1]
    shp, dt = newest.shape, newest.dtype
    bsz = shp[0] if newest.dim() > 1 else 1
    # (B, d, n): per-item trajectories; B == 1 reproduces the un-batched fit.
    V = torch.stack(
        [s.reshape(bsz, -1) for s in snapshots], dim=-1
    ).to(torch.float64)
    fits = [_dmd_fit_one(V[i], rank=rank, ridge=ridge) for i in range(bsz)]
    return (fits, shp, dt)


def _dmd_eval(fit, k: float) -> Optional[torch.Tensor]:
    """Advance a cached :func:`_dmd_fit` to (fractional) horizon ``k`` by
    eigenvalue powers (``Y_{t+k} ~= Phi @ (lambda**k * b)``) — one cheap
    evaluation per batch item, no re-decomposition.

    :returns: forecast tensor of the snapshot shape, or ``None`` when any batch
        item is degenerate, or the result is non-finite AFTER the output-dtype
        cast. The guard is deliberately post-cast: a finite float64 forecast can
        still overflow to ``inf`` in fp16, which the caller must fall back from
        rather than feed downstream.
    """
    fits, shp, dt = fit
    rows = []
    for f in fits:
        if f is None:
            return None
        Phi, evals, b = f
        rows.append((Phi @ (evals.pow(float(k)) * b)).real)
    pred = torch.stack(rows, dim=0) if len(shp) > 1 else rows[0]
    out = pred.to(dt).reshape(shp)
    if not torch.isfinite(out).all():
        return None
    return out


def _stream_key(current: Dict) -> tuple:
    return (current['stream'], current['layer'], current['module'])


def dmd_snapshot_update(cache_dic: Dict, current: Dict, feature: torch.Tensor):
    """
    Record a fully computed feature as the newest snapshot for its
    (stream, layer, module), keeping the most recent ``cache_dic['dmd_history']``.
    Called from ``derivative_approximation`` at 'full' steps when
    ``cache_dic['forecast_basis'] == 'dmd'``.

    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    :param feature: Fully computed feature at this step
    """
    store = cache_dic.setdefault('dmd_snapshots', {})
    key = _stream_key(current)
    stream = store.setdefault(key, [])
    # detach().clone(): a bare detached view shares storage with the pipeline
    # tensor, so buffer-reusing inference could overwrite the history in place.
    stream.append((current['step'], feature.detach().clone()))
    history = cache_dic.get('dmd_history', 6)
    if len(stream) > history:
        del stream[: len(stream) - history]
    # A new snapshot changes the window, so drop this stream's cached fit.
    cache_dic.get('dmd_fit', {}).pop(key, None)


def dmd_formula(cache_dic: Dict, current: Dict) -> Optional[torch.Tensor]:
    """
    Dynamic Mode Decomposition (Prony) forecast of the feature at the current
    step, or None so the caller falls back to ``taylor_formula``.

    Only the longest *uniformly spaced* suffix of the snapshot history is used —
    the identified propagator advances exactly one snapshot-spacing per
    application, so a mixed-spacing window (e.g. across the first_enhance
    boundary) would corrupt the fit. The horizon is fractional in
    snapshot-spacing units. The window floor is 4 snapshots: a real-valued
    trajectory spends two real degrees of freedom per complex pole, so even one
    oscillatory mode needs three snapshot pairs to identify.

    The eigendecomposition depends only on the snapshot window (not on the
    horizon), and the window cannot change between two skip steps, so it is
    fitted once and cached per stream in ``cache_dic['dmd_fit']``; each skip step
    only re-advances the cheap ``lambda**k`` horizon.

    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    """
    key = _stream_key(current)
    stream = cache_dic.get('dmd_snapshots', {}).get(key, [])
    if len(stream) < 4:
        return None
    steps = [s for s, _ in stream]
    spacing = steps[-1] - steps[-2]
    if spacing <= 0:
        return None
    tail = [stream[-1], stream[-2]]
    j = len(stream) - 2
    while j - 1 >= 0 and steps[j] - steps[j - 1] == spacing:
        tail.append(stream[j - 1])
        j -= 1
    if len(tail) < 4:
        return None
    vels = [v for _, v in reversed(tail)]
    k = (current['step'] - steps[-1]) / spacing
    fit_store = cache_dic.setdefault('dmd_fit', {})
    fit_key = (steps[-1], len(tail), spacing)
    cached = fit_store.get(key)
    if cached is None or cached[0] != fit_key:
        cached = (fit_key, _dmd_fit(vels, rank=cache_dic.get('dmd_rank', 0),
                                    ridge=cache_dic.get('dmd_ridge', 1e-8)))
        fit_store[key] = cached
    fit = cached[1]
    if fit is None:
        return None
    return _dmd_eval(fit, k)
