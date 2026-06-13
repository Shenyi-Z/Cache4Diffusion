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
"""
from typing import Dict, List, Optional

import torch


def _dmd_forecast(snapshots: List[torch.Tensor], k: float, rank: int = 0,
                  ridge: float = 1e-8) -> Optional[torch.Tensor]:
    """
    Forecast a feature `k` snapshot-spacings past the newest snapshot via Dynamic
    Mode Decomposition (Prony): identify the linear propagator A of the snapshot
    sequence (Y_{t+1} ~= A Y_t) through one economy SVD, eigendecompose it once,
    and advance by (possibly fractional) eigenvalue powers:

        Y_{t+k} ~= Phi @ (lambda**k * b),    b = pinv(Phi) @ Y_t

    :param snapshots: >= 3 same-shape tensors, OLDEST..NEWEST, uniformly spaced
    :param k: horizon in snapshot-spacing units (fractional allowed)
    :param rank: SVD truncation rank (0 = pick from the spectrum; rejects noise)
    :param ridge: Tikhonov term added to the inverted singular values
    :returns: forecast tensor, or None when degenerate (caller falls back)
    """
    if len(snapshots) < 3:
        return None
    shp, dt = snapshots[-1].shape, snapshots[-1].dtype
    V = torch.stack([s.reshape(-1) for s in snapshots], dim=1).to(torch.float64)
    X, Xp = V[:, :-1], V[:, 1:]
    try:
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    except Exception:
        return None
    if rank <= 0:
        rank = int((S > S[0] * 1e-4).sum().clamp(min=1).item())
    rank = max(1, min(rank, S.numel()))
    Ur, Sr, Vr = U[:, :rank], S[:rank], Vh[:rank].mH
    Sinv = (1.0 / (Sr + ridge)).to(torch.complex128)
    Atil = (Ur.mH @ Xp @ Vr).to(torch.complex128) * Sinv.unsqueeze(0)
    try:
        evals, W = torch.linalg.eig(Atil)
        Phi = ((Xp @ Vr).to(torch.complex128) * Sinv.unsqueeze(0)) @ W
        b = torch.linalg.lstsq(Phi, V[:, -1].to(torch.complex128).unsqueeze(1)).solution.squeeze(1)
    except Exception:
        return None
    pred = (Phi @ (evals.pow(float(k)) * b)).real
    if not torch.isfinite(pred).all():
        return None
    return pred.to(dt).reshape(shp)


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
    stream = store.setdefault(_stream_key(current), [])
    # detach().clone(): a bare detached view shares storage with the pipeline
    # tensor, so buffer-reusing inference could overwrite the history in place.
    stream.append((current['step'], feature.detach().clone()))
    history = cache_dic.get('dmd_history', 6)
    if len(stream) > history:
        del stream[: len(stream) - history]


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

    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    """
    stream = cache_dic.get('dmd_snapshots', {}).get(_stream_key(current), [])
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
    return _dmd_forecast(vels, k, rank=cache_dic.get('dmd_rank', 0),
                         ridge=cache_dic.get('dmd_ridge', 1e-8))
