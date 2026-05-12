"""Деконволюция записи с инверсным фильтром → импульсная характеристика.

Два способа сделать одно и то же:

1) **Time-domain convolution** записанного отклика с заранее построенным
   инверсным фильтром (см. `sweep.inverse_filter_ess`). Это «честный» метод
   Farina — он работает с любыми (даже не-period) сигналами, и даёт правильное
   разделение полезной IR и нелинейных искажений во времени.

2) **Frequency-domain division** Y(f) / S(f) с регуляризацией. Быстро и просто,
   но игнорирует нелинейные искажения (они «расплываются» по всей IR).

Для курсовой реализован метод (1) как основной. Метод (2) доступен как
альтернатива.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve


def deconvolve_time(
    recorded: np.ndarray,
    inverse_filter: np.ndarray,
) -> np.ndarray:
    """Получить IR через свёртку записи с инверсным фильтром (метод Farina).

    Математически:
        y(t) = (s * h)(t) + n(t)        — записанный сигнал
        ŷ(t) = y(t) * g(t)              — свёртка с инверсным фильтром g
              = ((s * g) * h)(t) + (n * g)(t)
              ≈ δ(t − T) * h(t) + шум
              = h(t − T) + шум

    То есть в ŷ(t) импульсная характеристика появляется со сдвигом T (длина свипа).
    Этот сдвиг отрезается на следующем этапе при поиске t₀.

    Args:
        recorded: записанный отклик (моно), форма (N,).
        inverse_filter: инверсный фильтр g(t), форма (M,).

    Returns:
        ir_full: «сырая» IR длины N + M − 1 со сдвигом, до обрезки.

    Note:
        Используется fftconvolve — асимптотически O(N log N), быстро даже для
        длинных свипов (10–20 с при 48 кГц = 480000–960000 отсчётов).
    """
    if recorded.ndim != 1:
        raise ValueError("recorded должен быть одномерным массивом (моно).")
    if inverse_filter.ndim != 1:
        raise ValueError("inverse_filter должен быть одномерным массивом.")

    # fftconvolve дешевле прямой свёртки для таких длин
    ir_full = fftconvolve(recorded, inverse_filter, mode="full")
    return ir_full


def deconvolve_freq(
    recorded: np.ndarray,
    played: np.ndarray,
    reg: float = 1e-3,
) -> np.ndarray:
    """Альтернативный метод: деконволюция в частотной области с регуляризацией.

    H(f) = Y(f) · conj(S(f)) / ( |S(f)|² + ε² )

    Регуляризация ε защищает от деления на ноль на частотах, где S(f) мала
    (т.е. вне полосы свипа).

    Args:
        recorded: записанный отклик y(t).
        played:   воспроизводившийся сигнал s(t).
        reg:      регуляризационный параметр, относительно max|S(f)|.

    Returns:
        ir: импульсная характеристика, длиной max(len(recorded), len(played)).

    Note:
        В отличие от метода Farina, здесь нелинейные искажения остаются
        внутри IR и могут добавить артефактов. Для качественных измерений
        предпочитайте deconvolve_time.
    """
    n = max(len(recorded), len(played))
    nfft = int(2 ** np.ceil(np.log2(2 * n)))

    Y = np.fft.rfft(recorded, n=nfft)
    S = np.fft.rfft(played, n=nfft)

    eps2 = (reg * np.max(np.abs(S))) ** 2
    H = Y * np.conj(S) / (np.abs(S) ** 2 + eps2)

    ir = np.fft.irfft(H, n=nfft)
    return ir[:n]


def trim_ir(
    ir_full: np.ndarray,
    sweep_length: int,
    pre_pad: int = 1024,
    post_length: int | None = None,
) -> tuple[np.ndarray, int]:
    """Обрезать «сырой» результат свёртки до полезной IR.

    После deconvolve_time главный пик находится примерно в районе
    `sweep_length` отсчётов от начала. Слева от него — задержка системы плюс
    нелинейные искажения. Справа — линейная IR помещения.

    Args:
        ir_full: результат deconvolve_time.
        sweep_length: длина исходного ESS в отсчётах.
        pre_pad: сколько отсчётов оставить ДО ожидаемого пика (запас для
                 поиска t₀ с корректным контекстом).
        post_length: сколько отсчётов взять ПОСЛЕ ожидаемого пика. Если None,
                     берём всё что есть.

    Returns:
        ir: обрезанная IR.
        offset: индекс в `ir`, соответствующий ожидаемому пику
                (т.е. ir[offset] ~ начало полезного сигнала).
    """
    expected_peak = sweep_length
    start = max(0, expected_peak - pre_pad)
    if post_length is not None:
        end = min(len(ir_full), expected_peak + post_length)
    else:
        end = len(ir_full)

    ir = ir_full[start:end].copy()
    offset = expected_peak - start
    return ir, offset


def normalize_ir(ir: np.ndarray, target_peak: float = 1.0) -> np.ndarray:
    """Нормировать IR так, чтобы пик абсолютной величины был = target_peak.

    Полезно для отображения и сравнения разных измерений.
    """
    peak = np.max(np.abs(ir))
    if peak < 1e-30:
        return ir.copy()
    return ir * (target_peak / peak)
