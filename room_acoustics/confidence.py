"""Карта доверия для квази-безэховой АЧХ.

Идея: на каждой частоте f мы знаем:
    • Какая длина окна реально использовалась — T(f)
    • Сколько периодов сигнала поместилось — N(f) = T(f) · f
    • Уровень сигнала относительно шумового пола — SNR(f)

Из этих величин строим классификацию:

    GREEN  — данные достоверны:
             N(f) >= n_cycles_target  AND  SNR(f) >= snr_green_db
    YELLOW — данные на границе:
             N(f) >= n_cycles_target/2  AND  SNR(f) >= snr_yellow_db
    RED    — данные недостоверны:
             всё остальное.

Этот цветовой код накладывается на график АЧХ как заштрихованная полоса
неопределённости. Дополнительно для каждой частоты можно посчитать
доверительный интервал в дБ (см. magnitude_uncertainty_db).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .windowing import FreqDependentResult


class ConfidenceLevel(Enum):
    GREEN = 2   # данные достоверны
    YELLOW = 1  # данные ограниченно достоверны
    RED = 0     # данные недостоверны


@dataclass
class ConfidenceMap:
    """Карта доверия для АЧХ."""

    freqs: np.ndarray              # та же сетка, что в FreqDependentResult
    levels: np.ndarray             # int массив ConfidenceLevel.value
    snr_db: np.ndarray             # отношение сигнал/шум на каждой f, дБ
    magnitude_uncertainty_db: np.ndarray  # верхняя оценка погрешности в дБ


def estimate_snr_per_band(
    ir: np.ndarray,
    fs: int,
    direct_idx: int,
    bands_freqs: np.ndarray,
    tail_fraction: float = 0.2,
    band_fraction: float = 1 / 12,
) -> np.ndarray:
    """Грубая оценка SNR для каждой полосы.

    Сигнал = энергия IR в полосе, в окне «прямой звук + раннее отражение»
             (первые 50 мс после direct_idx).
    Шум    = энергия IR в полосе в хвосте (последние tail_fraction).

    SNR(f) = 10 · log10( E_signal(f) / E_noise(f) )

    Args:
        ir, fs, direct_idx: стандартные.
        bands_freqs: центральные частоты полос (как из windowing).
        tail_fraction: доля IR с конца для оценки шума.
        band_fraction: ширина полосы для интегрирования.

    Returns:
        snr_db той же длины, что bands_freqs.
    """
    # FFT всего IR (для сигнала: первые 50 мс после direct)
    signal_end = min(len(ir), direct_idx + int(0.05 * fs))
    sig_seg = ir[direct_idx:signal_end]
    n_tail = int(tail_fraction * len(ir))
    noise_seg = ir[-n_tail:] if n_tail > 0 else ir[-1:]

    nfft_sig = max(8192, 2 ** int(np.ceil(np.log2(len(sig_seg) * 2))))
    nfft_noise = max(8192, 2 ** int(np.ceil(np.log2(len(noise_seg) * 2))))

    SIG = np.abs(np.fft.rfft(sig_seg, n=nfft_sig))
    NOISE = np.abs(np.fft.rfft(noise_seg, n=nfft_noise))

    sig_freqs = np.fft.rfftfreq(nfft_sig, d=1.0 / fs)
    noise_freqs = np.fft.rfftfreq(nfft_noise, d=1.0 / fs)

    snr_db = np.empty_like(bands_freqs)
    for i, f in enumerate(bands_freqs):
        f_lo = f * 2.0 ** (-band_fraction / 2.0)
        f_hi = f * 2.0 ** (+band_fraction / 2.0)
        sig_mask = (sig_freqs >= f_lo) & (sig_freqs <= f_hi)
        noise_mask = (noise_freqs >= f_lo) & (noise_freqs <= f_hi)
        e_sig = float(np.mean(SIG[sig_mask] ** 2)) if sig_mask.any() else 0.0
        e_noise = float(np.mean(NOISE[noise_mask] ** 2)) if noise_mask.any() else 0.0
        if e_noise < 1e-30:
            snr_db[i] = 80.0
        else:
            snr_db[i] = 10.0 * np.log10((e_sig + 1e-30) / e_noise)
    return snr_db


def build_confidence_map(
    fd_result: FreqDependentResult,
    ir: np.ndarray,
    fs: int,
    direct_idx: int,
    snr_green_db: float = 25.0,
    snr_yellow_db: float = 10.0,
    cycles_yellow_factor: float = 0.5,
    band_fraction: float = 1 / 12,
) -> ConfidenceMap:
    """Построить карту доверия из результата частотно-зависимого анализа.

    Правила (комбинированный критерий — берётся худший из двух):
        GREEN   ⟺ N(f) ≥ N_target  AND  SNR(f) ≥ snr_green_db
        YELLOW  ⟺ N(f) ≥ N_target·cycles_yellow_factor
                  AND  SNR(f) ≥ snr_yellow_db
        RED     иначе.

    magnitude_uncertainty_db оценивается как:
        Δdb ≈ 10 · log10(1 + 1/N(f))   — недостаток числа циклов
              ⊕  10 · log10(1 + 10^(−SNR/10))   — шум
    (обе компоненты суммируются по дБ, упрощённая модель).
    """
    n_cycles_target = fd_result.n_cycles_target
    n_actual = fd_result.n_cycles_actual

    snr_db = estimate_snr_per_band(
        ir, fs, direct_idx, fd_result.freqs, band_fraction=band_fraction
    )

    levels = np.empty_like(fd_result.freqs, dtype=int)
    for i in range(len(fd_result.freqs)):
        is_full_cycles = n_actual[i] >= n_cycles_target * 0.95  # 5% допуск
        is_half_cycles = n_actual[i] >= n_cycles_target * cycles_yellow_factor
        is_good_snr = snr_db[i] >= snr_green_db
        is_ok_snr = snr_db[i] >= snr_yellow_db

        if is_full_cycles and is_good_snr:
            levels[i] = ConfidenceLevel.GREEN.value
        elif is_half_cycles and is_ok_snr:
            levels[i] = ConfidenceLevel.YELLOW.value
        else:
            levels[i] = ConfidenceLevel.RED.value

    # Оценка погрешности
    # «Cycle deficit»: если N < N_target, то ошибка дискретизации спектра растёт
    cycle_deficit_db = np.where(
        n_actual > 0,
        np.abs(10.0 * np.log10(1.0 + (n_cycles_target / np.maximum(n_actual, 1e-3)))),
        20.0,
    )
    cycle_deficit_db = np.minimum(cycle_deficit_db, 20.0)
    # «Noise»: шум проявляется как ошибка ~10·log10(1 + 10^(−SNR/10))
    noise_unc_db = 10.0 * np.log10(1.0 + 10.0 ** (-np.clip(snr_db, -40, 80) / 10.0))
    uncertainty_db = np.sqrt(cycle_deficit_db ** 2 + noise_unc_db ** 2)

    return ConfidenceMap(
        freqs=fd_result.freqs.copy(),
        levels=levels,
        snr_db=snr_db,
        magnitude_uncertainty_db=uncertainty_db,
    )
