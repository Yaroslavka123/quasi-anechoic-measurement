"""Частотно-зависимое окно для IR — главная фишка проекта.

Идея
====

Стандартный подход к квази-безэховому измерению — взять IR, обрезать окном
длины T_win от прямого звука до первого отражения, и сделать FFT. Проблема:
    f_min = 1/T_win → если T_win = 5 мс, то ниже 200 Гц данные недостоверны.

При этом на ВЫСОКИХ частотах мы зря тратим окно — оно длиннее, чем нужно для
частотного разрешения. Длинное окно на ВЧ означает, что в FFT попадает больше
шума, и АЧХ становится грязнее.

Идея частотно-зависимого окна:

    T(f) = min( N_cycles / f,  T_win )

— на каждой частоте f используем окно ровно той длины, которая нужна для
N_cycles периодов сигнала. На НЧ это упирается в потолок T_win.

Это даёт:
    • Минимальный шум на ВЧ (короткое окно = меньше захвачено шума).
    • Корректную нижнюю границу: на f < N_cycles/T_win данные помечаются как
      «недостоверные» в confidence map.
    • Достоверную АЧХ в широкой полосе при ограниченном T_win.

Реализация
==========

Реализован через **banded analysis**: спектр делится на полосы по 1/N октавы,
в каждой полосе IR окнятся индивидуально, делается FFT, берутся значения
в полосе, склеиваются в итоговую АЧХ.

Это надёжно, контролируемо и легко объясняется в курсовой.

Альтернативные реализации (CWT, Constant-Q) тоже возможны — упомянуты для
полноты, но в проекте используется bands-метод.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def cosine_taper(n: int, taper_frac: float = 0.25) -> np.ndarray:
    """Косинусное (Tukey) окно длины n.

    Левый край — плоский (для прямого звука, который должен войти полностью),
    правый край — половина Hann-окна, плавно спадает к нулю. Это лучше, чем
    прямоугольное окно: меньше spectral leakage в FFT.

    Args:
        n: длина окна в отсчётах.
        taper_frac: доля окна, занятая спадающей частью. 0.25 = четверть
                    окна плавно убывает к нулю в конце.
    """
    w = np.ones(n)
    n_taper = int(n * taper_frac)
    if n_taper > 1:
        # Половина Hann на правом краю
        hann = 0.5 * (1.0 - np.cos(np.pi * np.arange(n_taper) / n_taper))
        w[-n_taper:] = hann[::-1]
    return w


def octave_band_centers(
    f_start: float,
    f_end: float,
    fraction: float = 1 / 12,
) -> np.ndarray:
    """Центральные частоты дольно-октавных полос от f_start до f_end.

    Стандарт: 1/3 октавы — крупные полосы (психоакустически), 1/12 — плавно,
    1/24 — почти непрерывно. Для аудио обычно 1/12 или 1/24.
    """
    if f_start <= 0 or f_end <= f_start:
        raise ValueError("Некорректные f_start / f_end.")
    n_octaves = np.log2(f_end / f_start)
    n_bands = int(np.ceil(n_octaves / fraction)) + 1
    return f_start * 2.0 ** (np.arange(n_bands) * fraction)


@dataclass
class FreqDependentResult:
    """Результат частотно-зависимого анализа АЧХ."""

    freqs: np.ndarray              # центральные частоты полос, Гц
    magnitude_db: np.ndarray       # амплитуда в дБ
    window_lengths_s: np.ndarray   # длина окна, использованного на каждой f
    n_cycles_actual: np.ndarray    # реально достигнутое число циклов:
                                   #   = window_lengths_s * freqs.
                                   # Если < n_cycles_target → НЧ недостоверны.
    n_cycles_target: float         # сколько циклов хотели


def frequency_dependent_response(
    ir: np.ndarray,
    fs: int,
    direct_idx: int,
    first_reflection_idx: int,
    f_start: float = 20.0,
    f_end: float | None = None,
    n_cycles: float = 6.0,
    band_fraction: float = 1 / 12,
    taper_frac: float = 0.25,
) -> FreqDependentResult:
    """Получить квази-безэховую АЧХ с частотно-зависимым окном.

    Алгоритм:
        1. Определяем максимальную длину окна:  T_win = (refl − direct) / fs.
        2. Строим список полос по 1/band_fraction октавы от f_start до f_end.
        3. Для каждой центральной частоты f:
            а) Желаемая длина окна:  T_des(f) = n_cycles / f.
            б) Реальная длина:        T(f) = min(T_des(f), T_win).
            в) Берём ir[direct_idx : direct_idx + T(f)·fs], домножаем на
               cosine taper, делаем FFT.
            г) Усредняем |FFT| в полосе [f·2^(−frac/2), f·2^(+frac/2)],
               получаем magnitude_db(f).
        4. Возвращаем результат.

    Args:
        ir:                  импульсная характеристика.
        fs:                  частота дискретизации.
        direct_idx:          индекс прямого звука.
        first_reflection_idx: индекс первого отражения.
        f_start:             нижняя анализируемая частота, Гц.
        f_end:               верхняя анализируемая частота, Гц (None = fs/2).
        n_cycles:            число циклов сигнала в окне.
        band_fraction:       доля октавы между соседними точками анализа.
        taper_frac:          доля Tukey-окна, занятая косинусным спадом.

    Returns:
        FreqDependentResult.
    """
    if f_end is None:
        f_end = fs / 2.0 * 0.95

    T_win_max = (first_reflection_idx - direct_idx) / fs
    if T_win_max <= 0:
        raise ValueError("first_reflection_idx должен быть больше direct_idx.")

    freqs = octave_band_centers(f_start, f_end, band_fraction)
    mag_db = np.empty_like(freqs)
    win_lens_s = np.empty_like(freqs)
    cycles_actual = np.empty_like(freqs)

    for i, f in enumerate(freqs):
        # Желаемая длина окна
        T_des = n_cycles / f
        T_use = min(T_des, T_win_max)
        n_win = max(8, int(round(T_use * fs)))

        # Берём кусок IR с прямого звука
        segment = ir[direct_idx : direct_idx + n_win]
        if len(segment) < 8:
            mag_db[i] = -np.inf
            win_lens_s[i] = 0.0
            cycles_actual[i] = 0.0
            continue

        # Tukey-окно: плоский старт + плавный спад
        w = cosine_taper(len(segment), taper_frac=taper_frac)
        windowed = segment * w

        # FFT с zero-padding для лучшего частотного разрешения
        nfft = max(8192, 2 ** int(np.ceil(np.log2(len(windowed) * 4))))
        H = np.fft.rfft(windowed, n=nfft)
        fft_freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)

        # Усредняем |H| в полосе [f·2^(−frac/2), f·2^(+frac/2)] (в линейной шкале)
        f_lo = f * 2.0 ** (-band_fraction / 2.0)
        f_hi = f * 2.0 ** (+band_fraction / 2.0)
        mask = (fft_freqs >= f_lo) & (fft_freqs <= f_hi)
        if mask.any():
            mag_lin = np.mean(np.abs(H[mask]))
        else:
            # Если полоса уже единичного бина — берём ближайший
            nearest = int(np.argmin(np.abs(fft_freqs - f)))
            mag_lin = np.abs(H[nearest])

        mag_db[i] = 20.0 * np.log10(mag_lin + 1e-30)
        win_lens_s[i] = T_use
        cycles_actual[i] = T_use * f

    return FreqDependentResult(
        freqs=freqs,
        magnitude_db=mag_db,
        window_lengths_s=win_lens_s,
        n_cycles_actual=cycles_actual,
        n_cycles_target=n_cycles,
    )


def fixed_window_response(
    ir: np.ndarray,
    fs: int,
    direct_idx: int,
    first_reflection_idx: int,
    nfft: int | None = None,
    taper_frac: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """Классическое окно фиксированной длины — для сравнения.

    Берём IR от direct_idx до first_reflection_idx, накладываем Tukey-окно,
    делаем FFT.

    Returns:
        (freqs, magnitude_db) — частотная сетка и уровни в дБ.
    """
    n_win = first_reflection_idx - direct_idx
    if n_win < 8:
        raise ValueError("Окно слишком короткое.")

    segment = ir[direct_idx : direct_idx + n_win]
    w = cosine_taper(len(segment), taper_frac=taper_frac)
    windowed = segment * w

    if nfft is None:
        nfft = max(8192, 2 ** int(np.ceil(np.log2(n_win * 4))))

    H = np.fft.rfft(windowed, n=nfft)
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    mag_db = 20.0 * np.log10(np.abs(H) + 1e-30)

    return freqs, mag_db
