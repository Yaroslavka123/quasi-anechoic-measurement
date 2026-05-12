"""Тесты частотно-зависимого окна."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import fftconvolve

from room_acoustics.deconv import deconvolve_time, normalize_ir, trim_ir
from room_acoustics.detection import auto_detect
from room_acoustics.sweep import generate_ess, inverse_filter_ess
from room_acoustics.windowing import (
    cosine_taper,
    fixed_window_response,
    frequency_dependent_response,
    octave_band_centers,
)


FS = 48000


def test_cosine_taper_properties():
    """Tukey-окно: плоское в начале, плавно убывает к нулю."""
    n = 1000
    w = cosine_taper(n, taper_frac=0.25)
    assert len(w) == n
    # Первые 75% — единицы
    assert np.all(w[: int(0.74 * n)] == 1.0)
    # Конец — близко к нулю
    assert w[-1] < 0.01
    # Монотонно невозрастает на убывающей части
    assert np.all(np.diff(w[-250:]) <= 1e-10)


def test_octave_band_centers():
    """1/3 октавы: 100, 125, 160, 200 Гц..."""
    bands = octave_band_centers(100.0, 800.0, 1.0 / 3.0)
    # Должно быть несколько полос
    assert len(bands) >= 3
    # Отношение соседних = 2^(1/3) ≈ 1.26
    ratios = bands[1:] / bands[:-1]
    np.testing.assert_allclose(ratios, 2.0 ** (1 / 3), rtol=1e-6)


def _synthetic_ir(
    fs: int = FS,
    direct_delay_ms: float = 3.0,
    reflection_ms: float = 8.0,
    reflection_amp: float = 0.5,
    rt60: float = 0.4,
    length_s: float = 0.5,
) -> np.ndarray:
    n = int(length_s * fs)
    ir = np.zeros(n)
    direct_idx = int(direct_delay_ms * 1e-3 * fs)
    refl_idx = int(reflection_ms * 1e-3 * fs)
    ir[direct_idx] = 1.0
    if refl_idx < n:
        ir[refl_idx] = reflection_amp
    rng = np.random.default_rng(42)
    t = np.arange(n) / fs
    decay = np.exp(-3.0 * np.log(10) * t / rt60)
    tail = rng.standard_normal(n) * 0.03 * decay
    tail[:direct_idx] = 0
    ir = ir + tail
    return ir


def test_frequency_dependent_response_basic():
    """Базовый запуск: должен вернуть массивы правильной формы."""
    ir = _synthetic_ir()
    direct = int(3e-3 * FS)
    refl = int(8e-3 * FS)
    result = frequency_dependent_response(
        ir, FS, direct, refl, f_start=100.0, f_end=10000.0, n_cycles=6
    )
    assert len(result.freqs) == len(result.magnitude_db)
    assert len(result.freqs) == len(result.window_lengths_s)
    assert len(result.freqs) == len(result.n_cycles_actual)
    # На ВЧ окно короче, чем T_win — реально достигаемое число циклов = n_cycles
    high_freq_mask = result.freqs > 5000
    np.testing.assert_allclose(
        result.n_cycles_actual[high_freq_mask], 6.0, rtol=0.1
    )
    # На НЧ упираемся в потолок T_win = 5 мс
    # При f=100 Гц нужно 6/100 = 60 мс, имеем 5 мс → cycles_actual = 5e-3 * 100 = 0.5
    low_freq_mask = result.freqs < 200
    assert np.all(result.n_cycles_actual[low_freq_mask] < 1.5)


def test_window_length_decreases_with_frequency():
    """С ростом частоты длина окна должна уменьшаться (или оставаться той же)."""
    ir = _synthetic_ir(reflection_ms=15.0)  # T_win = 12 мс
    direct = int(3e-3 * FS)
    refl = int(15e-3 * FS)
    result = frequency_dependent_response(
        ir, FS, direct, refl, f_start=50.0, f_end=15000.0, n_cycles=6
    )
    # Длины окон должны быть монотонно невозрастающими по частоте
    # (могут быть равными в зоне ограничения T_win_max)
    diffs = np.diff(result.window_lengths_s)
    assert np.all(diffs <= 1e-9), "Длина окна должна убывать с ростом частоты"


def test_full_pipeline_synthetic_anechoic_response():
    """Полный pipeline: ESS → IR с известной АЧХ источника → восстановить АЧХ.

    Используем «источник» с простой АЧХ: ВЧ-резонанс. Проверим, что
    частотно-зависимое окно его правильно показывает.
    """
    # Источник: фильтр с резонансом на 3 кГц
    # Реализуем как тонкий BPF второго порядка
    from scipy.signal import butter, lfilter

    f_res = 3000.0
    Q = 5.0
    bw = f_res / Q
    f_lo = f_res - bw / 2
    f_hi = f_res + bw / 2
    b, a = butter(2, [f_lo / (FS / 2), f_hi / (FS / 2)], btype="band")

    # IR источника = отклик BPF на импульс
    source_ir = lfilter(b, a, np.concatenate([[1.0], np.zeros(2047)]))
    # Сделаем главный пик «прямым звуком» — добавим явный импульс в начале
    # На самом деле filter response уже имеет нужную форму

    # Свёртка ESS с IR источника
    s = generate_ess(20.0, 20000.0, 3.0, FS)
    g = inverse_filter_ess(20.0, 20000.0, 3.0, FS)

    recorded = fftconvolve(s, source_ir, mode="full")
    ir_full = deconvolve_time(recorded, g)
    ir, offset = trim_ir(ir_full, sweep_length=len(s), pre_pad=1024)
    ir = normalize_ir(ir)

    # Прямой звук в IR — это импульс отклика, начинающийся на offset
    detection = auto_detect(ir, FS, search_start=offset - 100, search_end=offset + 500)

    # Возьмём окно от direct до конца IR (т.к. отражений нет — это
    # «безэховое» измерение источника)
    result = frequency_dependent_response(
        ir, FS, detection.direct_idx,
        first_reflection_idx=detection.direct_idx + 2048,  # длина BPF IR
        f_start=200.0,
        f_end=10000.0,
        n_cycles=6,
    )

    # Найдём пик АЧХ — должен быть в районе f_res = 3 кГц
    peak_idx = int(np.argmax(result.magnitude_db))
    peak_freq = result.freqs[peak_idx]
    # Допуск 1/3 октавы (BPF расширяет резонанс)
    assert abs(np.log2(peak_freq / f_res)) < 0.4, (
        f"Пик АЧХ на {peak_freq:.0f} Гц, ожидали {f_res:.0f} Гц"
    )


def test_fixed_window_vs_frequency_dependent():
    """Сравнение фикс. окна и частотно-зависимого: должны давать близкие
    результаты на тех частотах, где fixed window даёт >= n_cycles периодов.

    Фикс. окно «врёт» на низких частотах (где T_win < n_cycles/f), но в
    рабочей зоне частот результаты должны быть близки.
    """
    ir = _synthetic_ir(reflection_ms=20.0)  # T_win = 17 мс
    direct = int(3e-3 * FS)
    refl = int(20e-3 * FS)

    # Фикс. окно
    fixed_freqs, fixed_db = fixed_window_response(ir, FS, direct, refl)

    # Частотно-зависимое
    fd = frequency_dependent_response(
        ir, FS, direct, refl, f_start=100.0, f_end=10000.0,
        n_cycles=6, band_fraction=1 / 6,
    )

    # Интерполируем fixed_db к точкам fd.freqs
    fixed_db_interp = np.interp(fd.freqs, fixed_freqs, fixed_db)

    # На частотах где fd достиг полного количества циклов — должно совпадать в пределах разумного
    full_cycles_mask = fd.n_cycles_actual >= fd.n_cycles_target * 0.95
    if full_cycles_mask.any():
        diff_db = fixed_db_interp[full_cycles_mask] - fd.magnitude_db[full_cycles_mask]
        # Допуск 6 дБ — окна разной длины дают разную нормировку FFT
        median_diff = float(np.median(diff_db))
        # Они различаются в основном константой (нормировка), проверим разброс
        spread = float(np.std(diff_db - median_diff))
        assert spread < 5.0, f"Разброс между методами {spread:.2f} дБ слишком велик"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
