"""Тесты для генерации ESS и инверсного фильтра."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import fftconvolve

from room_acoustics.sweep import generate_ess, inverse_filter_ess


FS = 48000


def test_ess_basic_properties():
    """Базовые свойства сгенерированного свипа."""
    s = generate_ess(f_start=20, f_end=20000, duration=2.0, fs=FS,
                     fade_in_ms=10, fade_out_ms=50)
    # Длина совпадает с (duration * fs)
    assert len(s) == int(2.0 * FS)
    # Амплитуда не превышает 1 (учитывая что синус +- 1, fade-окна вниз)
    assert np.max(np.abs(s)) <= 1.0
    # Начало и конец близки к 0 благодаря fade-окнам
    assert abs(s[0]) < 1e-6
    assert abs(s[-1]) < 0.05


def test_ess_instantaneous_frequency():
    """Мгновенная частота свипа должна расти экспоненциально от f1 до f2."""
    f1, f2, T = 50.0, 5000.0, 2.0
    s = generate_ess(f_start=f1, f_end=f2, duration=T, fs=FS,
                     fade_in_ms=0, fade_out_ms=0)

    # Мгновенная частота через unwrap фазы аналитического сигнала
    from scipy.signal import hilbert
    analytic = hilbert(s)
    phase = np.unwrap(np.angle(analytic))
    inst_freq = np.diff(phase) * FS / (2 * np.pi)

    # Проверим в нескольких точках (отступив от краёв, где Гильберт врёт)
    margin = int(0.1 * len(s))
    # diff укорачивает массив на 1, выравниваем
    t = np.arange(len(s) - 1) / FS
    L = np.log(f2 / f1)
    f_theory = f1 * np.exp(L * t / T)

    # Допуск 10% — Гильберт-преобразование не идеально точное на нестационарном сигнале
    inst_freq_at = inst_freq[margin:-margin]
    f_theory_at = f_theory[margin:-margin]
    rel_err = np.abs(inst_freq_at - f_theory_at) / f_theory_at
    assert np.median(rel_err) < 0.1, f"Медианная ошибка частоты {np.median(rel_err):.2%}"


def test_inverse_filter_yields_impulse():
    """Свёртка ESS с инверсным фильтром должна давать импульсо-подобный сигнал.

    Это основное свойство, гарантирующее, что деконволюция работает.
    """
    f1, f2, T = 20.0, 20000.0, 3.0
    s = generate_ess(f_start=f1, f_end=f2, duration=T, fs=FS,
                     fade_in_ms=10, fade_out_ms=50)
    g = inverse_filter_ess(f_start=f1, f_end=f2, duration=T, fs=FS,
                           fade_in_ms=10, fade_out_ms=50)

    # Свёртка
    conv = fftconvolve(s, g, mode="full")

    # Где должен быть пик: примерно на отсчёте N (= длина свипа)
    expected_peak_idx = len(s) - 1  # full-mode свёртка центрирует на n_s + n_g - 1, центр = n_s - 1
    peak_idx = int(np.argmax(np.abs(conv)))

    # Допускаем сдвиг в несколько сэмплов из-за дискретизации
    assert abs(peak_idx - expected_peak_idx) < 100, (
        f"Пик на {peak_idx}, ожидали {expected_peak_idx}"
    )

    # Side-lobes должны быть существенно ниже пика (хотя бы −20 дБ)
    peak_val = np.abs(conv[peak_idx])
    # Маскируем окрестность пика и проверяем максимум вне
    mask = np.ones_like(conv, dtype=bool)
    mask[peak_idx - 200 : peak_idx + 200] = False
    sidelobe = np.max(np.abs(conv[mask]))
    sidelobe_db = 20 * np.log10(sidelobe / peak_val)
    assert sidelobe_db < -20, f"Sidelobes слишком высокие: {sidelobe_db:.1f} дБ"


def test_inverse_filter_recovers_delay():
    """Если в систему добавить чистую задержку, IR должна показать её правильно."""
    f1, f2, T = 50.0, 10000.0, 2.0
    s = generate_ess(f_start=f1, f_end=f2, duration=T, fs=FS,
                     fade_in_ms=10, fade_out_ms=50)
    g = inverse_filter_ess(f_start=f1, f_end=f2, duration=T, fs=FS,
                           fade_in_ms=10, fade_out_ms=50)

    delay_samples = 1234
    # Имитация записи: добавляем задержку
    y = np.concatenate([np.zeros(delay_samples), s])

    ir_full = fftconvolve(y, g, mode="full")
    peak_idx = int(np.argmax(np.abs(ir_full)))

    # Пик должен сместиться на delay_samples относительно случая без задержки
    expected_peak = (len(s) - 1) + delay_samples
    assert abs(peak_idx - expected_peak) < 50, (
        f"Пик на {peak_idx}, ожидали {expected_peak}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
