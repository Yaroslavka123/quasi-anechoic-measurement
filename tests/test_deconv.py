"""Тесты для деконволюции на синтетических IR.

Проверяем, что весь pipeline (generate_ess → convolve with synthetic IR
→ deconvolve) корректно восстанавливает заданную импульсную характеристику.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import fftconvolve

from room_acoustics.deconv import (
    deconvolve_freq,
    deconvolve_time,
    normalize_ir,
    trim_ir,
)
from room_acoustics.sweep import generate_ess, inverse_filter_ess


FS = 48000
F_START = 20.0
F_END = 20000.0
DURATION = 3.0


def synthetic_room_ir(
    fs: int = FS,
    direct_delay_ms: float = 3.0,
    reflections_ms: tuple[float, ...] = (8.0, 13.5, 22.0, 35.0),
    reflection_amps: tuple[float, ...] = (0.5, 0.3, 0.2, 0.1),
    rt60: float = 0.4,
    length_s: float = 1.0,
) -> tuple[np.ndarray, int]:
    """Синтетическая IR: прямой звук + 4 явных отражения + экспоненциальный хвост.

    Returns:
        (ir, direct_idx) — IR и индекс прямого звука в массиве.
    """
    n = int(length_s * fs)
    ir = np.zeros(n)

    direct_idx = int(direct_delay_ms * 1e-3 * fs)
    ir[direct_idx] = 1.0

    for t_ms, amp in zip(reflections_ms, reflection_amps, strict=True):
        idx = int(t_ms * 1e-3 * fs)
        if idx < n:
            ir[idx] = amp

    # Экспоненциально спадающий шумовой хвост (имитация реверберации)
    rng = np.random.default_rng(seed=42)
    noise = rng.standard_normal(n) * 0.03
    t = np.arange(n) / fs
    decay = np.exp(-3.0 * np.log(10) * t / rt60)  # RT60 = время падения на 60 дБ
    noise *= decay
    # Хвост начинаем после прямого звука
    noise[:direct_idx] = 0
    ir = ir + noise

    return ir, direct_idx


def test_pipeline_recovers_direct_sound():
    """Полный pipeline должен корректно показывать прямой звук в IR."""
    s = generate_ess(F_START, F_END, DURATION, FS)
    g = inverse_filter_ess(F_START, F_END, DURATION, FS)

    ir_true, direct_idx = synthetic_room_ir(fs=FS)

    # Имитация записи: свёртка свипа с IR помещения
    recorded = fftconvolve(s, ir_true, mode="full")

    # Деконволюция
    ir_full = deconvolve_time(recorded, g)
    ir, offset = trim_ir(ir_full, sweep_length=len(s), pre_pad=1024)

    # Главный пик в IR должен быть в районе offset + direct_idx
    peak_idx = int(np.argmax(np.abs(ir)))
    expected_peak = offset + direct_idx

    assert abs(peak_idx - expected_peak) < 10, (
        f"Прямой звук на {peak_idx}, ожидали {expected_peak}"
    )


def test_pipeline_recovers_first_reflection():
    """Первое отражение тоже должно быть видно в восстановленной IR."""
    s = generate_ess(F_START, F_END, DURATION, FS)
    g = inverse_filter_ess(F_START, F_END, DURATION, FS)

    direct_ms = 3.0
    refl_ms = (8.0,)
    refl_amp = (0.5,)

    ir_true, _ = synthetic_room_ir(
        fs=FS,
        direct_delay_ms=direct_ms,
        reflections_ms=refl_ms,
        reflection_amps=refl_amp,
        rt60=0.3,
    )

    recorded = fftconvolve(s, ir_true, mode="full")
    ir_full = deconvolve_time(recorded, g)
    ir, offset = trim_ir(ir_full, sweep_length=len(s), pre_pad=1024)

    # Нормируем по пику прямого звука, чтобы амплитуды стали относительными
    ir_norm = normalize_ir(ir, target_peak=1.0)

    # Найдём прямой звук
    direct_idx_in_ir = int(np.argmax(np.abs(ir_norm)))
    # Где должно быть отражение
    refl_offset_samples = int((refl_ms[0] - direct_ms) * 1e-3 * FS)
    expected_refl_idx = direct_idx_in_ir + refl_offset_samples

    # Смотрим окно вокруг ожидаемого отражения и проверяем, что там есть пик
    window = 50
    region = ir_norm[expected_refl_idx - window : expected_refl_idx + window]
    local_peak = np.max(np.abs(region))

    # Локальный пик должен быть близко к 0.5 (амплитуда отражения)
    # Допуск ±30% (учитываем затухание, шум, конечную длину свипа)
    assert 0.3 < local_peak < 0.8, f"Амплитуда отражения {local_peak:.3f}, ожидали ~0.5"


def test_deconv_freq_method_matches():
    """Метод деконволюции в частотной области должен давать похожий результат."""
    s = generate_ess(F_START, F_END, DURATION, FS)

    ir_true, direct_idx = synthetic_room_ir(fs=FS)

    recorded = fftconvolve(s, ir_true, mode="full")

    # Частотный метод — на коротком отрезке (длина = len(recorded))
    ir_freq = deconvolve_freq(recorded, s, reg=1e-3)
    ir_freq_norm = normalize_ir(ir_freq)

    # Должен быть отчётливый пик в районе direct_idx
    peak_idx = int(np.argmax(np.abs(ir_freq_norm)))
    assert abs(peak_idx - direct_idx) < 50, (
        f"Частотный метод: пик на {peak_idx}, ожидали ~{direct_idx}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
