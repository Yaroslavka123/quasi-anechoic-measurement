"""Тесты автодетекта t₀ и первого отражения."""

from __future__ import annotations

import numpy as np
import pytest

from room_acoustics.detection import (
    auto_detect,
    find_direct_sound,
    find_first_reflection,
    hilbert_envelope,
)


FS = 48000


def make_ir(
    fs: int = FS,
    direct_idx: int = 144,        # 3 мс @ 48 кГц
    direct_amp: float = 1.0,
    reflections: list[tuple[int, float]] | None = None,
    rt60: float = 0.4,
    length_s: float = 0.5,
    noise_amp: float = 1e-4,
    seed: int = 0,
) -> np.ndarray:
    """Синтетическая IR с заданными параметрами."""
    n = int(length_s * fs)
    ir = np.zeros(n)
    ir[direct_idx] = direct_amp
    if reflections:
        for idx, amp in reflections:
            if idx < n:
                ir[idx] = amp
    # Реверберационный хвост
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    decay = np.exp(-3.0 * np.log(10) * t / rt60)
    tail = rng.standard_normal(n) * 0.02 * decay
    tail[:direct_idx] = 0
    ir = ir + tail
    # Фоновый шум (равномерно по всему сигналу — имитация шумового пола записи)
    ir = ir + rng.standard_normal(n) * noise_amp
    return ir


def test_find_direct_sound_basic():
    """Прямой звук должен находиться с точностью до 1 отсчёта."""
    expected = 200
    ir = make_ir(direct_idx=expected, reflections=[(500, 0.4)])
    idx, sub = find_direct_sound(ir)
    assert abs(idx - expected) <= 1, f"Найден прямой звук на {idx}, ожидали {expected}"
    # Sub-sample должен быть около expected (в пределах 1 отсчёта)
    assert abs(sub - expected) < 1.0


def test_find_direct_sound_subsample_precision():
    """Параболическая интерполяция: пик «между» отсчётами должен находиться точнее.

    Создадим сигнал с пиком, размытым по нескольким отсчётам — параболическая
    интерполяция должна дать дробное значение, ближе к центру массы.
    """
    n = 1000
    ir = np.zeros(n)
    # Размажем «пик» по 3 отсчётам с амплитудами 0.5, 1.0, 0.5
    ir[499] = 0.5
    ir[500] = 1.0
    ir[501] = 0.5
    idx, sub = find_direct_sound(ir)
    assert idx == 500
    # Параболическая интерполяция для симметричного пика даёт ровно 500.0
    assert abs(sub - 500.0) < 0.01


def test_find_first_reflection_simple():
    """Простая комната: одно отражение на 8 мс. Должно найтись."""
    direct = 144  # 3 мс
    refl = 384    # 8 мс
    ir = make_ir(direct_idx=direct, reflections=[(refl, 0.5)])
    found = find_first_reflection(ir, FS, direct_idx=direct)
    assert found is not None
    # Допускаем небольшое расхождение из-за сглаживания
    assert abs(found - refl) < int(0.5e-3 * FS), (
        f"Отражение найдено на {found}, ожидали {refl} (±{int(0.5e-3*FS)})"
    )


def test_find_first_reflection_multiple():
    """Несколько отражений: должно найтись ПЕРВОЕ."""
    direct = 144
    refl1 = 384   # 8 мс — это первое
    refl2 = 720   # 15 мс
    refl3 = 1200  # 25 мс
    ir = make_ir(
        direct_idx=direct,
        reflections=[(refl1, 0.5), (refl2, 0.3), (refl3, 0.2)],
    )
    found = find_first_reflection(ir, FS, direct_idx=direct)
    assert found is not None
    assert abs(found - refl1) < int(0.5e-3 * FS)


def test_find_first_reflection_below_threshold():
    """Если отражения слишком тихие, должны не находиться."""
    direct = 144
    # −40 дБ относительно прямого = 0.01
    ir = make_ir(direct_idx=direct, reflections=[(400, 0.01)], rt60=0.1)
    found = find_first_reflection(
        ir, FS, direct_idx=direct, threshold_db=-15.0
    )
    # Может найти либо что-то от реверберационного хвоста, либо None
    # Главное — точно не должно найти на позиции 400 как «отражение»
    if found is not None:
        assert abs(found - 400) > 50  # не позиция искусственного отражения


def test_hilbert_envelope_is_smooth():
    """Огибающая должна быть «гладкой» (положительной, без осцилляций)."""
    fs = FS
    t = np.arange(fs) / fs
    # АМ-сигнал: 100 Гц несущая с медленной модуляцией
    sig = np.sin(2 * np.pi * 100 * t) * (1 + 0.3 * np.sin(2 * np.pi * 5 * t))
    env = hilbert_envelope(sig)
    assert np.all(env >= 0), "Огибающая должна быть неотрицательной"
    # Огибающая должна быть гораздо менее «дёрганой», чем сам сигнал
    sig_zero_crossings = np.sum(np.diff(np.sign(sig)) != 0)
    env_zero_crossings = np.sum(np.diff(np.sign(env - np.mean(env))) != 0)
    assert env_zero_crossings < sig_zero_crossings


def test_auto_detect_pipeline():
    """Полный pipeline auto_detect должен дать корректные значения."""
    direct = 144  # 3 мс
    refl = 384    # 8 мс → окно = 5 мс
    ir = make_ir(direct_idx=direct, reflections=[(refl, 0.5)])

    result = auto_detect(ir, FS, n_cycles=6)

    assert abs(result.direct_idx - direct) <= 1
    assert abs(result.first_reflection_idx - refl) < int(0.5e-3 * FS)
    # Длина окна ~5 мс
    assert 0.004 < result.window_length_s < 0.006
    # f_min_full = 6 / 0.005 = 1200 Гц
    assert 1000 < result.f_min_full < 1500
    # Шумовой пол должен быть «тихим» (хорошо ниже −40 дБ)
    assert result.floor_db < -30.0


def test_auto_detect_no_reflection_found():
    """Если отражений нет, first_reflection_idx устанавливается в конец IR."""
    direct = 144
    ir = make_ir(direct_idx=direct, reflections=[], rt60=0.3, noise_amp=1e-5)
    result = auto_detect(ir, FS, threshold_db=-15.0)
    # Если find_first_reflection вернул None, должен подставиться len(ir) - 1
    # Но может и найти что-то на хвосте — оба варианта приемлемы
    assert result.first_reflection_idx >= direct
    assert result.window_length_samples > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
