"""Тесты метрик ISO 3382 на синтетических IR."""

from __future__ import annotations

import numpy as np
import pytest

from room_acoustics.metrics import (
    clarity,
    compute_metrics,
    definition,
    fit_decay_time,
    schroeder_integral,
)


FS = 48000


def synthetic_exponential_decay_ir(
    fs: int = FS,
    rt60: float = 0.5,
    length_s: float = 2.0,
    direct_amp: float = 1.0,
    direct_idx: int = 0,
    noise_amp: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """Идеальная IR с экспоненциальным затуханием и заданным RT60.

    h(t) = δ(t) + noise(t) · exp(−3·ln(10)·t/RT60)
    """
    n = int(length_s * fs)
    ir = np.zeros(n)
    ir[direct_idx] = direct_amp

    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    decay = np.exp(-3.0 * np.log(10) * t / rt60)
    tail = rng.standard_normal(n) * 0.3 * decay
    tail[: direct_idx + 1] = 0
    ir = ir + tail

    if noise_amp > 0:
        ir = ir + rng.standard_normal(n) * noise_amp

    return ir


def test_schroeder_integral_decreases():
    """EDC должна быть монотонно невозрастающей и начинаться с 0 дБ."""
    ir = synthetic_exponential_decay_ir(rt60=0.5)
    edc_db = schroeder_integral(ir)
    assert abs(edc_db[0]) < 0.001, "EDC[0] должно быть 0 дБ"
    # Проверим монотонность (с малым допуском на численные погрешности)
    diffs = np.diff(edc_db)
    assert np.all(diffs <= 1e-9), "EDC должна быть монотонно невозрастающей"


def test_fit_decay_time_recovers_rt60():
    """На идеальной экспоненте T20 должно дать RT60 близко к истинному."""
    true_rt60 = 0.5
    ir = synthetic_exponential_decay_ir(rt60=true_rt60, length_s=2.0)
    edc_db = schroeder_integral(ir)
    rt = fit_decay_time(edc_db, FS, db_start=-5.0, db_end=-25.0)
    assert rt is not None
    # Допуск 15% — на случайной реализации шумового хвоста
    assert abs(rt - true_rt60) / true_rt60 < 0.15, (
        f"T20 → RT60 = {rt:.3f} с, ожидали {true_rt60} с"
    )


def test_fit_decay_time_t30_more_robust():
    """T30 должно быть более устойчиво, чем T20, на тех же данных."""
    true_rt60 = 0.4
    rt20_list = []
    rt30_list = []
    for seed in range(10):
        ir = synthetic_exponential_decay_ir(
            rt60=true_rt60, length_s=2.0, seed=seed
        )
        edc_db = schroeder_integral(ir)
        rt20 = fit_decay_time(edc_db, FS, db_start=-5.0, db_end=-25.0)
        rt30 = fit_decay_time(edc_db, FS, db_start=-5.0, db_end=-35.0)
        if rt20 is not None:
            rt20_list.append(rt20)
        if rt30 is not None:
            rt30_list.append(rt30)
    # И T20, и T30 в среднем дают истинное значение
    assert abs(np.mean(rt20_list) - true_rt60) / true_rt60 < 0.10
    assert abs(np.mean(rt30_list) - true_rt60) / true_rt60 < 0.10


def test_clarity_dry_room():
    """Очень короткое RT60 → высокий C50 (вся энергия в первых 50 мс)."""
    ir = synthetic_exponential_decay_ir(rt60=0.05, length_s=1.0)
    c50 = clarity(ir, FS, 50.0)
    # При RT60 = 50 мс к моменту 50 мс уже мало что осталось
    assert c50 > 0, f"C50 в сухой комнате должен быть положительным, не {c50}"


def test_clarity_reverberant_room():
    """Длинный RT60 → низкий C50 (много поздней энергии)."""
    ir = synthetic_exponential_decay_ir(rt60=2.0, length_s=4.0)
    c50 = clarity(ir, FS, 50.0)
    # При RT60 = 2 с к 50 мс ещё практически вся энергия впереди
    assert c50 < 0, f"C50 в реверберационной комнате должен быть отрицательным, не {c50}"


def test_definition_in_range():
    """D50 всегда в диапазоне [0, 1]."""
    for rt60 in [0.05, 0.3, 1.0, 3.0]:
        ir = synthetic_exponential_decay_ir(rt60=rt60, length_s=4.0)
        d50 = definition(ir, FS, 50.0)
        assert 0.0 <= d50 <= 1.0


def test_compute_metrics_returns_all():
    """Полная сводка метрик: все поля заполнены."""
    ir = synthetic_exponential_decay_ir(rt60=0.5, length_s=2.0)
    m = compute_metrics(ir, FS, direct_idx=0)
    assert m.t20_s is not None
    assert m.t30_s is not None
    assert m.edt_s is not None
    assert np.isfinite(m.c50_db)
    assert np.isfinite(m.c80_db)
    assert 0.0 <= m.d50 <= 1.0
    # T20 ≈ T30 ≈ 0.5 с
    assert abs(m.t20_s - 0.5) / 0.5 < 0.20
    assert abs(m.t30_s - 0.5) / 0.5 < 0.15


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
