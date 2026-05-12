"""Тесты модуля I/O — без реального аудиоустройства.

Тестируем только функцию detect_system_latency_loopback на синтетике.
Запись через sounddevice не тестируется в pytest (требует устройства).
"""

from __future__ import annotations

import numpy as np

from room_acoustics.io import detect_system_latency_loopback
from room_acoustics.sweep import generate_ess


FS = 48000


def test_loopback_latency_zero():
    """Если loopback идентичен played — задержка 0."""
    played = generate_ess(20, 20000, 1.0, FS)
    lag = detect_system_latency_loopback(played, played, FS)
    assert lag == 0


def test_loopback_latency_delayed():
    """Если loopback задержан на 200 отсчётов — должны это обнаружить."""
    played = generate_ess(20, 20000, 1.0, FS)
    delay = 200
    loopback = np.concatenate([np.zeros(delay), played])[: len(played)]
    lag = detect_system_latency_loopback(played, loopback, FS)
    # Точность ±1 отсчёт
    assert abs(lag - delay) <= 1, f"Найдена задержка {lag}, ожидали {delay}"


def test_loopback_latency_noisy():
    """С шумом результат должен оставаться разумным."""
    rng = np.random.default_rng(7)
    played = generate_ess(20, 20000, 1.0, FS)
    delay = 1500
    loopback = np.concatenate([np.zeros(delay), played])[: len(played)]
    # Добавим шум, но не слишком много
    loopback = loopback + rng.standard_normal(len(loopback)) * 0.05
    lag = detect_system_latency_loopback(played, loopback, FS)
    # Точность ±3 отсчёта в умеренном шуме
    assert abs(lag - delay) <= 3
