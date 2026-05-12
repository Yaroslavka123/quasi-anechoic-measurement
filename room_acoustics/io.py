"""Воспроизведение и одновременная запись через sounddevice.

Поддерживает:
    • Однонаправленную запись после воспроизведения (если loopback не нужен).
    • Двухканальную запись с loopback на отдельном канале (рекомендуется).
      Подключаешь TS-кабелем линейный выход аудиоинтерфейса на инструментальный
      вход — получаешь идеальный электрический референс, по которому можно
      точно вычислить задержку системы (D/A → A/D + кабель + Focusrite-буфер).

Windows + Focusrite Scarlett:
    sounddevice работает через PortAudio. Под Windows есть несколько API:
        WASAPI (по умолчанию)  — низкая задержка, хорошо для измерений
        WDM-KS, MME, DirectSound — старые, не рекомендуется
        ASIO — нужен ASIO4ALL или родной Focusrite-драйвер
    Если WASAPI не работает — попробуй sounddevice.query_devices() и явно
    укажи имя.

ВАЖНО: не запускай measure() в Jupyter-ноутбуке без чёткого контроля —
       при KeyboardInterrupt sounddevice не всегда корректно закрывает поток.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MeasurementResult:
    """Результат записи через sounddevice."""

    recorded: np.ndarray   # форма (n_samples,) или (n_samples, n_channels)
    fs: int
    input_device: str
    output_device: str


def list_devices() -> str:
    """Печать списка доступных аудиоустройств. Импорт sounddevice ленивый
    (он тащит PortAudio и может падать в окружениях без аудио — например, CI).
    """
    import sounddevice as sd
    return str(sd.query_devices())


def measure(
    signal: np.ndarray,
    fs: int,
    input_device: int | str | None = None,
    output_device: int | str | None = None,
    input_channels: int = 1,
    output_channels: int = 1,
    extra_record_s: float = 2.0,
    blocking: bool = True,
) -> MeasurementResult:
    """Воспроизвести `signal` и одновременно записать с микрофона.

    Args:
        signal:           массив отсчётов для воспроизведения (mono или (N, ch)).
        fs:               частота дискретизации.
        input_device:     индекс или имя входного устройства (None = default).
        output_device:    индекс или имя выходного устройства.
        input_channels:   1 — обычная запись микрофона; 2 — микрофон + loopback.
        output_channels:  1 — моно сигнал; 2 — стерео (для теста двух АС).
        extra_record_s:   на сколько секунд продолжать запись ПОСЛЕ окончания
                          воспроизведения. Нужно, чтобы захватить
                          реверберационный хвост.
        blocking:         True — ждать окончания записи; False — вернуться
                          сразу (используется только в продвинутых сценариях).

    Returns:
        MeasurementResult с записанным сигналом.

    Raises:
        ImportError если sounddevice / PortAudio не доступны.
    """
    import sounddevice as sd

    # Приведём сигнал к нужной форме
    if signal.ndim == 1 and output_channels > 1:
        signal_out = np.tile(signal[:, np.newaxis], (1, output_channels))
    elif signal.ndim == 1:
        signal_out = signal.reshape(-1, 1)
    else:
        signal_out = signal

    n_play = signal_out.shape[0]
    n_extra = int(extra_record_s * fs)
    n_total = n_play + n_extra

    # Запись и воспроизведение синхронно
    # sd.playrec возвращает запись той же длины, что и play (без extra).
    # Для extra_record используем sd.Stream напрямую, или ручную «склейку».
    # Простой путь: расширим сигнал тишиной в конце.
    padded = np.zeros((n_total, signal_out.shape[1]), dtype=np.float32)
    padded[:n_play] = signal_out.astype(np.float32)

    recorded = sd.playrec(
        padded,
        samplerate=fs,
        channels=input_channels,
        device=(input_device, output_device),
        blocking=blocking,
        dtype="float32",
    )

    # Имена устройств для лога
    in_info = sd.query_devices(input_device, kind="input") if input_device is not None else sd.query_devices(kind="input")
    out_info = sd.query_devices(output_device, kind="output") if output_device is not None else sd.query_devices(kind="output")

    return MeasurementResult(
        recorded=np.asarray(recorded),
        fs=fs,
        input_device=in_info["name"] if isinstance(in_info, dict) else str(in_info),
        output_device=out_info["name"] if isinstance(out_info, dict) else str(out_info),
    )


def detect_system_latency_loopback(
    played: np.ndarray,
    loopback_recorded: np.ndarray,
    fs: int,
    max_lag_ms: float = 500.0,
) -> int:
    """Найти задержку системы по loopback-записи через кросс-корреляцию.

    На вход — что играли (played) и что записал loopback-канал
    (loopback_recorded). Возвращает целочисленный сдвиг в отсчётах:
        lag > 0  →  loopback_recorded запаздывает от played на lag отсчётов.

    Используется для точной синхронизации перед деконволюцией.

    Args:
        played:              исходный воспроизведённый сигнал (например, ESS).
        loopback_recorded:   loopback-канал записи.
        fs:                  частота дискретизации.
        max_lag_ms:          максимальная искомая задержка (защита от мусора).

    Returns:
        Сдвиг в отсчётах. Если loopback слабый/нечистый — может быть мусором,
        поэтому всегда проверяй визуально first iteration.
    """
    from scipy.signal import fftconvolve

    max_lag = int(max_lag_ms * 1e-3 * fs)
    # Кросс-корреляция = свёртка с реверснутым сигналом
    # Берём только разумную часть, чтобы не корреллировать на длине файла
    n = min(len(played), len(loopback_recorded))
    a = played[:n]
    b = loopback_recorded[:n]

    corr = fftconvolve(b, a[::-1], mode="full")
    # Центр корреляции (нулевой lag) находится по индексу n-1
    center = n - 1
    # Ищем максимум в окне [center, center + max_lag] (нас интересуют
    # ПОЛОЖИТЕЛЬНЫЕ задержки)
    search_end = min(len(corr), center + max_lag + 1)
    peak_offset = int(np.argmax(np.abs(corr[center:search_end])))
    return peak_offset
