"""Генерация логарифмического синус-свипа (ESS) и инверсного фильтра.

Метод Farina (Farina, A. *Simultaneous measurement of impulse response and distortion
with a swept-sine technique*. AES 108, 2000) — стандарт de-facto в акустических измерениях.

Идея:
    s(t) = sin( φ(t) ),  где φ(t) такая, что мгновенная частота растёт экспоненциально
    от f1 до f2 за время T.

    Инверсный фильтр g(t) строится так, что свёртка записи y(t) = (s * h)(t) с g(t)
    даёт оценку импульсной характеристики:  y * g  ≈  h.

    Самое удобное свойство ESS: гармонические искажения системы во время свёртки
    «уезжают» в отрицательное время и легко отделяются от полезной IR.
"""

from __future__ import annotations

import numpy as np


def generate_ess(
    f_start: float,
    f_end: float,
    duration: float,
    fs: int,
    fade_in_ms: float = 10.0,
    fade_out_ms: float = 50.0,
) -> np.ndarray:
    """Сгенерировать exponential sine sweep.

    Мгновенная фаза:
        φ(t) = (2π·f1·T / L) · (exp(L·t/T) − 1)
    где
        L = ln(f2/f1)  — кол-во неперов между f1 и f2
        T = duration

    Производная даёт мгновенную частоту:
        f(t) = (1/2π) · dφ/dt = f1 · exp(L·t/T)
    т.е. экспоненциальный рост частоты от f1 при t=0 до f2 при t=T.

    Args:
        f_start: начальная частота, Гц (обычно 20).
        f_end:   конечная частота, Гц (обычно fs/2 - small_margin).
        duration: длительность свипа, секунды.
        fs:      частота дискретизации, Гц.
        fade_in_ms:  длительность плавного нарастания (Hann-half), мс.
                     Защищает АС от щелчка в начале и сглаживает DC.
        fade_out_ms: длительность плавного спадания, мс.
                     Подавляет «звон» от резкого обрыва.

    Returns:
        s: numpy.ndarray формы (N,), N = int(duration * fs), нормирован в [−1, +1].

    Raises:
        ValueError: если параметры физически некорректны.
    """
    if f_start <= 0 or f_end <= 0:
        raise ValueError("Частоты должны быть положительные.")
    if f_end <= f_start:
        raise ValueError("f_end должно быть больше f_start.")
    if f_end > fs / 2:
        raise ValueError(f"f_end={f_end} превышает Найквиста {fs/2}. Будет алиасинг.")
    if duration <= 0:
        raise ValueError("duration должно быть положительным.")

    n_samples = int(round(duration * fs))
    t = np.arange(n_samples) / fs
    T = n_samples / fs  # фактическая длительность с учётом округления

    L = np.log(f_end / f_start)
    # Фаза: φ(t) = 2π·f1·T/L · (exp(L·t/T) − 1)
    phi = 2.0 * np.pi * f_start * T / L * (np.exp(L * t / T) - 1.0)
    s = np.sin(phi)

    # Fade-in / fade-out half-Hann окнами для устранения щелчков
    n_in = int(round(fade_in_ms * 1e-3 * fs))
    n_out = int(round(fade_out_ms * 1e-3 * fs))
    if n_in > 0:
        w = 0.5 * (1.0 - np.cos(np.pi * np.arange(n_in) / n_in))
        s[:n_in] *= w
    if n_out > 0:
        w = 0.5 * (1.0 - np.cos(np.pi * np.arange(n_out) / n_out))
        s[-n_out:] *= w[::-1]

    return s.astype(np.float64)


def inverse_filter_ess(
    f_start: float,
    f_end: float,
    duration: float,
    fs: int,
    fade_in_ms: float = 10.0,
    fade_out_ms: float = 50.0,
) -> np.ndarray:
    """Сгенерировать инверсный фильтр для деконволюции ESS.

    Инверсный фильтр получается как:
        g(t) = s(T − t) · A(t)

    где A(t) — экспоненциальная амплитудная модуляция, компенсирующая «розовый»
    спектр ESS (амплитуда спектра ∝ 1/√f). После умножения на A(t) спектр
    инверсного фильтра становится ∝ √f, и свёртка s * g даёт почти белый импульс.

    A(t) = (f1/f2) ^ (t/T) ·  K
        =  exp(−L·t/T)  ·  K

    Это означает: в начале g(t) (где после реверса находятся ВЫСОКИЕ частоты)
    амплитуда максимальна, а к концу — минимальна. Это правильно: ESS имеет
    больше энергии на низких частотах (они длятся дольше), поэтому инверсный
    фильтр должен иметь меньше энергии на низких — он должен их «приглушать».

    Нормировка K выбрана так, что свёртка s * g в идеальной системе даёт
    единичный пик: max|conv| ≈ 1.

    Args:
        f_start, f_end, duration, fs, fade_in_ms, fade_out_ms:
            ДОЛЖНЫ совпадать с параметрами, переданными в generate_ess().

    Returns:
        g: numpy.ndarray, инверсный фильтр.
    """
    s = generate_ess(f_start, f_end, duration, fs, fade_in_ms, fade_out_ms)
    n = len(s)
    T = n / fs
    t = np.arange(n) / fs
    L = np.log(f_end / f_start)

    # Время-реверс
    s_rev = s[::-1]

    # Амплитудная коррекция: экспоненциально спадает от 1 до f1/f2
    # При t=0 (это конец оригинального свипа = ВЧ) множитель = 1.
    # При t=T (это начало оригинального свипа = НЧ) множитель = f1/f2.
    # Это компенсирует +3 дБ/окт спадение спектра ESS.
    env = np.exp(-L * t / T)

    g = s_rev * env

    # Нормировка K: подбираем так, чтобы свёртка s * g давала пик ≈ 1.
    # Для ESS с длиной T справедлива оценка K ≈ 2·L/T (выводится через
    # параметрическое интегрирование, см. Müller & Massarani 2001).
    K = 2.0 * L / T
    g *= K

    return g.astype(np.float64)


def expected_ir_delay(duration: float) -> float:
    """В какой момент времени после свёртки записи с инверсным фильтром
    ожидается главный пик IR.

    Поскольку s(t) (длина T) свёртывается с g(t) (длина T), результат имеет
    длину ~2T, и центр (точка δ-образного пика) находится на ~T от начала.

    Это полезно для выравнивания: позиция пика в свёртке `record * g` будет
    `T + acoustic_delay + system_latency`.

    Returns:
        Время в секундах, в которое окажется главный пик для идеальной системы
        без задержки.
    """
    return duration
