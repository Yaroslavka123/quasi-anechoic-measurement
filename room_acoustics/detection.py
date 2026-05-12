"""Автодетект ключевых точек в импульсной характеристике.

Главные точки, которые нужно найти автоматически:
    t₀     — момент прихода прямого звука в IR.
    t_refl — момент прихода первого отражения после прямого звука.

Эти точки определяют:
    1. Привязку IR к нулю по времени (всё, что левее t₀ — задержка системы и
       нелинейные искажения, в IR не относятся).
    2. Длину «чистого» окна для квази-безэхового измерения: T_win = t_refl − t₀.
    3. Максимальное частотно-зависимое окно: T(f) = min(N/f, T_win).

Подход к t₀:
    Используем огибающую Hilbert-аналитического сигнала, ищем максимум,
    уточняем sub-sample параболической интерполяцией.

Подход к t_refl:
    Огибающая → scipy.signal.find_peaks с минимальным расстоянием и порогом
    в дБ относительно прямого звука. Первый найденный пик после t₀ — наш
    кандидат на отражение.

    Минимальный зазор от t₀ нужен, чтобы не цепляться за «звон» прямого
    звука после ESS-деконволюции (он обычно длится 0.3–1 мс из-за конечной
    полосы свипа).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks, hilbert


@dataclass
class DetectionResult:
    """Результат автодетекта точек в IR.

    Все индексы — в отсчётах от начала переданного массива IR.
    Все времена — в секундах относительно t₀ (т.е. t_direct = 0).
    """

    direct_idx: int               # индекс пика прямого звука
    direct_subsample: float       # подвыборочное уточнение позиции, отсчёты
    first_reflection_idx: int     # индекс первого отражения
    window_length_samples: int    # t_refl - t₀ в отсчётах
    window_length_s: float        # то же в секундах
    f_min_full: float             # нижняя частота, которую окно поддерживает
                                  # для N_cycles = 6
    floor_db: float               # уровень шумового пола относительно пика, дБ


def hilbert_envelope(x: np.ndarray) -> np.ndarray:
    """Огибающая через аналитический сигнал Гильберта.

    E(t) = |x(t) + j·H{x}(t)|

    Это даёт «гладкую» амплитудную огибающую, нечувствительную к фазовым
    осцилляциям внутри одного цикла сигнала. Незаменимо для поиска пиков в IR.
    """
    return np.abs(hilbert(x))


def find_direct_sound(
    ir: np.ndarray,
    search_window: tuple[int, int] | None = None,
) -> tuple[int, float]:
    """Найти прямой звук — максимум огибающей.

    Args:
        ir: импульсная характеристика.
        search_window: (start, end) — где искать пик. По умолчанию вся IR.

    Returns:
        (idx, subsample_idx):
            idx — целочисленный индекс пика,
            subsample_idx — уточнение через параболическую интерполяцию
                           по трём соседним отсчётам огибающей.

    Параболическая интерполяция:
        Если максимум огибающей в дискретной точке idx, то реальный пик
        находится в

            δ = 0.5 · (E[idx-1] − E[idx+1]) / (E[idx-1] − 2·E[idx] + E[idx+1])

        и subsample_idx = idx + δ. Это стандартная техника DSP для повышения
        точности локализации пиков до долей отсчёта.
    """
    env = hilbert_envelope(ir)
    if search_window is None:
        start, end = 0, len(env)
    else:
        start, end = search_window
        start = max(0, start)
        end = min(len(env), end)

    region = env[start:end]
    idx_local = int(np.argmax(region))
    idx = start + idx_local

    # Sub-sample уточнение
    if 0 < idx < len(env) - 1:
        e_m = env[idx - 1]
        e_0 = env[idx]
        e_p = env[idx + 1]
        denom = e_m - 2 * e_0 + e_p
        if abs(denom) > 1e-30:
            delta = 0.5 * (e_m - e_p) / denom
        else:
            delta = 0.0
    else:
        delta = 0.0

    return idx, idx + delta


def find_first_reflection(
    ir: np.ndarray,
    fs: int,
    direct_idx: int,
    min_gap_ms: float = 0.5,
    threshold_db: float = -20.0,
    min_peak_distance_ms: float = 0.3,
) -> int | None:
    """Найти момент прихода первого отражения после прямого звука.

    Алгоритм (через scipy.signal.find_peaks):
        1. Огибающая Hilbert → переводим в дБ относительно пика прямого звука.
        2. Берём область после direct_idx + min_gap_ms.
        3. Ищем все пики огибающей с высотой >= threshold_db и минимальным
           расстоянием между ними `min_peak_distance_ms`.
        4. Возвращаем индекс ПЕРВОГО такого пика.

    Если ничего не найдено (комната очень заглушена / короткая IR / threshold
    слишком строгий), возвращает None — вызывающий код должен решать, что делать
    (например, использовать весь оставшийся хвост).

    Args:
        ir:           импульсная характеристика.
        fs:           частота дискретизации, Гц.
        direct_idx:   позиция прямого звука (из find_direct_sound).
        min_gap_ms:   минимальный зазор от прямого звука, мс. Защищает от
                      «звона» прямого звука сразу после пика.
        threshold_db: порог, ниже которого пики не считаются отражениями
                      (относительно пика прямого звука). Типично −20..−15 дБ.
        min_peak_distance_ms: минимальное расстояние между соседними пиками,
                      мс. Предотвращает «двойное считание» одного пика.

    Returns:
        Индекс первого отражения или None.
    """
    env = hilbert_envelope(ir)
    peak = env[direct_idx]
    if peak < 1e-30:
        return None

    env_db = 20.0 * np.log10(env / peak + 1e-30)

    gap_samples = int(min_gap_ms * 1e-3 * fs)
    search_start = direct_idx + gap_samples
    if search_start >= len(env_db) - 2:
        return None

    region = env_db[search_start:]

    # Параметр distance в find_peaks задаёт минимальное расстояние между пиками
    distance = max(1, int(min_peak_distance_ms * 1e-3 * fs))

    peaks, _ = find_peaks(region, height=threshold_db, distance=distance)
    if len(peaks) == 0:
        return None

    return int(search_start + peaks[0])


def estimate_noise_floor(
    ir: np.ndarray,
    fs: int,
    direct_idx: int,
    tail_fraction: float = 0.2,
) -> float:
    """Грубая оценка уровня шумового пола IR в дБ относительно пика.

    Берём последние tail_fraction отсчётов IR (там должна остаться только
    «тишина» — после полного затухания реверберации) и считаем RMS.

    Args:
        ir, fs, direct_idx: как обычно.
        tail_fraction: доля IR с конца, по которой считаем floor.

    Returns:
        Уровень в дБ относительно пика прямого звука. Обычно от −90 до −40 дБ
        в зависимости от условий записи.
    """
    env = hilbert_envelope(ir)
    peak = env[direct_idx]
    if peak < 1e-30:
        return -120.0

    n_tail = int(tail_fraction * len(ir))
    tail = ir[-n_tail:] if n_tail > 0 else ir[-1:]
    noise_rms = float(np.sqrt(np.mean(tail ** 2) + 1e-30))
    floor_db = 20.0 * np.log10(noise_rms / peak + 1e-30)
    return floor_db


def auto_detect(
    ir: np.ndarray,
    fs: int,
    search_start: int = 0,
    search_end: int | None = None,
    min_gap_ms: float = 0.5,
    threshold_db: float = -20.0,
    min_peak_distance_ms: float = 0.3,
    n_cycles: int = 6,
) -> DetectionResult:
    """Полный автодетект: t₀, первое отражение, окно, нижняя достоверная частота.

    Args:
        ir:           импульсная характеристика.
        fs:           частота дискретизации.
        search_start: начало области поиска прямого звука.
        search_end:   конец области поиска (None = до конца).
        min_gap_ms, threshold_db, min_peak_distance_ms: параметры
            find_first_reflection().
        n_cycles:     сколько периодов сигнала должно поместиться в окно для
                      того, чтобы частота считалась «полностью покрытой».
                      Обычно 4..10. Используется для расчёта f_min_full.

    Returns:
        DetectionResult с найденными параметрами.

    Если первое отражение не найдено, поле first_reflection_idx устанавливается
    в len(ir) − 1 (то есть окно занимает всю оставшуюся часть IR), а f_min_full
    рассчитывается соответственно.
    """
    direct_idx, direct_sub = find_direct_sound(ir, (search_start, search_end or len(ir)))

    refl_idx = find_first_reflection(
        ir, fs, direct_idx,
        min_gap_ms=min_gap_ms,
        threshold_db=threshold_db,
        min_peak_distance_ms=min_peak_distance_ms,
    )
    if refl_idx is None:
        refl_idx = len(ir) - 1

    window_samples = refl_idx - direct_idx
    window_s = window_samples / fs
    f_min_full = n_cycles / window_s if window_s > 0 else float("inf")

    floor_db = estimate_noise_floor(ir, fs, direct_idx)

    return DetectionResult(
        direct_idx=direct_idx,
        direct_subsample=direct_sub,
        first_reflection_idx=refl_idx,
        window_length_samples=window_samples,
        window_length_s=window_s,
        f_min_full=f_min_full,
        floor_db=floor_db,
    )
