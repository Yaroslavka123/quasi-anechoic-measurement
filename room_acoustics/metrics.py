"""Метрики помещения по ISO 3382-1.

Считаются из импульсной характеристики:

    T20, T30   — время реверберации (по разным участкам кривой затухания),
                 экстраполированное к падению на 60 дБ.
    EDT        — Early Decay Time, ранняя реверберация, по первым 10 дБ.
    C50, C80   — индексы ясности речи (50 мс) и музыки (80 мс), дБ.
    D50        — определённость (Deutlichkeit), доля ранней энергии.

Все параметры считаются из intgrированной по Шрёдеру кривой затухания:

    EDC(t) = ∫_t^∞ h²(τ) dτ            (Schroeder integral)

В дискретном виде:

    EDC[k] = Σ_{j=k}^{N-1} h[j]²

— это просто кумулятивная сумма квадратов с конца.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RoomMetrics:
    """Сводка акустических параметров помещения."""

    t20_s: float | None     # время реверберации по падению с −5 до −25 дБ (×3)
    t30_s: float | None     # то же с −5 до −35 дБ (×2). Более устойчиво.
    edt_s: float | None     # Early Decay Time: с 0 до −10 дБ (×6).
    c50_db: float           # Clarity for speech (50 мс), дБ
    c80_db: float           # Clarity for music (80 мс), дБ
    d50: float              # Definition: E[0..50мс] / E[0..конец], 0..1
    noise_floor_db: float   # оценка шумового пола относительно пика


def schroeder_integral(ir: np.ndarray) -> np.ndarray:
    """Кривая затухания Шрёдера в дБ.

    EDC(t) = ∫_t^T h²(τ) dτ  (нормирована к EDC(0))

    Returns:
        edc_db: dB-кривая той же длины, что и ir.
                edc_db[0] = 0 dB, монотонно убывает.
    """
    h2 = ir ** 2
    # Обратное кумулятивное суммирование
    edc = np.flip(np.cumsum(np.flip(h2)))
    edc_norm = edc / (edc[0] + 1e-30)
    edc_db = 10.0 * np.log10(edc_norm + 1e-30)
    return edc_db


def estimate_noise_floor_db(edc_db: np.ndarray) -> float:
    """Оценка шумового полa по выходу кривой Шрёдера на плато.

    Берём средний уровень самой плоской по наклону трети кривой.
    Это устойчивее, чем фиксированный 85-й перцентиль или RMS хвоста IR,
    потому что Schroeder integral в шумовом регионе всё ещё медленно убывает
    (накопленная шумовая энергия съедается из суммы).

    Метод:
        1. Считаем локальный наклон EDC через np.diff.
        2. Окно длиной 10% от EDC.
        3. Ищем окно с минимальной по модулю средней производной.
        4. Возвращаем среднее значение EDC в этом окне.
    """
    n = len(edc_db)
    if n < 100:
        return float(edc_db[-1])
    w = max(20, n // 10)
    slope = np.abs(np.diff(edc_db))
    # Средний наклон в скользящем окне
    kernel = np.ones(w) / w
    smooth_slope = np.convolve(slope, kernel, mode="valid")
    # Самое плоское окно — там, где slope минимален
    flattest_start = int(np.argmin(smooth_slope))
    flattest_end = flattest_start + w
    return float(np.mean(edc_db[flattest_start:flattest_end]))


def fit_decay_time(
    edc_db: np.ndarray,
    fs: int,
    db_start: float = -5.0,
    db_end: float = -25.0,
    noise_floor_db: float | None = None,
    min_margin_db: float = 10.0,
) -> float | None:
    """Линейная аппроксимация участка EDC между db_start и db_end.

    Возвращает время в секундах, за которое прямая пересекает уровень −60 дБ
    от db_start. То есть для T20 (db_end = −25): 60-дБ-время = (T_end − T_start) × 3.

    ISO 3382-1 требует, чтобы между db_end и шумовым полом был запас ≥ 10 дБ —
    иначе оценка ненадёжна, и возвращается None.

    Args:
        edc_db: кривая Шрёдера в дБ (отрицательные значения, монотонно убывает).
        fs:     частота дискретизации.
        db_start, db_end: участок аппроксимации.
        noise_floor_db: оценка шумового пола в дБ (отн. пика). Если задана,
            проверяется условие  db_end ≤ noise_floor + min_margin_db, иначе None.
        min_margin_db: минимальный запас по ISO 3382 (по умолчанию 10 дБ).

    Returns:
        Время реверберации T60, секунды. None, если данных недостаточно или
        шумовой пол слишком высокий.
    """
    # Проверка шумового пола (ISO 3382 §3.3)
    if noise_floor_db is not None and db_end < noise_floor_db + min_margin_db:
        return None

    # Находим первое пересечение db_start и db_end
    below_start = np.where(edc_db <= db_start)[0]
    below_end = np.where(edc_db <= db_end)[0]
    if len(below_start) == 0 or len(below_end) == 0:
        return None
    i_start = below_start[0]
    i_end = below_end[0]
    if i_end <= i_start:
        return None

    # Линейная регрессия в дБ vs время на этом участке
    t = np.arange(i_start, i_end + 1) / fs
    y = edc_db[i_start : i_end + 1]
    # y = a*t + b  →  slope = a  (дБ/с)
    A = np.vstack([t, np.ones_like(t)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    if slope >= 0:
        return None  # неубывающая регрессия — IR испорчена шумом

    # Время до падения на 60 дБ от intercept (extrapolation)
    # 60-dB drop time = -60 / slope
    rt60 = -60.0 / slope
    return float(rt60)


def clarity(ir: np.ndarray, fs: int, t_split_ms: float) -> float:
    """Индекс ясности: 10·log10(E[0..t_split] / E[t_split..end]) в дБ.

    C50 = clarity at 50 ms (речь)
    C80 = clarity at 80 ms (музыка)

    Args:
        ir:         IR начиная с прямого звука (т.е. ir[0] ≈ direct).
        fs:         частота дискретизации.
        t_split_ms: точка разделения «ранняя/поздняя энергия», мс.

    Returns:
        Уровень ясности в дБ.
    """
    n_split = int(t_split_ms * 1e-3 * fs)
    n_split = min(n_split, len(ir) - 1)
    if n_split < 1:
        return -np.inf
    e_early = float(np.sum(ir[:n_split] ** 2))
    e_late = float(np.sum(ir[n_split:] ** 2))
    if e_late < 1e-30:
        return float("inf")
    return 10.0 * np.log10(e_early / e_late)


def definition(ir: np.ndarray, fs: int, t_split_ms: float = 50.0) -> float:
    """Определённость D50 = E[0..50мс] / E[0..end]. Безразмерно, 0..1.

    Часто выражают в %. Используется как мера «разборчивости» в помещении.
    """
    n_split = int(t_split_ms * 1e-3 * fs)
    n_split = min(n_split, len(ir))
    if n_split < 1:
        return 0.0
    e_early = float(np.sum(ir[:n_split] ** 2))
    e_total = float(np.sum(ir ** 2))
    if e_total < 1e-30:
        return 0.0
    return e_early / e_total


def compute_metrics(
    ir: np.ndarray,
    fs: int,
    direct_idx: int = 0,
    tail_fraction_for_noise: float = 0.1,
) -> RoomMetrics:
    """Полная сводка ISO 3382 параметров.

    Args:
        ir:         импульсная характеристика.
        fs:         частота дискретизации.
        direct_idx: индекс прямого звука. EDC, C50/C80, D50 считаются от него.
        tail_fraction_for_noise: доля IR с конца для оценки шумового пола.
    """
    # Берём IR начиная с прямого звука
    ir_from_direct = ir[direct_idx:]

    edc_db = schroeder_integral(ir_from_direct)
    # Шумовой пол по плато кривой Шрёдера (см. estimate_noise_floor_db)
    noise_floor_db = estimate_noise_floor_db(edc_db)

    # Передаём оценку шумового пола в фит — он отбракует ненадёжные значения по ISO 3382
    t20 = fit_decay_time(
        edc_db, fs, db_start=-5.0, db_end=-25.0, noise_floor_db=noise_floor_db
    )
    t30 = fit_decay_time(
        edc_db, fs, db_start=-5.0, db_end=-35.0, noise_floor_db=noise_floor_db
    )
    edt = fit_decay_time(
        edc_db, fs, db_start=0.0, db_end=-10.0, noise_floor_db=noise_floor_db
    )

    # T20 экстраполяция: уже идёт от −5 до −25, что есть 20 дБ. Умножаем на 3
    # чтобы получить эквивалент 60-дБ падения. Но fit_decay_time уже возвращает
    # экстраполированное T60. Так что просто берём как есть.
    t20_t60 = t20 if t20 is not None else None
    t30_t60 = t30 if t30 is not None else None
    edt_t60 = edt if edt is not None else None

    c50 = clarity(ir_from_direct, fs, 50.0)
    c80 = clarity(ir_from_direct, fs, 80.0)
    d50 = definition(ir_from_direct, fs, 50.0)

    return RoomMetrics(
        t20_s=t20_t60,
        t30_s=t30_t60,
        edt_s=edt_t60,
        c50_db=c50,
        c80_db=c80,
        d50=d50,
        noise_floor_db=noise_floor_db,
    )


def format_metrics(m: RoomMetrics) -> str:
    """Человеко-читаемый отчёт по метрикам."""

    def fmt_t(v: float | None) -> str:
        return f"{v * 1000:.0f} мс" if v is not None else "—"

    return (
        f"  EDT    : {fmt_t(m.edt_s)}\n"
        f"  T20    : {fmt_t(m.t20_s)}\n"
        f"  T30    : {fmt_t(m.t30_s)}\n"
        f"  C50    : {m.c50_db:+.1f} дБ\n"
        f"  C80    : {m.c80_db:+.1f} дБ\n"
        f"  D50    : {m.d50 * 100:.1f} %\n"
        f"  Floor  : {m.noise_floor_db:.1f} дБ\n"
    )
