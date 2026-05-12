"""Графики: IR во времени, ETC, АЧХ, спектрограмма."""

from __future__ import annotations

import numpy as np


def plot_ir(ax, ir: np.ndarray, fs: int, title: str = "Импульсная характеристика") -> None:
    """График IR во времени (линейная амплитуда)."""
    t_ms = np.arange(len(ir)) / fs * 1000.0
    ax.plot(t_ms, ir, linewidth=0.7)
    ax.set_xlabel("Время, мс")
    ax.set_ylabel("Амплитуда")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def plot_etc(
    ax,
    ir: np.ndarray,
    fs: int,
    title: str = "ETC (Energy Time Curve)",
    floor_db: float = -80.0,
) -> None:
    """Energy Time Curve = 20·log₁₀|h(t)| / max.

    Главный инструмент для визуального поиска первого отражения и оценки RT60.
    """
    env = np.abs(ir) + 1e-30
    env_db = 20.0 * np.log10(env / np.max(env))
    t_ms = np.arange(len(ir)) / fs * 1000.0
    ax.plot(t_ms, env_db, linewidth=0.7)
    ax.set_xlabel("Время, мс")
    ax.set_ylabel("Уровень, дБ")
    ax.set_ylim(floor_db, 5)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def plot_magnitude_response(
    ax,
    h: np.ndarray,
    fs: int,
    nfft: int | None = None,
    title: str = "АЧХ",
    label: str | None = None,
    f_min: float = 20.0,
    f_max: float | None = None,
    smooth_oct: float | None = None,
) -> None:
    """АЧХ модуля по импульсной характеристике (или её окнённой части).

    Args:
        smooth_oct: сглаживание по 1/N октаве (например, 1/24 или 1/12).
                    None — без сглаживания.
    """
    if nfft is None:
        nfft = int(2 ** np.ceil(np.log2(max(len(h), 1024))))

    H = np.fft.rfft(h, n=nfft)
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)

    mag = np.abs(H) + 1e-30
    mag_db = 20.0 * np.log10(mag)

    if smooth_oct is not None and smooth_oct > 0:
        mag_db = _fractional_octave_smooth(freqs, mag_db, smooth_oct)

    ax.semilogx(freqs, mag_db, label=label, linewidth=1.0)
    ax.set_xlabel("Частота, Гц")
    ax.set_ylabel("Уровень, дБ")
    ax.set_title(title)
    ax.set_xlim(f_min, f_max if f_max else fs / 2)
    ax.grid(True, which="both", alpha=0.3)
    if label:
        ax.legend()


def _fractional_octave_smooth(
    freqs: np.ndarray,
    mag_db: np.ndarray,
    fraction: float,
) -> np.ndarray:
    """Сглаживание АЧХ по дольной октаве: на каждой f усредняем в полосе [f/2^(1/(2N)), f·2^(1/(2N))].

    Это стандартный способ показывать АЧХ для аудио — отражает то, как ухо
    воспринимает плавность характеристики.
    """
    smoothed = np.empty_like(mag_db)
    half = fraction / 2.0
    for i, f in enumerate(freqs):
        if f <= 0:
            smoothed[i] = mag_db[i]
            continue
        f_lo = f * 2.0 ** (-half)
        f_hi = f * 2.0 ** (+half)
        mask = (freqs >= f_lo) & (freqs <= f_hi)
        if mask.any():
            # Усреднение в линейной шкале безопаснее, чем в дБ (избегает -inf)
            lin = 10.0 ** (mag_db[mask] / 20.0)
            smoothed[i] = 20.0 * np.log10(np.mean(lin) + 1e-30)
        else:
            smoothed[i] = mag_db[i]
    return smoothed
