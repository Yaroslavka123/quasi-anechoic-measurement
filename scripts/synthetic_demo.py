"""Демо полного pipeline на синтетической комнате (без реального микрофона).

Что делает:
    1. Строит модельную IR (прямой звук + 4 явных отражения + затухающий хвост).
    2. Генерирует ESS-свип.
    3. Имитирует "запись": свёртка свипа с IR.
    4. Деконволюция → восстановленная IR.
    5. Рисует 4 графика: модельная IR, восстановленная IR, ETC, АЧХ.

Запуск:
    python scripts/synthetic_demo.py [--output data/results/synthetic.png]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # без GUI

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import fftconvolve

from room_acoustics.deconv import deconvolve_time, normalize_ir, trim_ir
from room_acoustics.plotting import plot_etc, plot_ir, plot_magnitude_response
from room_acoustics.sweep import generate_ess, inverse_filter_ess


FS = 48000


def synthetic_room_ir(
    fs: int = FS,
    direct_delay_ms: float = 3.0,
    length_s: float = 0.8,
) -> tuple[np.ndarray, int]:
    """Реалистичная синтетическая IR небольшой комнаты."""
    n = int(length_s * fs)
    ir = np.zeros(n)

    direct_idx = int(direct_delay_ms * 1e-3 * fs)
    ir[direct_idx] = 1.0

    # Раннее отражение (пол)
    reflections = [
        (8.0, 0.55),    # пол
        (12.0, 0.40),   # ближняя стена
        (18.0, 0.30),   # потолок
        (28.0, 0.25),   # дальняя стена
        (45.0, 0.18),   # вторичные
    ]
    for t_ms, amp in reflections:
        idx = int(t_ms * 1e-3 * fs)
        if idx < n:
            ir[idx] += amp

    # Реверберационный хвост: цветной шум с экспоненциальным спаданием
    rng = np.random.default_rng(seed=12345)
    rt60 = 0.45
    t = np.arange(n) / fs
    decay = np.exp(-3.0 * np.log(10) * t / rt60)
    noise = rng.standard_normal(n) * 0.06 * decay
    noise[:direct_idx] = 0
    ir = ir + noise

    return ir, direct_idx


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/results/synthetic.png"))
    parser.add_argument("--duration", type=float, default=3.0, help="Длина ESS, с")
    parser.add_argument("--f-start", type=float, default=20.0)
    parser.add_argument("--f-end", type=float, default=20000.0)
    args = parser.parse_args()

    print(f"[1/5] Генерация ESS {args.f_start}..{args.f_end} Гц, {args.duration} c")
    s = generate_ess(args.f_start, args.f_end, args.duration, FS)
    g = inverse_filter_ess(args.f_start, args.f_end, args.duration, FS)

    print("[2/5] Построение синтетической IR")
    ir_true, direct_idx = synthetic_room_ir(fs=FS)

    print("[3/5] Имитация записи (свёртка свипа с IR)")
    # Добавим немного шума для реалистичности
    rng = np.random.default_rng(0)
    recorded = fftconvolve(s, ir_true, mode="full")
    snr_db = 60.0
    sig_power = np.mean(recorded ** 2)
    noise_power = sig_power * 10 ** (-snr_db / 10)
    recorded = recorded + rng.standard_normal(len(recorded)) * np.sqrt(noise_power)

    print("[4/5] Деконволюция → IR")
    ir_full = deconvolve_time(recorded, g)
    ir_recovered, offset = trim_ir(ir_full, sweep_length=len(s), pre_pad=1024)
    ir_recovered = normalize_ir(ir_recovered)

    # Выровним отрисовку: ir_true начинается в 0, ir_recovered начинается на offset
    # Возьмём кусок IR_recovered, начиная с offset, длиной как у ir_true
    ir_recovered_aligned = ir_recovered[offset : offset + len(ir_true)]

    print("[5/5] Построение графиков → " + str(args.output))
    args.output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # IR
    plot_ir(axes[0, 0], ir_true, FS, title="Модельная IR (истина)")
    plot_ir(axes[0, 1], ir_recovered_aligned, FS, title="Восстановленная IR (Farina)")

    # ETC сравнение на одном графике
    ax = axes[1, 0]
    t_ms = np.arange(len(ir_true)) / FS * 1000.0
    env_true_db = 20 * np.log10(np.abs(ir_true) / np.max(np.abs(ir_true)) + 1e-30)
    env_rec_db = 20 * np.log10(
        np.abs(ir_recovered_aligned) / np.max(np.abs(ir_recovered_aligned)) + 1e-30
    )
    ax.plot(t_ms, env_true_db, label="истина", linewidth=0.7)
    ax.plot(t_ms, env_rec_db, label="восстановлено", linewidth=0.7, alpha=0.7)
    ax.set_xlabel("Время, мс")
    ax.set_ylabel("Уровень, дБ")
    ax.set_ylim(-80, 5)
    ax.set_xlim(0, 80)
    ax.set_title("ETC (Energy Time Curve)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # АЧХ сравнение
    plot_magnitude_response(
        axes[1, 1], ir_true, FS, title="АЧХ", label="истина", smooth_oct=1 / 12
    )
    plot_magnitude_response(
        axes[1, 1],
        ir_recovered_aligned,
        FS,
        label="восстановлено",
        smooth_oct=1 / 12,
    )

    fig.suptitle("Synthetic demo: проверка алгоритма деконволюции", fontsize=13)
    fig.tight_layout()
    fig.savefig(args.output, dpi=120)
    print(f"OK. Сохранено: {args.output}")

    # Печатные метрики
    print()
    print("=== Проверка корректности ===")
    peak_true = int(np.argmax(np.abs(ir_true)))
    peak_rec = int(np.argmax(np.abs(ir_recovered_aligned)))
    print(f"  Пик прямого звука (истина):       {peak_true} отсчётов = {peak_true / FS * 1000:.2f} мс")
    print(f"  Пик прямого звука (восстановлено): {peak_rec} отсчётов = {peak_rec / FS * 1000:.2f} мс")
    print(f"  Расхождение: {abs(peak_true - peak_rec)} отсчётов")

    # Среднеквадратическая ошибка после нормировки
    mse = np.mean((ir_true - ir_recovered_aligned) ** 2)
    print(f"  MSE между истиной и восстановлением: {mse:.2e}")


if __name__ == "__main__":
    main()
