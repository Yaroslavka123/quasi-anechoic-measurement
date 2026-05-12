"""Сравнение нескольких измерений одного помещения: «как менялась комната».

Запуск:
    python scripts/compare_treatment.py \\
        data/reference/1.wav data/reference/2.wav \\
        data/reference/3.wav data/reference/4.wav \\
        --labels "до обработки" "после баса" "после среднечастот" "финальный вариант" \\
        --output data/results/room_progression.png

Что показывает:
    1. RT60 (T20, T30) по каждому измерению — должно падать с добавлением панелей.
    2. EDT — раннее затухание, тоже падает.
    3. C50 / C80 — индексы ясности, растут.
    4. D50 — определённость, растёт.
    5. АЧХ всех измерений на одном графике (для сравнения окраски комнаты).
    6. Кривые Шрёдера всех 4-х.

Это «case study» — главная иллюстрация в курсовой того, что алгоритм
действительно измеряет и фиксирует изменения.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from room_acoustics.deconv import normalize_ir
from room_acoustics.detection import auto_detect
from room_acoustics.metrics import compute_metrics, schroeder_integral
from room_acoustics.windowing import frequency_dependent_response


def analyze_one(
    path: Path,
    window_ms: float | None = None,
    n_cycles: float = 6.0,
) -> dict:
    ir_raw, fs = sf.read(str(path))
    if ir_raw.ndim > 1:
        ir_raw = ir_raw[:, 0]
    ir = normalize_ir(ir_raw)
    det = auto_detect(ir, fs, n_cycles=int(n_cycles))

    # Forced window для квази-безэховой АЧХ
    if window_ms is not None:
        refl = det.direct_idx + int(window_ms * 1e-3 * fs)
        refl = min(refl, len(ir) - 1)
    else:
        refl = det.first_reflection_idx

    fd = frequency_dependent_response(
        ir, fs, det.direct_idx, refl,
        f_start=20.0, f_end=20000.0, n_cycles=n_cycles,
    )
    metrics = compute_metrics(ir, fs, det.direct_idx)
    ir_from_direct = ir[det.direct_idx:]
    edc_db = schroeder_integral(ir_from_direct)

    return {
        "path": path,
        "fs": fs,
        "ir": ir,
        "direct_idx": det.direct_idx,
        "ir_from_direct": ir_from_direct,
        "edc_db": edc_db,
        "fd": fd,
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="IR.wav файлы")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Подписи измерений (по умолчанию = имена файлов)")
    parser.add_argument("--window-ms", type=float, default=5.0,
                        help="Длина окна для квази-безэховой АЧХ, мс")
    parser.add_argument("--n-cycles", type=float, default=6.0)
    parser.add_argument(
        "--output", type=Path, default=Path("data/results/room_progression.png")
    )
    args = parser.parse_args()

    if args.labels is None:
        args.labels = [p.stem for p in args.paths]
    if len(args.labels) != len(args.paths):
        parser.error("Количество --labels должно совпадать с количеством файлов.")

    results = []
    for p, label in zip(args.paths, args.labels, strict=True):
        print(f"[*] Анализ {p}  ({label})")
        results.append(analyze_one(p, window_ms=args.window_ms, n_cycles=args.n_cycles))

    # Печатаем таблицу метрик
    print()
    print(
        f"{'Измерение':<25} | {'EDT':>7} | {'T20':>7} | {'T30':>7} | "
        f"{'C50':>6} | {'C80':>6} | {'D50':>5} | {'Floor':>7}"
    )
    print("-" * 90)
    for label, r in zip(args.labels, results, strict=True):
        m = r["metrics"]

        def t_ms(v):
            return f"{v*1000:.0f}мс" if v is not None else "—"

        print(
            f"{label:<25} | {t_ms(m.edt_s):>7} | {t_ms(m.t20_s):>7} | {t_ms(m.t30_s):>7} | "
            f"{m.c50_db:>+5.1f} | {m.c80_db:>+5.1f} | {m.d50*100:>4.1f}% | "
            f"{m.noise_floor_db:>+6.1f}"
        )
    print("\n«—» означает: шумовой пол слишком высок для надёжной оценки по ISO 3382 (< 10 дБ запаса).")

    # Графики
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.32)

    colors = plt.cm.viridis(np.linspace(0, 0.85, len(results)))

    # 1. ETC всех на одном графике
    ax = fig.add_subplot(gs[0, :2])
    for r, label, c in zip(results, args.labels, colors, strict=True):
        env_db = 20 * np.log10(
            np.abs(r["ir_from_direct"]) / np.max(np.abs(r["ir_from_direct"])) + 1e-30
        )
        t_ms = np.arange(len(r["ir_from_direct"])) / r["fs"] * 1000
        ax.plot(t_ms, env_db, label=label, color=c, linewidth=0.5, alpha=0.7)
    ax.set_xlim(0, 500)
    ax.set_ylim(-80, 5)
    ax.set_xlabel("Время от t₀, мс")
    ax.set_ylabel("ETC, дБ")
    ax.set_title("Energy Time Curve — все измерения")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # 2. Schroeder integration
    ax = fig.add_subplot(gs[0, 2])
    for r, label, c in zip(results, args.labels, colors, strict=True):
        t_ms = np.arange(len(r["edc_db"])) / r["fs"] * 1000
        ax.plot(t_ms, r["edc_db"], label=label, color=c, linewidth=1.0)
    ax.axhline(-5, color="grey", linestyle=":", linewidth=0.6)
    ax.axhline(-25, color="grey", linestyle=":", linewidth=0.6)
    ax.set_xlim(0, 800)
    ax.set_ylim(-60, 5)
    ax.set_xlabel("Время от t₀, мс")
    ax.set_ylabel("Уровень, дБ")
    ax.set_title("Schroeder decay curves")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. RT60 (T20 + T30) bar — пропускаем None («—»)
    ax = fig.add_subplot(gs[1, 0])
    x = np.arange(len(results))
    t20 = [r["metrics"].t20_s * 1000 if r["metrics"].t20_s is not None else np.nan
           for r in results]
    t30 = [r["metrics"].t30_s * 1000 if r["metrics"].t30_s is not None else np.nan
           for r in results]
    edt = [r["metrics"].edt_s * 1000 if r["metrics"].edt_s is not None else np.nan
           for r in results]
    width = 0.27
    ax.bar(x - width, edt, width, label="EDT", color="lightblue")
    ax.bar(x, t20, width, label="T20", color="steelblue")
    ax.bar(x + width, t30, width, label="T30", color="navy")
    ax.set_xticks(x)
    ax.set_xticklabels(args.labels, rotation=15, fontsize=8)
    ax.set_ylabel("Время, мс")
    ax.set_title("Время реверберации")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    # 4. C50 + C80 bar
    ax = fig.add_subplot(gs[1, 1])
    c50 = [r["metrics"].c50_db for r in results]
    c80 = [r["metrics"].c80_db for r in results]
    ax.bar(x - width / 2, c50, width, label="C50", color="coral")
    ax.bar(x + width / 2, c80, width, label="C80", color="darkred")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(args.labels, rotation=15, fontsize=8)
    ax.set_ylabel("Уровень, дБ")
    ax.set_title("Индексы ясности")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    # 5. D50
    ax = fig.add_subplot(gs[1, 2])
    d50 = [r["metrics"].d50 * 100 for r in results]
    ax.bar(x, d50, color="green", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(args.labels, rotation=15, fontsize=8)
    ax.set_ylabel("D50, %")
    ax.set_ylim(0, 100)
    ax.set_title("Определённость речи D50")
    ax.grid(True, axis="y", alpha=0.3)

    # 6. АЧХ — все на одном графике
    ax = fig.add_subplot(gs[2, :])
    for r, label, c in zip(results, args.labels, colors, strict=True):
        fd = r["fd"]
        # Нормируем по медиане
        db = fd.magnitude_db - np.median(fd.magnitude_db)
        ax.semilogx(fd.freqs, db, label=label, color=c, linewidth=1.2)
    ax.set_xlim(50, 20000)
    ax.set_ylim(-25, 15)
    ax.set_xlabel("Частота, Гц")
    ax.set_ylabel("Уровень, дБ (отн. медианы)")
    ax.set_title(f"Квази-безэховая АЧХ (окно {args.window_ms} мс, {int(args.n_cycles)} циклов)")
    ax.legend(loc="lower center", fontsize=9, ncol=len(results))
    ax.grid(True, which="both", alpha=0.3)

    fig.suptitle("Прогресс акустической обработки помещения", fontsize=14)
    fig.savefig(args.output, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nГрафик сохранён: {args.output}")


if __name__ == "__main__":
    main()
