"""CLI: основная точка входа `room-acoustics`.

Подкоманды:
    make-sweep    — сгенерировать ESS-свип и сохранить в WAV.
    analyze       — взять готовую IR.wav и выдать графики + метрики ISO 3382.
    measure       — измерить вживую через sounddevice + Focusrite.
    validate-rew  — сравнить мою АЧХ с экспортом REW (.txt).

Установка пакета добавляет команду `room-acoustics` в PATH (см. pyproject.toml).
"""

from __future__ import annotations

from pathlib import Path

import click


@click.group()
def main() -> None:
    """Автоматическое квази-безэховое измерение АЧХ помещения и источника."""


@main.command("make-sweep")
@click.option("--duration", type=float, default=10.0, help="Длительность ESS, с")
@click.option("--f-start", type=float, default=20.0, help="Нижняя частота, Гц")
@click.option("--f-end", type=float, default=20000.0, help="Верхняя частота, Гц")
@click.option("--fs", type=int, default=48000, help="Sample rate")
@click.option("--gain-db", type=float, default=-6.0, help="Уровень в dBFS")
@click.option("--silence-before", type=float, default=0.5)
@click.option("--silence-after", type=float, default=2.0)
@click.option(
    "--output", type=click.Path(path_type=Path), default=Path("data/raw/sweep.wav")
)
def make_sweep_cmd(
    duration: float,
    f_start: float,
    f_end: float,
    fs: int,
    gain_db: float,
    silence_before: float,
    silence_after: float,
    output: Path,
) -> None:
    """Сгенерировать ESS-свип и сохранить в WAV."""
    import numpy as np
    import soundfile as sf

    from .sweep import generate_ess

    sweep = generate_ess(f_start, f_end, duration, fs)
    sweep *= 10.0 ** (gain_db / 20.0)

    pre = np.zeros(int(silence_before * fs))
    post = np.zeros(int(silence_after * fs))
    signal = np.concatenate([pre, sweep, post])

    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, signal.astype(np.float32), fs, subtype="FLOAT")

    click.echo(f"OK. Сохранено: {output}")
    click.echo(
        f"  Длина: {len(signal) / fs:.2f} с, fs={fs} Hz, gain={gain_db:+.1f} dBFS"
    )


@main.command("analyze")
@click.argument("ir_path", type=click.Path(exists=True, path_type=Path))
@click.option("--output", type=click.Path(path_type=Path), default=None,
              help="PNG с графиками. По умолчанию data/results/<name>.png")
@click.option("--f-start", type=float, default=20.0)
@click.option("--f-end", type=float, default=20000.0)
@click.option("--n-cycles", type=float, default=6.0,
              help="Число циклов в окне для частотно-зависимого анализа")
@click.option("--threshold-db", type=float, default=-20.0,
              help="Порог детектирования первого отражения, дБ от пика")
@click.option("--min-gap-ms", type=float, default=0.5,
              help="Минимальный зазор от t₀ при поиске первого отражения, мс. "
                   "Увеличь до 2-5 мс, если в IR есть «звон» рядом с прямым звуком.")
@click.option("--window-ms", type=float, default=None,
              help="Принудительная длина окна, мс. Перекрывает автодетект. "
                   "Используй, когда знаешь геометрию записи.")
def analyze_cmd(
    ir_path: Path,
    output: Path | None,
    f_start: float,
    f_end: float,
    n_cycles: float,
    threshold_db: float,
    min_gap_ms: float,
    window_ms: float | None,
) -> None:
    """Проанализировать готовую IR.wav: детект t0, окна, АЧХ, метрики."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import soundfile as sf

    from .confidence import ConfidenceLevel, build_confidence_map
    from .deconv import normalize_ir
    from .detection import auto_detect
    from .metrics import compute_metrics, format_metrics, schroeder_integral
    from .plotting import plot_etc, plot_ir
    from .windowing import fixed_window_response, frequency_dependent_response

    ir_raw, fs = sf.read(str(ir_path))
    if ir_raw.ndim > 1:
        ir_raw = ir_raw[:, 0]
    ir = normalize_ir(ir_raw)

    # Автодетект
    detection = auto_detect(
        ir, fs,
        threshold_db=threshold_db,
        min_gap_ms=min_gap_ms,
        n_cycles=int(n_cycles),
    )
    # Опционально перекрываем длину окна вручную
    if window_ms is not None:
        forced_refl = detection.direct_idx + int(window_ms * 1e-3 * fs)
        forced_refl = min(forced_refl, len(ir) - 1)
        detection.first_reflection_idx = forced_refl
        detection.window_length_samples = forced_refl - detection.direct_idx
        detection.window_length_s = detection.window_length_samples / fs
        detection.f_min_full = (
            n_cycles / detection.window_length_s if detection.window_length_s > 0 else float("inf")
        )
        click.echo(f"  (--window-ms={window_ms} мс перекрывает автодетект)")
    click.echo("=== Автодетект ===")
    click.echo(f"  t0 (прямой звук):      idx={detection.direct_idx}  "
               f"({detection.direct_idx / fs * 1000:.2f} мс)")
    click.echo(f"  Первое отражение:      idx={detection.first_reflection_idx}  "
               f"({detection.first_reflection_idx / fs * 1000:.2f} мс)")
    click.echo(f"  T_окна:                {detection.window_length_s * 1000:.2f} мс")
    click.echo(f"  f_min при N={int(n_cycles)} цикл.: {detection.f_min_full:.1f} Гц")
    click.echo(f"  Шумовой пол:           {detection.floor_db:.1f} дБ")

    # АЧХ — частотно-зависимое окно
    fd = frequency_dependent_response(
        ir, fs, detection.direct_idx, detection.first_reflection_idx,
        f_start=f_start, f_end=f_end, n_cycles=n_cycles,
    )
    # Confidence map
    conf = build_confidence_map(fd, ir, fs, detection.direct_idx)

    # Метрики
    m = compute_metrics(ir, fs, detection.direct_idx)
    click.echo("\n=== Метрики ISO 3382 ===")
    click.echo(format_metrics(m))

    # Сохранить графики
    if output is None:
        output = Path("data/results") / f"{ir_path.stem}_analysis.png"
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # IR во времени (zoom around direct sound)
    ax = axes[0, 0]
    t_ms = (np.arange(len(ir)) - detection.direct_idx) / fs * 1000.0
    ax.plot(t_ms, ir, linewidth=0.6)
    ax.axvline(0, color="green", linewidth=0.7, alpha=0.7, label="t₀")
    ax.axvline((detection.first_reflection_idx - detection.direct_idx) / fs * 1000,
               color="orange", linewidth=0.7, alpha=0.7, label="первое отражение")
    ax.set_xlim(-2, 50)
    ax.set_xlabel("Время от t₀, мс")
    ax.set_ylabel("Амплитуда")
    ax.set_title("Импульсная характеристика")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ETC + кривая Шрёдера
    ax = axes[0, 1]
    ir_from_direct = ir[detection.direct_idx:]
    t_ms2 = np.arange(len(ir_from_direct)) / fs * 1000.0
    env_db = 20 * np.log10(np.abs(ir_from_direct) / np.max(np.abs(ir_from_direct)) + 1e-30)
    ax.plot(t_ms2, env_db, linewidth=0.5, label="ETC", alpha=0.6)
    edc_db = schroeder_integral(ir_from_direct)
    ax.plot(t_ms2, edc_db, linewidth=1.5, label="Шрёдер", color="black")
    ax.axhline(-5, color="grey", linestyle=":", linewidth=0.7)
    ax.axhline(-25, color="grey", linestyle=":", linewidth=0.7,
               label="T20 уровни (−5, −25)")
    ax.set_xlim(0, min(1000, t_ms2[-1]))
    ax.set_ylim(-90, 5)
    ax.set_xlabel("Время от t₀, мс")
    ax.set_ylabel("Уровень, дБ")
    ax.set_title("ETC + кривая затухания Шрёдера")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # АЧХ с частотно-зависимым окном + сравнение с фикс. окном
    ax = axes[1, 0]
    fixed_freqs, fixed_db = fixed_window_response(
        ir, fs, detection.direct_idx, detection.first_reflection_idx
    )
    # Сместим обе кривые так, чтобы средний уровень совпадал (для наглядности)
    f_mask = (fixed_freqs >= f_start) & (fixed_freqs <= f_end)
    fixed_db_shifted = fixed_db - np.median(fixed_db[f_mask])
    fd_db_shifted = fd.magnitude_db - np.median(fd.magnitude_db)
    ax.semilogx(fixed_freqs, fixed_db_shifted, linewidth=0.6,
                color="lightgrey", label="фикс. окно")
    ax.semilogx(fd.freqs, fd_db_shifted, linewidth=1.5,
                label=f"частотно-зависимое ({int(n_cycles)} циклов)")
    ax.set_xlim(f_start, f_end)
    ax.set_ylim(-30, 15)
    ax.set_xlabel("Частота, Гц")
    ax.set_ylabel("Уровень, дБ (отн. медианы)")
    ax.set_title("АЧХ")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    # Confidence map
    ax = axes[1, 1]
    # Раскраска точек АЧХ по уровню доверия
    colors = {
        ConfidenceLevel.GREEN.value: "green",
        ConfidenceLevel.YELLOW.value: "orange",
        ConfidenceLevel.RED.value: "red",
    }
    for level_val, color in colors.items():
        mask = conf.levels == level_val
        if mask.any():
            label = {
                ConfidenceLevel.GREEN.value: "достоверно",
                ConfidenceLevel.YELLOW.value: "на границе",
                ConfidenceLevel.RED.value: "недостоверно",
            }[level_val]
            ax.semilogx(conf.freqs[mask], fd_db_shifted[mask],
                        "o", markersize=3, color=color, label=label)
    # Заштрихуем «полосу неопределённости» вокруг кривой
    ax.fill_between(
        fd.freqs,
        fd_db_shifted - conf.magnitude_uncertainty_db,
        fd_db_shifted + conf.magnitude_uncertainty_db,
        color="grey", alpha=0.2, label="±неопр. (дБ)",
    )
    ax.set_xlim(f_start, f_end)
    ax.set_ylim(-30, 15)
    ax.set_xlabel("Частота, Гц")
    ax.set_ylabel("Уровень, дБ")
    ax.set_title("АЧХ с картой доверия")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    fig.suptitle(f"Анализ: {ir_path.name}", fontsize=13)
    fig.tight_layout()
    fig.savefig(output, dpi=120)
    plt.close(fig)
    click.echo(f"\nГрафик сохранён: {output}")


@main.command("validate-rew")
@click.argument("ir_wav", type=click.Path(exists=True, path_type=Path))
@click.argument("rew_txt", type=click.Path(exists=True, path_type=Path))
@click.option("--output", type=click.Path(path_type=Path), default=None)
@click.option("--n-cycles", type=float, default=6.0)
@click.option("--window-ms", type=float, default=None,
              help="Принудительная длина окна, мс. Без него — автодетект.")
@click.option("--min-gap-ms", type=float, default=0.5)
@click.option("--threshold-db", type=float, default=-20.0)
def validate_rew_cmd(
    ir_wav: Path,
    rew_txt: Path,
    output: Path | None,
    n_cycles: float,
    window_ms: float | None,
    min_gap_ms: float,
    threshold_db: float,
) -> None:
    """Сравнить мою АЧХ с экспортированной из REW (Freq/SPL/Phase в txt)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import soundfile as sf

    from .deconv import normalize_ir
    from .detection import auto_detect
    from .windowing import frequency_dependent_response

    # 1. Загружаем IR и считаем нашу АЧХ
    ir_raw, fs = sf.read(str(ir_wav))
    if ir_raw.ndim > 1:
        ir_raw = ir_raw[:, 0]
    ir = normalize_ir(ir_raw)

    detection = auto_detect(
        ir, fs,
        threshold_db=threshold_db,
        min_gap_ms=min_gap_ms,
        n_cycles=int(n_cycles),
    )
    if window_ms is not None:
        forced_refl = detection.direct_idx + int(window_ms * 1e-3 * fs)
        forced_refl = min(forced_refl, len(ir) - 1)
        detection.first_reflection_idx = forced_refl
        detection.window_length_samples = forced_refl - detection.direct_idx
        detection.window_length_s = detection.window_length_samples / fs
        detection.f_min_full = n_cycles / detection.window_length_s
    fd = frequency_dependent_response(
        ir, fs, detection.direct_idx, detection.first_reflection_idx,
        f_start=20.0, f_end=20000.0, n_cycles=n_cycles,
    )

    # 2. Парсим REW txt: пропускаем строки начинающиеся с *, читаем 3 колонки
    rew_freqs = []
    rew_db = []
    with open(rew_txt, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("*"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    rew_freqs.append(float(parts[0]))
                    rew_db.append(float(parts[1]))
                except ValueError:
                    continue
    rew_freqs = np.array(rew_freqs)
    rew_db_arr = np.array(rew_db)

    # 3. Выравниваем уровни: вычитаем медиану в перекрывающемся диапазоне
    f_overlap = (fd.freqs >= max(rew_freqs.min(), 50)) & (fd.freqs <= min(rew_freqs.max(), 10000))
    rew_overlap = (rew_freqs >= 50) & (rew_freqs <= 10000)
    my_db_shifted = fd.magnitude_db - np.median(fd.magnitude_db[f_overlap])
    rew_db_shifted = rew_db_arr - np.median(rew_db_arr[rew_overlap])

    # 4. Интерполируем для прямого сравнения
    my_db_interp = np.interp(rew_freqs, fd.freqs, my_db_shifted)
    diff_db = my_db_interp - rew_db_shifted
    # Считаем разницу только в полосе, где у нас оба источника есть надёжные данные
    cmp_mask = (rew_freqs >= max(detection.f_min_full * 1.5, 50)) & (rew_freqs <= 15000)
    mae = float(np.mean(np.abs(diff_db[cmp_mask])))
    median_abs = float(np.median(np.abs(diff_db[cmp_mask])))

    click.echo(f"=== Сравнение с REW ({rew_txt.name}) ===")
    click.echo(f"  Полоса сравнения: {detection.f_min_full * 1.5:.0f} .. 15000 Гц")
    click.echo(f"  MAE по полосе:    {mae:.2f} дБ")
    click.echo(f"  Медиана |diff|:   {median_abs:.2f} дБ")

    # 5. График
    if output is None:
        output = Path("data/results") / f"{ir_wav.stem}_vs_rew.png"
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax1.semilogx(rew_freqs, rew_db_shifted, linewidth=1.2, label="REW (эталон)", color="black")
    ax1.semilogx(fd.freqs, my_db_shifted, linewidth=1.2, label=f"мой алгоритм ({int(n_cycles)} циклов)",
                 color="tab:blue")
    ax1.axvline(detection.f_min_full, color="red", linestyle=":", linewidth=0.8,
                label=f"f_min (полные циклы) = {detection.f_min_full:.0f} Гц")
    ax1.set_xlim(20, 20000)
    ax1.set_ylim(-30, 15)
    ax1.set_ylabel("Уровень, дБ (отн. медианы)")
    ax1.set_title(f"АЧХ: мой алгоритм vs REW — {ir_wav.name}")
    ax1.legend()
    ax1.grid(True, which="both", alpha=0.3)

    ax2.semilogx(rew_freqs, diff_db, linewidth=1.0, color="tab:red")
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.axhline(3, color="grey", linestyle=":", linewidth=0.5)
    ax2.axhline(-3, color="grey", linestyle=":", linewidth=0.5)
    ax2.set_xlim(20, 20000)
    ax2.set_ylim(-15, 15)
    ax2.set_xlabel("Частота, Гц")
    ax2.set_ylabel("Разница (мой − REW), дБ")
    ax2.set_title(f"Расхождение  | MAE = {mae:.2f} дБ в полосе {detection.f_min_full * 1.5:.0f}..15000 Гц")
    ax2.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output, dpi=120)
    plt.close(fig)
    click.echo(f"График сохранён: {output}")


if __name__ == "__main__":
    main()
