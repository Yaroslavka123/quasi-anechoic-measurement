"""Сгенерировать ESS-свип и сохранить в WAV.

Используется отдельно (а) для воспроизведения через любой плеер,
(б) для проверки, что сгенерированный сигнал звучит так, как ожидается.

Запуск:
    python scripts/make_sweep.py --duration 10 --f-start 20 --f-end 20000 \
        --output data/raw/sweep.wav
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

from room_acoustics.sweep import generate_ess


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=10.0, help="Длительность, с")
    parser.add_argument("--f-start", type=float, default=20.0, help="Нижняя частота, Гц")
    parser.add_argument("--f-end", type=float, default=20000.0, help="Верхняя частота, Гц")
    parser.add_argument("--fs", type=int, default=48000, help="Sample rate, Гц")
    parser.add_argument(
        "--gain-db",
        type=float,
        default=-6.0,
        help="Выходной уровень в дБFS (рекомендуется -6 для запаса от клиппинга)",
    )
    parser.add_argument(
        "--silence-before",
        type=float,
        default=0.5,
        help="Тишина в начале файла, с (даёт системе устаканиться)",
    )
    parser.add_argument(
        "--silence-after",
        type=float,
        default=2.0,
        help="Тишина в конце файла, с (захватывает реверберационный хвост)",
    )
    parser.add_argument("--output", type=Path, default=Path("data/raw/sweep.wav"))
    args = parser.parse_args()

    sweep = generate_ess(args.f_start, args.f_end, args.duration, args.fs)

    gain = 10.0 ** (args.gain_db / 20.0)
    sweep *= gain

    pre = np.zeros(int(args.silence_before * args.fs))
    post = np.zeros(int(args.silence_after * args.fs))
    signal = np.concatenate([pre, sweep, post])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, signal.astype(np.float32), args.fs, subtype="FLOAT")

    total_s = len(signal) / args.fs
    print(f"OK. Сохранено: {args.output}")
    print(f"   Длина файла: {total_s:.2f} с (тишина {args.silence_before} + свип {args.duration} + тишина {args.silence_after})")
    print(f"   Sample rate: {args.fs} Гц")
    print(f"   Уровень:     {args.gain_db:+.1f} dBFS")
    print(f"   Полоса:      {args.f_start} .. {args.f_end} Гц")


if __name__ == "__main__":
    main()
