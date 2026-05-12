# quasi-anechoic-measurement

Алгоритм автоматического квази-безэхового измерения АЧХ помещения и источника звука с
**частотно-зависимым окном** импульсной характеристики.

Курсовая работа. Идея — построить полностью автоматический pipeline, который:

1. Воспроизводит логарифмический синус-свип (exponential sine sweep, ESS).
2. Записывает отклик микрофоном.
3. Деконволюцией восстанавливает импульсную характеристику (IR).
4. Автоматически определяет:
   - сквозную задержку системы (через loopback),
   - момент прихода прямого звука `t₀`,
   - момент прихода первого отражения `t_refl`.
5. Применяет **частотно-зависимое окно** — на каждой частоте `f` окно имеет длину
   `T(f) = min(N_cycles / f, t_refl - t₀)`. Это даёт максимум информации на ВЧ
   и честно ограничивает достоверность на НЧ.
6. Выдаёт:
   - квази-безэховую АЧХ источника,
   - АЧХ помещения,
   - **карту доверия** (где данным можно верить, где — нет),
   - параметры помещения по ISO 3382 (RT60, EDT, C50, C80, D50).

## Аппаратура (под которую написан код)

- Аудио-интерфейс: **Focusrite Scarlett Solo 3rd gen** (или любой ASIO/WASAPI 2-канальный).
- Микрофон: **Audio-Technica AT2020** (кардиоидный конденсаторный).
  - В курсовой обсуждаются ограничения, связанные с использованием неизмерительного
    микрофона. По оси (on-axis) AT2020 достаточно ровный, чтобы получить квази-безэховую
    АЧХ источника с погрешностью ±2–3 дБ.
- Кабель TS моно-джек для **loopback-калибровки** задержки.

## Установка

```powershell
# Windows / PowerShell
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -e .[dev,sim]
```

```bash
# Linux / macOS
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,sim]"
```

## Быстрый старт

### 1. Сгенерировать и сохранить тестовый ESS

```bash
python scripts/make_sweep.py --duration 10 --f-start 20 --f-end 20000 --output data/raw/sweep.wav
```

### 2. Прогнать алгоритм на синтетической комнате (без железа)

```bash
python scripts/synthetic_demo.py
```

Этот скрипт симулирует комнату через `pyroomacoustics`, прогоняет через неё свип,
и показывает результаты полного pipeline. Полезно для отладки алгоритма до подключения
реального микрофона.

### 3. Реальное измерение

```bash
# Записать (с loopback)
python scripts/measure.py --output data/raw/room1.wav

# Проанализировать
python scripts/analyze.py data/raw/room1.wav --output data/results/room1
```

## Структура проекта

```
room_acoustics/
├── sweep.py          # генерация ESS + инверсный фильтр (Farina 2000)
├── deconv.py         # деконволюция → импульсная характеристика
├── io.py             # play+record через sounddevice (loopback support)
├── detection.py      # автодетект t0 и первого отражения
├── windowing.py      # частотно-зависимое окно
├── metrics.py        # RT60, EDT, C50/C80, D50 (ISO 3382)
├── confidence.py     # карта доверия
├── plotting.py       # все графики
└── cli.py            # точка входа
```

## Литература

1. Farina, A. (2000). *Simultaneous measurement of impulse response and distortion with a swept-sine technique*. AES 108th Convention.
2. Müller, S. & Massarani, P. (2001). *Transfer-function measurement with sweeps*. JAES 49(6).
3. ISO 3382-1:2009. *Acoustics — Measurement of room acoustic parameters*.
4. Holters, M., Corbach, T., & Zölzer, U. (2009). *Impulse response measurement techniques and their applicability in the real world*. DAFx-09.

## Лицензия

MIT
