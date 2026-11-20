#!/usr/bin/env python3
"""Render the four-experiment Mac budget from a project-local JSON file."""
import argparse
import json
import math
from pathlib import Path
from icl_lab.runtime import configure_process, project_path, require_local_environment


def number(value):
    return (f'{value:.1f}' if value < 100 else f'{value:.0f}').replace('.', ',')


def main():
    configure_process()
    require_local_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        source = project_path(args.input, label='budget JSON', must_exist=True)
        output = project_path(args.output, label='budget image')
        data = json.loads(source.read_text())
        rows, margin = data['experiments'], float(data.get('planning_margin', .30))
        if len(rows) != 4 or not math.isfinite(margin) or not 0 <= margin <= 1:
            raise ValueError('Expected four experiments and a planning margin in [0, 1]')
        hours = []
        for row in rows:
            if len(row['seeds']) != 3 or len(set(row['seeds'])) != 3:
                raise ValueError('Each experiment must contain three distinct seeds')
            seconds = float(row['suite_seconds_per_seed'])
            if not math.isfinite(seconds) or seconds <= 0:
                raise ValueError('suite_seconds_per_seed must be finite and positive')
            hours.append(seconds * len(row['seeds']) / 3600)
        if output.suffix.lower() != '.png':
            raise ValueError('--output must end with .png')
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.error(str(error))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    font = FontProperties(fname=str(Path(matplotlib.get_data_path()) / 'fonts/ttf/DejaVuSans.ttf'))
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.unicode_minus': False})
    colors = {'garg': '#3973BD', 'assign_add': '#168A83',
              'kobayashi': '#8B69BA', 'kobayashi_hyper': '#604394'}
    total, ceiling = sum(hours), max(hours) * (1 + margin)
    figure = plt.figure(figsize=(14, 7), dpi=100, facecolor='#FAFBFD')
    axis = figure.add_axes((.25, .25, .59, .44), facecolor='#FAFBFD')
    figure.text(.05, .93, 'Время первой серии на Mac', fontproperties=font, fontsize=25, weight='bold')
    figure.text(.05, .875, '4 эксперимента · 3 сида · 12 обучений', fontproperties=font,
                fontsize=15, color='#536171')
    figure.text(.05, .79, f'Всего ≈ {number(total)} ч / {number(total / 24)} сут',
                fontproperties=font, fontsize=20, color='#172536')
    figure.text(.53, .79, f'С запасом ≈ {number(total * (1 + margin))} ч / '
                f'{number(total * (1 + margin) / 24)} сут', fontproperties=font, fontsize=16, color='#536171')
    for position, (row, value) in enumerate(zip(rows, hours)):
        axis.barh(position, value, height=.43, color=colors.get(row['key'], '#8B69BA'), zorder=3)
        axis.barh(position, value * margin, left=value, height=.43, color='#D6DBE3', zorder=2)
        axis.scatter(value, position, s=35, color='#172536', edgecolor='white', linewidth=.8, zorder=4)
        axis.text(value * (1 + margin) + ceiling * .025, position, f'≈ {number(value)} ч',
                  va='center', fontproperties=font, fontsize=12, color='#172536')
    axis.set_yticks(range(4), [row['label'] + '\n3 сида' for row in rows], fontproperties=font, fontsize=12)
    axis.set_ylim(3.6, -.6)
    axis.set_xlim(0, ceiling * 1.19)
    axis.xaxis.set_major_locator(MaxNLocator(nbins=6))
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: number(value)))
    axis.set_xlabel('Часы · линейная шкала от нуля', fontproperties=font, fontsize=11, labelpad=10)
    axis.grid(axis='x', color='#E3E7ED', linewidth=.8, zorder=0)
    axis.tick_params(axis='both', length=0, pad=10, colors='#536171')
    for spine in axis.spines.values():
        spine.set_visible(False)
    for label in axis.get_xticklabels():
        label.set_fontproperties(font)
    figure.text(.05, .135, f'● Оценка   ▬ Серым: запас +{margin:.0%}, не доверительный интервал',
                fontproperties=font, fontsize=12, color='#536171')
    figure.text(.05, .085, 'MPS · FP32 · последовательные запуски | оценка по короткому замеру',
                fontproperties=font, fontsize=11, color='#536171')
    figure.text(.05, .04, str(data.get('machine_label', 'Mac')), fontproperties=font, fontsize=11, color='#536171')
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=100, facecolor=figure.get_facecolor())
    plt.close(figure)
    print(output)


if __name__ == '__main__':
    main()
