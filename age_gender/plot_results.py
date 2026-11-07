"""Regenerate historical summaries and figures directly from committed CSVs."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root = Path(__file__).resolve().parents[1]
    records = []
    histories = {}
    for path in sorted((root / 'experiments/logs').glob('*.csv')):
        with path.open(encoding='utf-8-sig') as stream:
            rows = list(csv.DictReader(stream))
        histories[path.stem] = rows
        row = rows[-1]
        records.append({'run': path.stem, 'epochs': int(row['epoch']),
                        'val_mae_years': float(row['val_mae']),
                        'val_accuracy_percent': float(row['val_acc']),
                        'source': path.relative_to(root).as_posix()})
    out = root / 'experiments'
    (out / 'figures').mkdir(exist_ok=True)
    (out / 'summary.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    table = ['# Historical validation results', '', 'Generated from final rows of committed CSV logs. Not independently remeasured test results.', '', '| Run | Epochs | Age MAE (years) | Gender accuracy (%) | Evidence |', '| --- | ---: | ---: | ---: | --- |']
    for r in records:
        table.append(f'| {r["run"]} | {r["epochs"]} | {r["val_mae_years"]:.4f} | {r["val_accuracy_percent"]:.4f} | [{r["run"]}](../{r["source"]}) |')
    table += ['', 'The 15-epoch comparisons and 35-epoch main experiment have different training budgets. See [provenance](provenance.md) for split and checkpoint limitations.', '', '![Validation curves](../experiments/figures/validation_curves.png)', '', '![Comparison](../experiments/figures/comparison.png)']
    (root / 'docs/experiments.md').write_text('\n'.join(table) + '\n', encoding='utf-8')
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for name, rows in histories.items():
        epochs = [int(r['epoch']) for r in rows]
        for ax, field in zip(axes, ['val_mae', 'val_acc']):
            ax.plot(epochs, [float(r[field]) for r in rows], label=name, linewidth=1.6)
    for ax, ylabel in zip(axes, ['Age MAE (years)', 'Gender accuracy (%)']):
        ax.set(xlabel='Epoch', ylabel=ylabel)
        ax.grid(alpha=0.2)
    axes[1].legend(fontsize=7)
    fig.suptitle('Historical validation records · not independently remeasured')
    fig.tight_layout()
    fig.savefig(out / 'figures/validation_curves.png', dpi=180)
    plt.close(fig)
    short = [r for r in records if r['epochs'] == 15]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    labels = [r['run'].replace('_15', '').replace('_', '\n') for r in short]
    for ax, field, title in zip(axes, ['val_mae_years', 'val_accuracy_percent'], ['Age MAE (years) ↓', 'Gender accuracy (%) ↑']):
        ax.bar(labels, [r[field] for r in short], color='#3976ad')
        ax.set_title(title)
        ax.tick_params(axis='x', labelsize=7)
        ax.grid(axis='y', alpha=0.2)
    axes[1].set_ylim(95, 100)
    fig.suptitle('15-epoch historical validation comparison')
    fig.tight_layout()
    fig.savefig(out / 'figures/comparison.png', dpi=180)
    plt.close(fig)
    print(f'Generated summaries for {len(records)} runs')


if __name__ == '__main__':
    main()
