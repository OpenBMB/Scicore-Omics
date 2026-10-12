from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

OUT = Path(__file__).resolve().parent / 'figure_outputs'
OUT.mkdir(exist_ok=True)
STAGE_COLORS = ['#8C9AA6', '#4E79A7', '#A78BD0']
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11,
    'pdf.fonttype': 42, 'svg.fonttype': 'none', 'axes.linewidth': 0.9,
})


def save(fig, name):
    for ext in ('png', 'pdf', 'svg'):
        path = OUT / f'{name}.{ext}'
        fig.savefig(path, dpi=600, facecolor='white')
        print(path)
    plt.close(fig)


def plot_compact():
    fig, (top, bottom) = plt.subplots(
        2, 1, sharex=True, figsize=(5.2, 5.6),
        gridspec_kw={'hspace': 0.10},
    )
    x = [0, 1, 2]
    series = [
        ('BLEU-4', [0.024, 0.641, 0.791], '#1F77B4', 'o'),
        ('ROUGE-L', [0.022, 0.177, 0.606], '#FF7F0E', 's'),
        ('BERTScore', [0.346, 0.585, 0.831], '#2CA02C', '^'),
    ]
    for name, values, color, marker in series:
        top.plot(x, values, label=name, color=color, marker=marker,
                 markersize=5, linewidth=1.5)
        for i, value in enumerate(values):
            if name == 'BLEU-4':
                offset = (0, 7) if i != 2 else (0, -14)
            elif name == 'ROUGE-L':
                offset = (0, -14) if i == 0 else (0, 7)
            else:
                offset = (0, 7) if i != 1 else (0, -15)
            top.annotate(f'{value:.3f}', (i, value), xytext=offset,
                         textcoords='offset points', ha='center',
                         fontsize=8.5, color=color)
    top.set_ylabel('Automatic metric score', fontsize=11.5, labelpad=7)
    top.set_ylim(-0.06, 1.04)
    top.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    top.legend(loc='upper left', frameon=False, fontsize=8.5, ncol=3,
               handlelength=1.2, columnspacing=0.7)
    top.tick_params(axis='x', bottom=False, labelbottom=False, labelsize=9)
    # 300 ratings per stage; no normalization or invented uncertainty.
    values = [179 / 300, 655 / 300, 793 / 300]
    bottom.bar(x, values, width=0.56, color=STAGE_COLORS,
               edgecolor='#454545', linewidth=0.65)
    for i, value in enumerate(values):
        bottom.text(i, value + 0.055, f'{value:.3f}', ha='center',
                    va='bottom', fontsize=10)
    bottom.set_ylabel('Mean expert rating (0–3)', fontsize=11.5, labelpad=7)
    bottom.set_ylim(0, 3.05)
    bottom.yaxis.set_major_locator(MultipleLocator(0.5))
    bottom.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    bottom.set_xticks(x, ['Base', 'Stage II', 'Stage III'], fontsize=10)
    bottom.set_xlabel('Model stage', fontsize=11.5, labelpad=6)
    for label, ax in zip(['d', 'e'], [top, bottom]):
        ax.set_xlim(-0.5, 2.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', color='#E4E4E4', linewidth=0.55)
        ax.set_axisbelow(True)
        ax.text(-0.16, 1.01, label, transform=ax.transAxes,
                fontweight='bold', fontsize=15, va='bottom')
    fig.subplots_adjust(left=0.19, right=0.975, bottom=0.13, top=0.95)
    fig.align_ylabels([top, bottom])
    save(fig, 'SciCore_Omics_Fig2de_compact')


def plot_distribution():
    # Counts for raw scores 0, 1, 2, 3. Each row sums to 300 ratings.
    counts = [[198, 40, 47, 15], [48, 27, 47, 178], [10, 18, 41, 231]]
    stages = ['base_stage', 'stage2', 'stage3']
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    width = 0.24
    for i, (stage, row) in enumerate(zip(stages, counts)):
        assert sum(row) == 300
        percentages = [count / 300 * 100 for count in row]
        positions = [score + (i - 1) * width for score in range(4)]
        ax.bar(positions, percentages, width=width, color=STAGE_COLORS[i],
               label=stage, edgecolor='#333333', linewidth=0.4)
    ax.set_xticks(range(4), ['0', '1', '2', '3'])
    ax.set_xlabel('Raw expert score')
    ax.set_ylabel('Ratings (%)')
    ax.set_ylim(0, 85)
    ax.yaxis.set_major_locator(MultipleLocator(20))
    ax.grid(axis='y', color='#D9D9D9', linewidth=0.55)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=False, ncol=3, loc='upper center',
              bbox_to_anchor=(0.5, 1.13), fontsize=9)
    fig.tight_layout(pad=0.8)
    save(fig, 'Fig_score_distribution_purple')


if __name__ == '__main__':
    plot_compact()
    plot_distribution()
