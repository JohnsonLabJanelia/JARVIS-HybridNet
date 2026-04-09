#!/usr/bin/env python3
"""
Plot comparison of baseline vs improved model results.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def load_results(path):
    with open(path) as f:
        return json.load(f)


def plot_comparison(baseline_path, improved_paths, output_path='model_comparison.png'):
    """
    Plot comprehensive comparison between baseline and improved models.
    improved_paths: dict of {label: path}
    """
    baseline = load_results(baseline_path)

    fig = plt.figure(figsize=(20, 14), facecolor='#1a1a2e')
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    colors = {'Baseline': '#ff6b6b', 'Temporal': '#4ecdc4',
              'Bone+CVA': '#45b7d1', 'Final': '#96f550'}
    all_results = {'Baseline': baseline}
    for label, path in improved_paths.items():
        all_results[label] = load_results(path)

    def style_ax(ax, title):
        ax.set_facecolor('#16213e')
        ax.set_title(title, color='white', fontsize=13, fontweight='bold', pad=10)
        ax.tick_params(colors='#a0a0a0')
        for spine in ax.spines.values():
            spine.set_color('#2a2a4a')
        ax.grid(True, alpha=0.15, color='white')

    # 1. Overall MPJPE bar chart
    ax1 = fig.add_subplot(gs[0, 0])
    style_ax(ax1, 'Overall MPJPE (mm)')
    labels = list(all_results.keys())
    means = [r['mpjpe_mean'] for r in all_results.values()]
    medians = [r['mpjpe_median'] for r in all_results.values()]
    x = np.arange(len(labels))
    w = 0.35
    bars1 = ax1.bar(x - w/2, means, w, label='Mean',
                     color=[colors.get(l, '#888') for l in labels], alpha=0.85)
    bars2 = ax1.bar(x + w/2, medians, w, label='Median',
                     color=[colors.get(l, '#888') for l in labels], alpha=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, color='white', fontsize=10)
    ax1.set_ylabel('MPJPE (mm)', color='white')
    ax1.legend(facecolor='#16213e', edgecolor='#2a2a4a',
               labelcolor='white', fontsize=9)
    for bar, val in zip(bars1, means):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.1f}', ha='center', va='bottom', color='white', fontsize=9)

    # 2. Per-joint comparison
    ax2 = fig.add_subplot(gs[0, 1:])
    style_ax(ax2, 'Per-Joint MPJPE (mm)')
    if 'per_joint' in baseline:
        joints = list(baseline['per_joint'].keys())
        x_j = np.arange(len(joints))
        width = 0.8 / len(all_results)
        for i, (label, res) in enumerate(all_results.items()):
            if 'per_joint' in res:
                vals = [res['per_joint'].get(j, {}).get('mean', 0) for j in joints]
                ax2.bar(x_j + i*width - 0.4 + width/2, vals, width,
                       label=label, color=colors.get(label, '#888'), alpha=0.8)
        ax2.set_xticks(x_j)
        ax2.set_xticklabels(joints, rotation=45, ha='right', color='white', fontsize=8)
        ax2.set_ylabel('MPJPE (mm)', color='white')
        ax2.legend(facecolor='#16213e', edgecolor='#2a2a4a',
                   labelcolor='white', fontsize=9)

    # 3. Error distribution (CDF)
    ax3 = fig.add_subplot(gs[1, 0])
    style_ax(ax3, 'Error Distribution (CDF)')
    for label, res in all_results.items():
        errors = sorted(res['per_frame_errors'])
        cdf = np.arange(1, len(errors)+1) / len(errors)
        ax3.plot(errors, cdf, color=colors.get(label, '#888'),
                linewidth=2, label=label)
    ax3.set_xlabel('MPJPE (mm)', color='white')
    ax3.set_ylabel('Cumulative fraction', color='white')
    ax3.set_xlim(0, 80)
    ax3.legend(facecolor='#16213e', edgecolor='#2a2a4a',
               labelcolor='white', fontsize=9)

    # 4. Bone length consistency
    ax4 = fig.add_subplot(gs[1, 1])
    style_ax(ax4, 'Bone Length Consistency')
    if 'bone_lengths' in baseline:
        bones = list(baseline['bone_lengths'].keys())
        # Show std for each bone
        for i, (label, res) in enumerate(all_results.items()):
            if 'bone_lengths' in res:
                stds = [res['bone_lengths'].get(b, {}).get('std', 0) for b in bones]
                ax4.barh(np.arange(len(bones)) + i*0.3 - 0.15, stds, 0.25,
                        label=label, color=colors.get(label, '#888'), alpha=0.8)
        ax4.set_yticks(np.arange(len(bones)))
        ax4.set_yticklabels([b.replace('-', '\n') for b in bones],
                           fontsize=6, color='white')
        ax4.set_xlabel('Bone length std (mm)', color='white')
        ax4.legend(facecolor='#16213e', edgecolor='#2a2a4a',
                   labelcolor='white', fontsize=9)

    # 5. Speed comparison
    ax5 = fig.add_subplot(gs[1, 2])
    style_ax(ax5, 'Inference Speed')
    fps_vals = []
    fps_labels = []
    for label, res in all_results.items():
        fps_vals.append(res.get('fps', 0))
        fps_labels.append(label)
    bars = ax5.barh(fps_labels, fps_vals,
                    color=[colors.get(l, '#888') for l in fps_labels], alpha=0.85)
    ax5.set_xlabel('FPS', color='white')
    for bar, val in zip(bars, fps_vals):
        ax5.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}', va='center', color='white', fontsize=10)
    ax5.tick_params(axis='y', colors='white')

    # Summary text
    fig.suptitle('JARVIS-HybridNet: Baseline vs Improved Models',
                 color='white', fontsize=16, fontweight='bold', y=0.98)

    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved comparison plot to {output_path}")


if __name__ == '__main__':
    import argparse, os
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--improved', nargs='+', default=[],
                        help='label:path pairs')
    parser.add_argument('--output', default='model_comparison.png')
    args = parser.parse_args()

    improved = {}
    for item in args.improved:
        label, path = item.split(':', 1)
        improved[label] = path
    plot_comparison(args.baseline, improved, args.output)
