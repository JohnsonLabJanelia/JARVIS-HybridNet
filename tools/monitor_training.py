#!/usr/bin/env python3
"""
Live training dashboard for JARVIS HybridNet (and other nets).

Usage:
    # Auto-detect latest run:
    python tools/monitor_training.py --project mouseJan30 --net HybridNet

    # Point at a specific run directory:
    python tools/monitor_training.py --logdir projects/mouseJan30/logs/HybridNet/Run_20260316-222315

    # Monitor cluster logs (path auto-resolves via mount):
    python tools/monitor_training.py \\
        --logdir /mnt/johnson_lab/doq/JARVIS-HybridNet/projects/mouseJan30/logs/HybridNet/Run_20260316-222315

    # Also tail a LSF job output:
    python tools/monitor_training.py --project mouseJan30 --net HybridNet --jobid 148808346

Options:
    --project       JARVIS project name (e.g. mouseJan30)
    --net           Network: HybridNet | CenterDetect | KeypointDetect
    --logdir        Explicit path to the TF events directory
    --jobdir        Base JARVIS project root (default: auto-detect)
    --jobid         LSF job ID to tail via bpeek over SSH
    --ssh-host      SSH host for bpeek (default: doq@login1)
    --interval      Refresh interval in seconds (default: 30)
    --no-show       Don't open interactive window; save PNG only
    --outdir        Directory to save dashboard PNGs (default: /tmp)
"""

import argparse
import os
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ── TensorBoard event reader ─────────────────────────────────────────────────
def read_tf_events(logdir):
    """Return dict of tag -> list of (step, value) sorted by step."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator, TENSORS, SCALARS,
        )
    except ImportError:
        print("tensorboard not found — pip install tensorboard")
        return {}

    ea = EventAccumulator(str(logdir), size_guidance={SCALARS: 0})
    ea.Reload()
    data = {}
    for tag in ea.Tags().get('scalars', []):
        events = ea.Scalars(tag)
        data[tag] = [(e.step, e.value) for e in events]
    return data


def find_latest_logdir(project_root, net):
    """Return path to the most-recently-modified run directory."""
    base = Path(project_root) / 'logs' / net
    if not base.exists():
        return None
    runs = sorted(base.iterdir(), key=lambda p: p.stat().st_mtime)
    return runs[-1] if runs else None


def bpeek_tail(job_id, ssh_host, n_lines=60):
    """Fetch last n_lines from a running LSF job via bpeek over SSH."""
    try:
        result = subprocess.run(
            ['ssh', '-o', 'ConnectTimeout=8', ssh_host, f'bpeek {job_id}'],
            capture_output=True, text=True, timeout=15,
        )
        lines = (result.stdout + result.stderr).splitlines()
        # Strip tqdm progress bar noise (lines with \r overwriting)
        lines = [l for l in lines if '\r' not in l]
        return lines[-n_lines:]
    except Exception as e:
        return [f"[bpeek error: {e}]"]


# ── Plotting ─────────────────────────────────────────────────────────────────
COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']

def make_dashboard(scalars, logdir, job_lines=None, outdir='/tmp', save=True, show=True, fig=None):
    """Render dashboard figure; optionally save PNG and/or display."""
    ts = datetime.now().strftime('%H:%M:%S')

    # Group tags
    loss_tags   = sorted([t for t in scalars if 'loss' in t.lower()])
    acc_tags    = sorted([t for t in scalars if 'acc' in t.lower()])
    lr_tags     = sorted([t for t in scalars if 'lr' in t.lower() or 'learning' in t.lower()])
    other_tags  = sorted([t for t in scalars if t not in loss_tags + acc_tags + lr_tags])

    n_plots = bool(loss_tags) + bool(acc_tags) + bool(lr_tags) + bool(other_tags)
    has_log = bool(job_lines)
    n_rows = max(1, n_plots) + (2 if has_log else 0)

    if fig is None:
        fig = plt.figure(figsize=(14, 4 * n_rows), facecolor='#1a1a2e')
    else:
        fig.clear()
        fig.set_facecolor('#1a1a2e')
    fig.suptitle(
        f"Training Dashboard  ·  {Path(logdir).name}  ·  {ts}",
        color='white', fontsize=13, y=0.98,
    )

    gs = gridspec.GridSpec(n_rows, 2, figure=fig, hspace=0.45, wspace=0.35)
    ax_idx = 0
    style = dict(facecolor='#16213e')

    def next_ax():
        nonlocal ax_idx
        row, col = divmod(ax_idx, 2)
        ax = fig.add_subplot(gs[row, col], **style)
        ax_idx += 1
        return ax

    def plot_group(tags, title, log_scale=False):
        if not tags:
            return
        ax = next_ax()
        for i, tag in enumerate(tags):
            pts = scalars[tag]
            if not pts:
                continue
            steps, vals = zip(*pts)
            label = tag.split('/')[-1]
            ax.plot(steps, vals, color=COLORS[i % len(COLORS)], lw=1.5,
                    label=label, marker='o' if len(steps) < 30 else None, ms=3)
        ax.set_title(title, color='white', fontsize=10)
        ax.tick_params(colors='#aaaaaa', labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#333355')
        ax.set_facecolor('#0f3460')
        ax.set_xlabel('step', color='#aaaaaa', fontsize=8)
        if log_scale:
            ax.set_yscale('log')
        ax.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white',
                  framealpha=0.7, loc='best')
        ax.grid(True, color='#2a2a4a', linewidth=0.5)

    plot_group(loss_tags, 'Loss', log_scale=False)
    plot_group(acc_tags,  'Accuracy (%)')
    plot_group(lr_tags,   'Learning Rate', log_scale=True)
    plot_group(other_tags, 'Other')

    # Progress summary panel
    if n_plots == 0:
        ax = next_ax()
        ax.text(0.5, 0.5, 'No scalar data yet\n(waiting for first epoch)',
                ha='center', va='center', color='#aaaaaa', fontsize=11,
                transform=ax.transAxes)
        ax.set_facecolor('#0f3460')

    # Job log panel (spans full width)
    if has_log:
        log_row = (ax_idx + 1) // 2
        ax_log = fig.add_subplot(gs[log_row:log_row+2, :], facecolor='#0d0d1a')
        ax_log.set_title('Job stdout (recent)', color='white', fontsize=10)
        ax_log.axis('off')
        text = '\n'.join(job_lines[-40:])
        ax_log.text(0.01, 0.99, text, transform=ax_log.transAxes,
                    va='top', ha='left', fontsize=6.5, color='#00ff88',
                    fontfamily='monospace', wrap=False)

    if save:
        outpath = Path(outdir) / f"dashboard_{Path(logdir).name}.png"
        fig.savefig(outpath, dpi=110, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"[{ts}] Saved → {outpath}")

    if show:
        fig.canvas.draw_idle()
        plt.pause(0.1)
    else:
        plt.close(fig)
        fig = None

    return fig


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--project',  default=None)
    parser.add_argument('--net',      default='HybridNet',
                        choices=['HybridNet', 'CenterDetect', 'KeypointDetect'])
    parser.add_argument('--logdir',   default=None)
    parser.add_argument('--jobdir',   default=None,
                        help='Root of JARVIS project dir (contains projects/)')
    parser.add_argument('--jobid',    default=None, help='LSF job ID for bpeek')
    parser.add_argument('--ssh-host', default='doq@login1')
    parser.add_argument('--interval', type=float, default=30,
                        help='Refresh interval in seconds')
    parser.add_argument('--no-show',  action='store_true',
                        help='Save PNG only, no interactive window')
    parser.add_argument('--outdir',   default='/tmp')
    args = parser.parse_args()

    # Resolve logdir
    logdir = args.logdir
    if logdir is None:
        # Search known roots
        roots = []
        if args.jobdir:
            roots.append(Path(args.jobdir) / 'projects' / args.project)
        elif args.project:
            candidates = [
                Path('/mnt/johnson_lab/doq/JARVIS-HybridNet/projects') / args.project,
                Path.cwd() / 'projects' / args.project,
            ]
            roots = [c for c in candidates if c.exists()]
        if not roots:
            sys.exit("Specify --logdir or --project with a valid project root.")
        logdir = find_latest_logdir(roots[0], args.net)
        if logdir is None:
            sys.exit(f"No log runs found under {roots[0]}/logs/{args.net}/")
        print(f"Using log dir: {logdir}")

    show = not args.no_show
    if show:
        matplotlib.use('TkAgg' if 'DISPLAY' in os.environ else 'Agg')
    else:
        matplotlib.use('Agg')

    print(f"Monitoring {logdir}")
    print(f"Refresh every {args.interval}s  |  Ctrl-C to stop")

    try:
        fig = None
        while True:
            scalars = read_tf_events(logdir)

            job_lines = None
            if args.jobid:
                job_lines = bpeek_tail(args.jobid, args.ssh_host)

            fig = make_dashboard(scalars, logdir,
                                 job_lines=job_lines,
                                 outdir=args.outdir,
                                 save=True,
                                 show=show,
                                 fig=fig)

            total_steps = sum(pts[-1][0] for pts in scalars.values() if pts) if scalars else 0
            print(f"  tags={list(scalars.keys())}  latest_step={total_steps}")

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == '__main__':
    main()
