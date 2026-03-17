#!/usr/bin/env python3
"""
Live training dashboard for JARVIS HybridNet (and other nets).

Usage:
    python tools/monitor_training.py --project mouseJan30 --net HybridNet
    python tools/monitor_training.py \\
        --logdir /mnt/johnson_lab/doq/JARVIS-HybridNet/projects/mouseJan30/logs/HybridNet/Run_20260316-222315 \\
        --jobid 148808346 --interval 300 --no-show --outdir /tmp

Options:
    --project       JARVIS project name (e.g. mouseJan30)
    --net           Network: HybridNet | CenterDetect | KeypointDetect
    --logdir        Explicit path to the TF events directory
    --jobid         LSF job ID to tail via bpeek over SSH
    --ssh-host      SSH host for bpeek (default: doq@login1)
    --interval      Refresh interval in seconds (default: 30)
    --no-show       Save PNG only, no interactive window
    --outdir        Directory to save dashboard PNGs (default: /tmp)
"""

import argparse, os, sys, time, subprocess
from pathlib import Path
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import numpy as np

# ── Palette ───────────────────────────────────────────────────────────────────
BG     = '#0e1117'
PANEL  = '#161b27'
PANEL2 = '#0b0d16'
BORDER = '#252a3a'
TEXT   = '#dde1f0'
DIM    = '#6b7294'
BLUE   = '#4d8ef5'
ORANGE = '#f59b4d'
GREEN  = '#4dcf8e'
PURPLE = '#9b7cf5'
RED    = '#f54d6b'
YELLOW = '#f5d34d'
SERIES = [BLUE, ORANGE, GREEN, PURPLE, RED, YELLOW]
GRID   = '#181d2c'


# ── TF reader ─────────────────────────────────────────────────────────────────
def read_tf_events(logdir):
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator, SCALARS)
    except ImportError:
        sys.exit("pip install tensorboard")
    ea = EventAccumulator(str(logdir), size_guidance={SCALARS: 0})
    ea.Reload()
    return {tag: [(e.step, e.value) for e in ea.Scalars(tag)]
            for tag in ea.Tags().get('scalars', [])}


def find_latest_logdir(project_root, net):
    base = Path(project_root) / 'logs' / net
    if not base.exists():
        return None
    runs = sorted(base.iterdir(), key=lambda p: p.stat().st_mtime)
    return runs[-1] if runs else None


def bpeek_tail(job_id, ssh_host, n=40):
    try:
        r = subprocess.run(['ssh', '-o', 'ConnectTimeout=8', ssh_host,
                            f'bpeek {job_id}'],
                           capture_output=True, text=True, timeout=15)
        lines = [l for l in (r.stdout + r.stderr).splitlines()
                 if '\r' not in l and l.strip()]
        return lines[-n:]
    except Exception as e:
        return [f'[bpeek error: {e}]']


def latest(scalars, tags):
    for t in tags:
        pts = scalars.get(t, [])
        if pts:
            return pts[-1]
    return None


# ── Axis styling ──────────────────────────────────────────────────────────────
def style_ax(ax, title, ylabel='', log_scale=False):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)
    ax.tick_params(colors=DIM, labelsize=8, length=2)
    ax.set_title(title, color=TEXT, fontsize=10, fontweight='bold', pad=8)
    ax.set_xlabel('Epoch', color=DIM, fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, color=DIM, fontsize=8)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if log_scale:
        ax.set_yscale('log')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def plot_series(ax, scalars, tags):
    plotted = False
    for i, tag in enumerate(tags):
        pts = scalars.get(tag, [])
        if not pts:
            continue
        steps, vals = zip(*pts)
        c = SERIES[i % len(SERIES)]
        label = (tag.replace('Train ', '').replace('Validation ', 'Val ')
                    .replace(' Loss', '').strip())
        dashed = 'val' in tag.lower()
        ax.plot(steps, vals, color=c, lw=2.2,
                linestyle='--' if dashed else '-',
                marker='o' if len(steps) <= 15 else None,
                ms=5, markerfacecolor=BG, markeredgewidth=1.5,
                label=label, zorder=3, solid_capstyle='round')
        if not dashed:
            ax.fill_between(steps, vals, alpha=0.08, color=c, zorder=2)
        plotted = True
    if plotted:
        ax.legend(fontsize=8, facecolor=PANEL2, labelcolor=TEXT,
                  framealpha=0.95, edgecolor=BORDER, loc='best')


# ── Dashboard ─────────────────────────────────────────────────────────────────
def make_dashboard(scalars, logdir, job_lines=None,
                   outdir='/tmp', save=True, show=True, fig=None):
    ts  = datetime.now().strftime('%H:%M:%S')
    run = Path(logdir).name

    loss_tags = sorted(t for t in scalars if 'loss' in t.lower())
    acc_tags  = sorted(t for t in scalars if 'acc'  in t.lower())
    lr_tags   = sorted(t for t in scalars if 'lr' in t.lower() or 'learning' in t.lower())
    has_data  = bool(scalars)
    has_log   = bool(job_lines)

    # Row heights (inches): cards | plots | [log]
    row_h = [1.4, 3.8] + ([2.2] if has_log else [])
    fig_h = sum(row_h) + 0.9   # +0.9 for top header
    fig_w = 15

    if fig is None:
        fig = plt.figure(figsize=(fig_w, fig_h), facecolor=BG)
    else:
        fig.clear()
        fig.set_facecolor(BG)
        fig.set_size_inches(fig_w, fig_h)

    # Header
    fig.text(0.5, 1 - 0.12/fig_h, 'JARVIS  Training  Monitor',
             ha='center', va='top', color=TEXT,
             fontsize=16, fontweight='bold', fontfamily='monospace',
             transform=fig.transFigure)
    fig.text(0.5, 1 - 0.52/fig_h, f'{run}    {ts}',
             ha='center', va='top', color=DIM, fontsize=9,
             transform=fig.transFigure)

    # GridSpec — row 0 = cards, row 1 = plots, row 2 = log (optional)
    n_rows = 2 + int(has_log)
    gs = gridspec.GridSpec(
        n_rows, 1, figure=fig,
        height_ratios=row_h,
        top=1 - 0.82/fig_h,
        bottom=0.03,
        left=0.05, right=0.97,
        hspace=0.35,
    )

    # ── Stat cards row ────────────────────────────────────────────────────────
    train_loss_pt = latest(scalars, [t for t in loss_tags if 'train' in t.lower()])
    val_loss_pt   = latest(scalars, [t for t in loss_tags if 'val'   in t.lower()])
    train_acc_pt  = latest(scalars, [t for t in acc_tags  if 'train' in t.lower()])
    val_acc_pt    = latest(scalars, [t for t in acc_tags  if 'val'   in t.lower()])
    lr_pt         = latest(scalars, lr_tags)
    epoch         = train_loss_pt[0] if train_loss_pt else 0

    card_data = [
        ('Epoch',      str(epoch),                                          BLUE),
        ('Train Loss', f"{train_loss_pt[1]:.4f}" if train_loss_pt else '—', BLUE),
        ('Val Loss',   f"{val_loss_pt[1]:.4f}"   if val_loss_pt   else '—', ORANGE),
        ('Train Acc',  f"{train_acc_pt[1]:.1f}%" if train_acc_pt  else '—', GREEN),
        ('Val Acc',    f"{val_acc_pt[1]:.1f}%"   if val_acc_pt    else '—', GREEN),
        ('LR',         f"{lr_pt[1]:.2e}"          if lr_pt         else '—', PURPLE),
    ]

    # Use a nested GridSpec for cards inside row 0
    gs_cards = gridspec.GridSpecFromSubplotSpec(
        1, len(card_data), subplot_spec=gs[0], wspace=0.06)

    for i, (lbl, val, col) in enumerate(card_data):
        ax = fig.add_subplot(gs_cards[i], facecolor=PANEL2)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_edgecolor(BORDER); sp.set_linewidth(0.8)
        # top accent bar
        ax.axhline(y=0.88, color=col, linewidth=3, alpha=0.7, xmin=0.08, xmax=0.92)
        ax.text(0.5, 0.55, val, ha='center', va='center',
                color=col, fontsize=17, fontweight='bold', transform=ax.transAxes)
        ax.text(0.5, 0.18, lbl, ha='center', va='center',
                color=DIM, fontsize=8.5, transform=ax.transAxes)

    # ── Plots row ─────────────────────────────────────────────────────────────
    if has_data:
        gs_plots = gridspec.GridSpecFromSubplotSpec(
            1, 3, subplot_spec=gs[1], wspace=0.30)

        ax_loss = fig.add_subplot(gs_plots[0])
        style_ax(ax_loss, 'Loss', 'loss')
        plot_series(ax_loss, scalars, loss_tags)

        ax_acc = fig.add_subplot(gs_plots[1])
        style_ax(ax_acc, 'Accuracy', '%')
        plot_series(ax_acc, scalars, acc_tags)

        ax_lr = fig.add_subplot(gs_plots[2])
        style_ax(ax_lr, 'Learning Rate', 'lr', log_scale=True)
        plot_series(ax_lr, scalars, lr_tags)
    else:
        ax_w = fig.add_subplot(gs[1], facecolor=PANEL)
        ax_w.axis('off')
        ax_w.text(0.5, 0.55, 'Waiting for first epoch to complete...',
                  ha='center', va='center', color=DIM,
                  fontsize=14, transform=ax_w.transAxes)
        ax_w.text(0.5, 0.38,
                  'TensorBoard scalars are written once per epoch.\n'
                  'Dashboard will populate automatically.',
                  ha='center', va='center', color=DIM, fontsize=9.5,
                  transform=ax_w.transAxes, linespacing=2.0)

    # ── Log row ───────────────────────────────────────────────────────────────
    if has_log:
        ax_log = fig.add_subplot(gs[2], facecolor=PANEL2)
        ax_log.axis('off')
        for sp in ax_log.spines.values():
            sp.set_visible(True); sp.set_edgecolor(BORDER)
        ax_log.text(0.01, 0.97, 'Job stdout',
                    color=DIM, fontsize=8, fontweight='bold',
                    va='top', transform=ax_log.transAxes)
        ax_log.text(0.01, 0.88, '\n'.join(job_lines),
                    va='top', ha='left', fontsize=6.8, color=GREEN,
                    fontfamily='monospace', transform=ax_log.transAxes)

    if save:
        out = Path(outdir) / f"dashboard_{run}.png"
        fig.savefig(out, dpi=120, bbox_inches='tight',
                    facecolor=BG, edgecolor='none')
        print(f"[{ts}] Saved -> {out}")

    if show:
        fig.canvas.draw_idle()
        plt.pause(0.1)
    else:
        plt.close(fig)
        fig = None

    return fig


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--project',  default=None)
    p.add_argument('--net',      default='HybridNet',
                   choices=['HybridNet', 'CenterDetect', 'KeypointDetect'])
    p.add_argument('--logdir',   default=None)
    p.add_argument('--jobdir',   default=None)
    p.add_argument('--jobid',    default=None)
    p.add_argument('--ssh-host', default='doq@login1')
    p.add_argument('--interval', type=float, default=30)
    p.add_argument('--no-show',  action='store_true')
    p.add_argument('--outdir',   default='/tmp')
    args = p.parse_args()

    logdir = args.logdir
    if logdir is None:
        if args.project:
            candidates = [
                Path('/mnt/johnson_lab/doq/JARVIS-HybridNet/projects') / args.project,
                Path.cwd() / 'projects' / args.project,
            ]
            roots = [c for c in candidates if c.exists()]
        elif args.jobdir:
            roots = [Path(args.jobdir) / 'projects' / args.project]
        else:
            sys.exit("Specify --logdir or --project")
        if not roots:
            sys.exit(f"Project root not found")
        logdir = find_latest_logdir(roots[0], args.net)
        if logdir is None:
            sys.exit(f"No runs found")
        print(f"Using: {logdir}")

    show = not args.no_show
    matplotlib.use('TkAgg' if (show and 'DISPLAY' in os.environ) else 'Agg')

    print(f"Monitoring {logdir}")
    print(f"Interval: {args.interval}s  |  Ctrl-C to stop")

    try:
        fig = None
        while True:
            scalars   = read_tf_events(logdir)
            job_lines = bpeek_tail(args.jobid, args.ssh_host) if args.jobid else None
            fig = make_dashboard(scalars, logdir,
                                 job_lines=job_lines, outdir=args.outdir,
                                 save=True, show=show, fig=fig)
            n = sum(len(v) for v in scalars.values())
            print(f"  {len(scalars)} tags, {n} points")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == '__main__':
    main()
