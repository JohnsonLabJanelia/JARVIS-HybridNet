#!/usr/bin/env python3
"""
Fine-tune HybridNet with bone length loss and cross-view attention.

Usage:
    python tools/train_hybridnet_improved.py --project mouseJan30 --epochs 15
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from jarvis.config.project_manager import ProjectManager
from jarvis.dataset.dataset3D import Dataset3D
from jarvis.hybridnet.hybridnet import HybridNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', required=True)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--bone_weight', type=float, default=0.1)
    parser.add_argument('--cross_view_attention', action='store_true', default=True)
    parser.add_argument('--no_cross_view_attention', dest='cross_view_attention',
                        action='store_false')
    parser.add_argument('--weights', default=None,
                        help='Starting weights (default: latest)')
    parser.add_argument('--mode', default='all',
                        help='Training mode: all, bifpn, last_layers, 3D_only')
    args = parser.parse_args()

    project = ProjectManager()
    project.load(args.project)
    cfg = project.get_cfg()

    # Apply config overrides (keep unfrozen for dataset loading)
    cfg.defrost()
    cfg.HYBRIDNET.BONE_LENGTH_LOSS_WEIGHT = args.bone_weight
    cfg.HYBRIDNET.USE_CROSS_VIEW_ATTENTION = args.cross_view_attention
    cfg.HYBRIDNET.MAX_LEARNING_RATE = args.lr
    cfg.HYBRIDNET.NUM_EPOCHS = args.epochs

    print(f"{'='*60}")
    print(f"  HybridNet Fine-tuning")
    print(f"{'='*60}")
    print(f"  Bone loss weight: {args.bone_weight}")
    print(f"  Cross-view attention: {args.cross_view_attention}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Training mode: {args.mode}")

    # Find best existing weights
    if args.weights is None:
        weights_path = os.path.join(
            cfg.PARENT_DIR, 'projects', cfg.PROJECT_NAME, 'models',
            'HybridNet', 'Run_20260314-232403', 'HybridNet-medium_final.pth')
    else:
        weights_path = args.weights
    print(f"  Starting weights: {weights_path}")

    # Load datasets (cfg must be unfrozen - dataset sets IMAGE_SIZE)
    print(f"\nLoading datasets...")
    training_set = Dataset3D(cfg=cfg, set='train')
    validation_set = Dataset3D(cfg=cfg, set='val')
    print(f"  Train: {len(training_set)} samples")
    print(f"  Val: {len(validation_set)} samples")

    # Build model - load weights first without cross-view attention,
    # then enable it (new params initialized fresh)
    print(f"\nBuilding model...")

    # Temporarily disable CVA to load old weights cleanly
    cfg.HYBRIDNET.USE_CROSS_VIEW_ATTENTION = False

    hybridnet = HybridNet('train', cfg, weights=weights_path)

    # Now enable CVA by adding the module
    if args.cross_view_attention:
        from jarvis.hybridnet.cross_view_attention import CrossViewAttention
        hybridnet.model.reproLayer.use_cross_view_attention = True
        hybridnet.model.reproLayer.cross_view_attn = CrossViewAttention(
            num_cameras=cfg.HYBRIDNET.NUM_CAMERAS,
            num_joints=cfg.KEYPOINTDETECT.NUM_JOINTS
        ).cuda()
        # Re-create optimizer to include new CVA params
        hybridnet.optimizer = __import__('torch').optim.AdamW(
            hybridnet.model.parameters(), cfg.HYBRIDNET.MAX_LEARNING_RATE)
        print(f"  Cross-view attention enabled (5.2K new params)")

    # Re-enable bone loss
    cfg.HYBRIDNET.BONE_LENGTH_LOSS_WEIGHT = args.bone_weight
    cfg.HYBRIDNET.USE_CROSS_VIEW_ATTENTION = args.cross_view_attention
    hybridnet.cfg = cfg

    # Load bone stats
    if args.bone_weight > 0:
        from jarvis.hybridnet.physics_loss import BoneLengthLoss
        bone_stats_path = os.path.join(
            cfg.PARENT_DIR, 'projects', cfg.PROJECT_NAME, 'bone_stats.json')
        if os.path.isfile(bone_stats_path):
            hybridnet.bone_criterion = BoneLengthLoss.from_stats_file(
                bone_stats_path,
                list(cfg.KEYPOINT_NAMES),
                list(cfg.SKELETON)).cuda()
            hybridnet.bone_loss_weight = args.bone_weight
            print(f"  Bone loss loaded (weight={args.bone_weight})")

    hybridnet.set_training_mode(args.mode)

    print(f"\nStarting training...")
    results = hybridnet.train(training_set, validation_set,
                              num_epochs=args.epochs)

    print(f"\n{'='*60}")
    print(f"  Training Complete")
    print(f"{'='*60}")
    print(f"  Final train loss: {results['train_loss']:.4f}")
    print(f"  Final train acc:  {results['train_acc']:.2f} mm")
    print(f"  Final val loss:   {results['val_loss']:.4f}")
    print(f"  Final val acc:    {results['val_acc']:.2f} mm")
    print(f"  Model saved to:   {hybridnet.model_savepath}")


if __name__ == '__main__':
    main()
