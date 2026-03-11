"""ONNX export utilities for JARVIS EfficientTrack models.

Exports CenterDetect and KeypointDetect models to ONNX format for
inference in RED (C++ / ONNX Runtime).

Handles the InstanceNorm2d numerical divergence issue by replacing
all InstanceNorm2d modules with manual mean/var computation before
export. This produces numerically faithful ONNX models.

Usage (standalone):
    python -m jarvis.utils.onnx_export <project_name> [--output_dir <dir>]

Usage (from training pipeline):
    from jarvis.utils.onnx_export import export_to_onnx
    export_to_onnx(model, input_size, output_path, num_joints)
"""

import os
import copy
import torch
import torch.nn as nn


class ManualInstanceNorm2d(nn.Module):
    """Drop-in replacement for nn.InstanceNorm2d that uses explicit
    mean/var arithmetic instead of the InstanceNormalization ONNX op.

    ONNX Runtime's InstanceNormalization op produces incorrect results
    when channels have zero variance (which happens in trained BiFPN
    networks where some attention weights are zero). This replacement
    avoids that op entirely.
    """
    def __init__(self, num_features, eps=1e-5, affine=False):
        super().__init__()
        self.eps = eps
        self.affine = affine
        self.num_features = num_features
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        # x: (N, C, H, W)
        mean = x.mean(dim=(2, 3), keepdim=True)
        var = x.var(dim=(2, 3), keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        if self.affine:
            x = x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return x


def replace_instance_norm(model):
    """Replace all nn.InstanceNorm2d modules with ManualInstanceNorm2d.

    This must be done before ONNX export to avoid numerical divergence
    in ONNX Runtime's InstanceNormalization op.
    """
    for name, module in model.named_children():
        if isinstance(module, nn.InstanceNorm2d):
            replacement = ManualInstanceNorm2d(
                module.num_features,
                eps=module.eps,
                affine=module.affine
            )
            if module.affine and module.weight is not None:
                replacement.weight.data = module.weight.data.clone()
                replacement.bias.data = module.bias.data.clone()
            setattr(model, name, replacement)
        else:
            replace_instance_norm(module)


def export_to_onnx(model, input_size, output_path, num_joints=None,
                   opset_version=17, device='cpu'):
    """Export an EfficientTrack model to ONNX format.

    Args:
        model: EfficientTrackBackbone instance (already loaded with weights)
        input_size: int, spatial input size (e.g., 320 for CenterDetect)
        output_path: str, path to write the .onnx file
        num_joints: int, number of output channels (for metadata only)
        opset_version: int, ONNX opset version (17 recommended)
        device: str, device for tracing ('cpu' recommended for export)
    """
    model = copy.deepcopy(model)
    model.to(device)
    model.eval()

    # Replace InstanceNorm2d with manual implementation
    replace_instance_norm(model)

    # Disable memory-efficient Swish if present (EfficientNet compatibility)
    for m in model.modules():
        if hasattr(m, 'set_swish'):
            m.set_swish(memory_efficient=False)

    # Create dummy input
    dummy_input = torch.randn(1, 3, input_size, input_size, device=device)

    # Export
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['image'],
        output_names=['heatmap_low', 'heatmap_high'],
        opset_version=opset_version,
        do_constant_folding=True,
    )

    # Verify the export
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f'  Exported: {output_path} ({size_mb:.1f} MB, opset {opset_version})')
    if num_joints:
        print(f'  Joints: {num_joints}, Input: {input_size}x{input_size}')


def _find_latest_weights(models_dir):
    """Find the latest .pth weights file in a JARVIS models subdirectory."""
    import glob
    if not os.path.isdir(models_dir):
        return None
    # Find all run directories, sort by name (timestamp-based)
    runs = sorted(glob.glob(os.path.join(models_dir, 'Run_*')))
    if not runs:
        return None
    latest_run = runs[-1]
    # Find final.pth in the latest run
    finals = glob.glob(os.path.join(latest_run, '*_final.pth'))
    if finals:
        return finals[0]
    # Fallback: highest epoch checkpoint
    pths = sorted(glob.glob(os.path.join(latest_run, '*.pth')))
    return pths[-1] if pths else None


def export_project(project_path, output_dir=None, weights_cd='latest',
                   weights_kd='latest'):
    """Export CenterDetect and KeypointDetect from a JARVIS project.

    Args:
        project_path: str, path to JARVIS project directory (containing config.yaml)
                      OR a project name (resolved via ProjectManager)
        output_dir: str, directory to write ONNX files (default: project/models/onnx/)
        weights_cd: str, CenterDetect weights ('latest' or path to .pth)
        weights_kd: str, KeypointDetect weights ('latest' or path to .pth)
    """
    from jarvis.config.project_manager import ProjectManager
    from jarvis.efficienttrack.efficienttrack import EfficientTrack

    # Support both project name (via ProjectManager) and direct path
    if os.path.isdir(project_path) and os.path.exists(os.path.join(project_path, 'config.yaml')):
        # Direct path to project directory
        from yacs.config import CfgNode
        import yaml
        with open(os.path.join(project_path, 'config.yaml')) as f:
            cfg_dict = yaml.safe_load(f)
        from jarvis.config.config import _C as default_cfg
        cfg = default_cfg.clone()
        cfg.merge_from_other_cfg(CfgNode(cfg_dict))
        cfg.PARENT_DIR = os.path.dirname(project_path)
        project_name = os.path.basename(project_path)

        # Resolve weight paths
        if weights_cd == 'latest':
            cd_dir = os.path.join(project_path, 'models', 'CenterDetect')
            weights_cd = _find_latest_weights(cd_dir)
        if weights_kd == 'latest':
            kd_dir = os.path.join(project_path, 'models', 'KeypointDetect')
            weights_kd = _find_latest_weights(kd_dir)
    else:
        # Project name via ProjectManager
        project = ProjectManager()
        project.load(project_path)
        cfg = project.cfg
        project_name = project_path

    if output_dir is None:
        output_dir = os.path.join(project_path if os.path.isdir(project_path)
                                  else '.', 'models', 'onnx')
    os.makedirs(output_dir, exist_ok=True)

    print(f'Exporting JARVIS models to ONNX...')
    print(f'  Project: {project_name}')
    print(f'  Output:  {output_dir}')

    # Build models directly from config + weights (avoids ProjectManager dependency)
    from jarvis.efficienttrack.model import EfficientTrackBackbone

    cd_input_size = cfg.CENTERDETECT.IMAGE_SIZE
    kd_input_size = cfg.KEYPOINTDETECT.BOUNDING_BOX_SIZE
    num_joints = cfg.KEYPOINTDETECT.NUM_JOINTS

    # CenterDetect
    print(f'\nCenterDetect:')
    print(f'  Weights: {weights_cd}')
    cd_model = EfficientTrackBackbone(
        cfg.CENTERDETECT, model_size=cfg.CENTERDETECT.MODEL_SIZE,
        output_channels=1)
    if weights_cd and os.path.exists(weights_cd):
        cd_model.load_state_dict(torch.load(weights_cd, map_location='cpu'))
    export_to_onnx(
        cd_model, cd_input_size,
        os.path.join(output_dir, 'center_detect.onnx'),
        num_joints=1)

    # KeypointDetect
    print(f'\nKeypointDetect:')
    print(f'  Weights: {weights_kd}')
    kd_model = EfficientTrackBackbone(
        cfg.KEYPOINTDETECT, model_size=cfg.KEYPOINTDETECT.MODEL_SIZE,
        output_channels=num_joints)
    if weights_kd and os.path.exists(weights_kd):
        kd_model.load_state_dict(torch.load(weights_kd, map_location='cpu'))
    export_to_onnx(
        kd_model, kd_input_size,
        os.path.join(output_dir, 'keypoint_detect.onnx'),
        num_joints=num_joints)

    # Write metadata (for RED to read when loading these models)
    import json
    metadata = {
        'project_name': project_name,
        'num_joints': num_joints,
        'center_detect': {
            'input_size': cd_input_size,
            'model_size': cfg.CENTERDETECT.MODEL_SIZE,
            'onnx_file': 'center_detect.onnx',
        },
        'keypoint_detect': {
            'input_size': kd_input_size,
            'model_size': cfg.KEYPOINTDETECT.MODEL_SIZE,
            'num_joints': num_joints,
            'onnx_file': 'keypoint_detect.onnx',
        },
    }
    # Try to include skeleton info from config
    try:
        if hasattr(cfg, 'KEYPOINTDETECT'):
            if hasattr(cfg.KEYPOINTDETECT, 'KEYPOINT_NAMES'):
                metadata['joint_names'] = list(cfg.KEYPOINTDETECT.KEYPOINT_NAMES)
            if hasattr(cfg.KEYPOINTDETECT, 'SKELETON'):
                metadata['skeleton'] = list(cfg.KEYPOINTDETECT.SKELETON)
    except:
        pass
    meta_path = os.path.join(output_dir, 'model_info.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f'\nMetadata: {meta_path}')
    print(f'Done!')

    return output_dir


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Export JARVIS models to ONNX')
    parser.add_argument('project_name', help='JARVIS project name')
    parser.add_argument('--output_dir', default=None,
                        help='Output directory (default: project/models/onnx/)')
    parser.add_argument('--weights_cd', default='latest',
                        help='CenterDetect weights (default: latest)')
    parser.add_argument('--weights_kd', default='latest',
                        help='KeypointDetect weights (default: latest)')
    args = parser.parse_args()

    export_project(args.project_name, args.output_dir,
                   args.weights_cd, args.weights_kd)
