"""CoreML export for JARVIS EfficientTrack models.

Converts .pth weights → torch.jit.trace → coremltools → .mlpackage

The .mlpackage files can be loaded by RED's CoreML inference engine
for GPU/ANE-accelerated prediction on Apple Silicon.

Usage:
    conda run -n jarvis python -m jarvis.utils.coreml_export <project_path>
    conda run -n jarvis python -m jarvis.utils.coreml_export <project_path> --output_dir <dir>
"""

import os
import sys
import time
import json
import shutil
import argparse
import torch
import torch.nn as nn
import numpy as np


def _find_latest_weights(models_dir):
    """Find the latest .pth weights file in a JARVIS models subdirectory."""
    import glob
    if not os.path.isdir(models_dir):
        return None
    runs = sorted(glob.glob(os.path.join(models_dir, 'Run_*')))
    if not runs:
        return None
    latest_run = runs[-1]
    finals = glob.glob(os.path.join(latest_run, '*_final.pth'))
    if finals:
        return finals[0]
    pths = sorted(glob.glob(os.path.join(latest_run, '*.pth')))
    return pths[-1] if pths else None


class NormalizedModel(nn.Module):
    """Wraps a model with ImageNet normalization.

    Input: [0, 1] RGB tensor (after CoreML's scale=1/255).
    Output: ImageNet-normalized tensor fed to the backbone.
    """
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        return self.backbone((x - self.mean) / self.std)


def convert_to_mlpackage(model, input_size, output_path, model_name='model'):
    """Convert a traced PyTorch model to CoreML .mlpackage.

    The model is wrapped with ImageNet normalization so CoreML only needs to
    do BGRA→RGB conversion and scale to [0,1]. The normalization runs inside
    the model on GPU/ANE.
    """
    import coremltools as ct

    # Wrap model with ImageNet normalization
    wrapped = NormalizedModel(model)
    wrapped.eval()

    dummy = torch.randn(1, 3, input_size, input_size)
    with torch.no_grad():
        traced = torch.jit.trace(wrapped, dummy)

    t0 = time.time()

    # ImageType input: CoreML handles BGRA→RGB + scale to [0,1]
    # ImageNet normalization is baked into the model via NormalizedModel
    inp = ct.ImageType(
        name='image',
        shape=(1, 3, input_size, input_size),
        color_layout=ct.colorlayout.RGB,
        scale=1.0 / 255.0,
        bias=[0, 0, 0],
    )

    coreml_model = ct.convert(
        traced,
        inputs=[inp],
        convert_to='mlprogram',
        minimum_deployment_target=ct.target.macOS13,
        compute_precision=ct.precision.FLOAT16,
    )
    elapsed = time.time() - t0

    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    coreml_model.save(output_path)

    size = sum(os.path.getsize(os.path.join(dp, f))
               for dp, _, fns in os.walk(output_path) for f in fns)
    return elapsed, size / (1024 * 1024)


def export_project(project_path, output_dir=None):
    """Export CenterDetect + KeypointDetect from a JARVIS project to .mlpackage."""
    import yaml
    from yacs.config import CfgNode
    from jarvis.config.config import _C as default_cfg
    from jarvis.efficienttrack.model import EfficientTrackBackbone

    # Load config
    cfg_path = os.path.join(project_path, 'config.yaml')
    if not os.path.exists(cfg_path):
        print(f'Error: config.yaml not found at {cfg_path}')
        sys.exit(1)

    with open(cfg_path) as f:
        cfg_dict = yaml.safe_load(f)
    cfg = default_cfg.clone()
    cfg.merge_from_other_cfg(CfgNode(cfg_dict))

    model_size = cfg.CENTERDETECT.MODEL_SIZE
    num_joints = cfg.KEYPOINTDETECT.NUM_JOINTS
    cd_input_size = cfg.CENTERDETECT.IMAGE_SIZE
    kd_input_size = cfg.KEYPOINTDETECT.BOUNDING_BOX_SIZE
    project_name = os.path.basename(project_path)

    # Find weights
    cd_weights = _find_latest_weights(os.path.join(project_path, 'models', 'CenterDetect'))
    kd_weights = _find_latest_weights(os.path.join(project_path, 'models', 'KeypointDetect'))

    if not cd_weights:
        print('Error: No CenterDetect weights found')
        sys.exit(1)
    if not kd_weights:
        print('Error: No KeypointDetect weights found')
        sys.exit(1)

    if output_dir is None:
        output_dir = os.path.join(project_path, 'models', 'onnx')
    os.makedirs(output_dir, exist_ok=True)

    print(f'Exporting JARVIS models to CoreML .mlpackage...')
    print(f'  Project: {project_name}')
    print(f'  Output:  {output_dir}')

    # CenterDetect
    print(f'\nCenterDetect ({cd_input_size}x{cd_input_size}):')
    print(f'  Weights: {cd_weights}')
    cd_model = EfficientTrackBackbone(
        cfg.CENTERDETECT, model_size=model_size, output_channels=1)
    cd_model.load_state_dict(torch.load(cd_weights, map_location='cpu'))
    cd_model.eval()
    cd_path = os.path.join(output_dir, 'center_detect.mlpackage')
    cd_time, cd_mb = convert_to_mlpackage(cd_model, cd_input_size, cd_path, 'CenterDetect')
    print(f'  Exported in {cd_time:.1f}s ({cd_mb:.1f} MB)')

    # KeypointDetect
    print(f'\nKeypointDetect ({kd_input_size}x{kd_input_size}, {num_joints} joints):')
    print(f'  Weights: {kd_weights}')
    kd_model = EfficientTrackBackbone(
        cfg.KEYPOINTDETECT, model_size=model_size, output_channels=num_joints)
    kd_model.load_state_dict(torch.load(kd_weights, map_location='cpu'))
    kd_model.eval()
    kd_path = os.path.join(output_dir, 'keypoint_detect.mlpackage')
    kd_time, kd_mb = convert_to_mlpackage(kd_model, kd_input_size, kd_path, 'KeypointDetect')
    print(f'  Exported in {kd_time:.1f}s ({kd_mb:.1f} MB)')

    # Update model_info.json
    meta_path = os.path.join(output_dir, 'model_info.json')
    metadata = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            metadata = json.load(f)

    metadata.update({
        'project_name': project_name,
        'num_joints': num_joints,
        'center_detect': {
            **metadata.get('center_detect', {}),
            'input_size': cd_input_size,
            'model_size': model_size,
            'mlpackage_file': 'center_detect.mlpackage',
        },
        'keypoint_detect': {
            **metadata.get('keypoint_detect', {}),
            'input_size': kd_input_size,
            'model_size': model_size,
            'num_joints': num_joints,
            'mlpackage_file': 'keypoint_detect.mlpackage',
        },
        'coreml_info': {
            'format': 'mlprogram',
            'precision': 'float16',
            'input_color_layout': 'RGB',
            'input_scale': 1.0 / 255.0,
            'imagenet_normalization': 'baked into model',
        },
    })
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f'\nMetadata: {meta_path}')
    print(f'Done!')
    return output_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export JARVIS models to CoreML .mlpackage')
    parser.add_argument('project_path', help='Path to JARVIS project directory')
    parser.add_argument('--output_dir', default=None,
                        help='Output directory (default: project/models/onnx/)')
    args = parser.parse_args()
    export_project(args.project_path, args.output_dir)
