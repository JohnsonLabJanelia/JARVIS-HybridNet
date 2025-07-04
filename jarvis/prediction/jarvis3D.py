"""
JARVIS-MoCap (https://jarvis-mocap.github.io/jarvis-docs)
Copyright (c) 2022 Timo Hueser.
https://github.com/JARVIS-MoCap/JARVIS-HybridNet
Licensed under GNU Lesser General Public License v2.1
"""

import os

import torch
import torch.nn as nn
from torchvision import transforms

from jarvis.efficienttrack.efficienttrack import EfficientTrack
from jarvis.hybridnet.hybridnet import HybridNet
from jarvis.utils.reprojection import ReprojectionTool


class JarvisPredictor3D(nn.Module):
    def __init__(
        self,
        cfg,
        weights_center_detect="latest",
        weights_hybridnet="latest",
        trt_mode="off",
    ):
        super(JarvisPredictor3D, self).__init__()
        self.cfg = cfg

        self.centerDetect = EfficientTrack(
            "CenterDetectInference", self.cfg, weights_center_detect
        ).model
        self.hybridNet = HybridNet(
            "inference", self.cfg, weights_hybridnet
        ).model

        self.transform_mean = torch.tensor(
            self.cfg.DATASET.MEAN, device=torch.device("cuda")
        ).view(3, 1, 1)
        self.transform_std = torch.tensor(
            self.cfg.DATASET.STD, device=torch.device("cuda")
        ).view(3, 1, 1)
        self.bbox_hw = int(self.cfg.KEYPOINTDETECT.BOUNDING_BOX_SIZE / 2)
        self.num_cameras = self.cfg.HYBRIDNET.NUM_CAMERAS
        self.bounding_box_size = self.cfg.KEYPOINTDETECT.BOUNDING_BOX_SIZE

        self.reproTool = ReprojectionTool()
        self.center_detect_img_size = int(self.cfg.CENTERDETECT.IMAGE_SIZE)

        if trt_mode == "new":
            self.compile_trt_models()

        elif trt_mode == "previous":
            self.load_trt_models()

    def load_trt_models(self):
        # TODO: add try except here!!
        import torch_tensorrt

        transpose2D_lib_dir = os.path.join(
            self.cfg.PARENT_DIR,
            "libs",
            "conv_transpose2d_converter.cpython-39-x86_64-linux-gnu.so",
        )
        transpose3D_lib_dir = os.path.join(
            self.cfg.PARENT_DIR,
            "libs",
            "conv_transpose3d_converter.cpython-39-x86_64-linux-gnu.so",
        )
        torch.ops.load_library(transpose3D_lib_dir)
        torch.ops.load_library(transpose2D_lib_dir)

        trt_path = os.path.join(
            self.cfg.PARENT_DIR,
            "projects",
            self.cfg.PROJECT_NAME,
            "trt-models",
            "predict3D",
        )

        # TODO: Check if files actually exist
        self.centerDetect = torch.jit.load(
            os.path.join(trt_path, "centerDetect.pt")
        )
        self.hybridNet.effTrack = torch.jit.load(
            os.path.join(trt_path, "keypointDetect.pt")
        )
        self.hybridNet.v2vNet = torch.jit.load(
            os.path.join(trt_path, "hybridNet.pt")
        )

    def compile_trt_models(self):
        # TODO: add try except here!!
        import torch_tensorrt

        transpose2D_lib_dir = os.path.join(
            self.cfg.PARENT_DIR,
            "libs",
            "conv_transpose2d_converter.cpython-39-x86_64-linux-gnu.so",
        )
        transpose3D_lib_dir = os.path.join(
            self.cfg.PARENT_DIR,
            "libs",
            "conv_transpose3d_converter.cpython-39-x86_64-linux-gnu.so",
        )
        torch.ops.load_library(transpose3D_lib_dir)
        torch.ops.load_library(transpose2D_lib_dir)

        trt_path = os.path.join(
            self.cfg.PARENT_DIR,
            "projects",
            self.cfg.PROJECT_NAME,
            "trt-models",
            "predict3D",
        )
        os.makedirs(trt_path, exist_ok=True)

        self.centerDetect = self.centerDetect.eval().cuda()
        print("h0")

        traced_model = torch.jit.trace(
            self.centerDetect, [torch.randn((1, 3, 256, 256)).to("cuda")]
        )
        print("h1")

        self.centerDetect = torch_tensorrt.compile(
            traced_model,
            inputs=[
                torch_tensorrt.Input(
                    (
                        self.cfg.HYBRIDNET.NUM_CAMERAS,
                        3,
                        self.cfg.CENTERDETECT.IMAGE_SIZE,
                        self.cfg.CENTERDETECT.IMAGE_SIZE,
                    ),
                    dtype=torch.float,
                )
            ],
            enabled_precisions={torch.half},
        )
        print("h2")

        torch.jit.save(
            self.centerDetect, os.path.join(trt_path, "centerDetect.pt")
        )

        self.hybridNet.effTrack.eval().cuda()
        traced_model = torch.jit.trace(
            self.hybridNet.effTrack,
            [
                torch.randn(
                    (1, 3, self.bounding_box_size, self.bounding_box_size)
                ).to("cuda")
            ],
        )
        self.hybridNet.effTrack = torch_tensorrt.compile(
            traced_model,
            inputs=[
                torch_tensorrt.Input(
                    (
                        self.cfg.HYBRIDNET.NUM_CAMERAS,
                        3,
                        self.bounding_box_size,
                        self.bounding_box_size,
                    ),
                    dtype=torch.float,
                )
            ],
            enabled_precisions={torch.half},
        )
        torch.jit.save(
            self.hybridNet.effTrack,
            os.path.join(trt_path, "keypointDetect.pt"),
        )

        self.hybridNet.v2vNet.eval().cuda()
        grid_size = int(
            self.cfg.HYBRIDNET.ROI_CUBE_SIZE / self.cfg.HYBRIDNET.GRID_SPACING
        )
        traced_model = torch.jit.trace(
            self.hybridNet.v2vNet,
            [
                torch.randn(
                    (
                        1,
                        self.cfg.KEYPOINTDETECT.NUM_JOINTS,
                        grid_size,
                        grid_size,
                        grid_size,
                    )
                ).to("cuda")
            ],
        )
        self.hybridNet.v2vNet = torch_tensorrt.compile(
            traced_model,
            inputs=[
                torch_tensorrt.Input(
                    (
                        1,
                        self.cfg.KEYPOINTDETECT.NUM_JOINTS,
                        grid_size,
                        grid_size,
                        grid_size,
                    ),
                    dtype=torch.float,
                )
            ],
            enabled_precisions={torch.half},
        )
        torch.jit.save(
            self.hybridNet.v2vNet, os.path.join(trt_path, "hybridNet.pt")
        )

    def forward(
        self, imgs, cameraMatrices, intrinsicMatrices, distortionCoefficients
    ):
        self.reproTool.cameraMatrices = cameraMatrices
        self.reproTool.intrinsicMatrices = intrinsicMatrices
        self.reproTool.distortionCoefficients = distortionCoefficients

        # img_size = torch.tensor([imgs.shape[3], imgs.shape[2]],
        #             device = torch.device('cuda'))

        # width, height

        # img_size = imgs[3].shape  # height, width
        # imgs_orig = np.zeros((len(imgs), img_size[1], img_size[0], 3)).astype(
        #     np.uint8
        # )
        # imgs = (
        #     torch.from_numpy(imgs_orig)
        #     .cuda()
        #     .float()
        #     .permute(0, 3, 1, 2)[:, [2, 1, 0]]
        #     / 255.0
        # )

        # this needs to be a vector of [N, 2], rather than [2]
        # downsampling_scale = torch.tensor(
        #     [
        #         imgs.shape[3] / float(self.center_detect_img_size),
        #         imgs.shape[2] / float(self.center_detect_img_size),
        #     ],
        #     device=torch.device("cuda"),
        # ).float()
        # imgs.shape[3] height, imgs.shpe[2] is width

        # imgs_resized = transforms.functional.resize(
        #     imgs, [self.center_detect_img_size, self.center_detect_img_size]
        # )  # do it in a loop for each img, and then stack them together

        img_sizes = torch.zeros(
            len(imgs), 2, dtype=torch.float32, device="cuda"
        )
        downsampling_scale = torch.zeros_like(img_sizes)
        imgs_resized = torch.zeros(
            len(imgs),
            3,
            self.center_detect_img_size,
            self.center_detect_img_size,
            dtype=torch.float32,
            device="cuda",
        )
        img_tensors = []

        for i, img_np in enumerate(imgs):
            h, w = img_np.shape[:2]
            img_sizes[i] = torch.tensor(
                [w, h], dtype=torch.float32, device="cuda"
            )
            downsampling_scale[i] = torch.tensor(
                [w, h], dtype=torch.float32, device="cuda"
            )

            img_tensor = (
                torch.from_numpy(img_np)
                .cuda()
                .float()
                .permute(2, 0, 1)[[2, 1, 0]]  # BGR to RGB
                / 255.0
            )

            img_tensors.append(img_tensor)
            imgs_resized[i] = transforms.functional.resize(
                img_tensor,
                [self.center_detect_img_size, self.center_detect_img_size],
            )

        imgs_resized = (
            imgs_resized - self.transform_mean
        ) / self.transform_std
        outputs = self.centerDetect(imgs_resized)
        heatmaps_gpu = outputs[1].view(
            outputs[1].shape[0], outputs[1].shape[1], -1
        )
        m = heatmaps_gpu.argmax(2).view(
            heatmaps_gpu.shape[0], heatmaps_gpu.shape[1], 1
        )
        preds = torch.cat(
            (m % outputs[1].shape[2], m // outputs[1].shape[3]), dim=2
        )
        maxvals = heatmaps_gpu.gather(2, m)
        num_cams_detect = torch.numel(maxvals[maxvals > 50])
        maxvals = maxvals / 255.0

        # debugging
        # num_cams_detect = 2  # temp for debugging
        # debugging
        if num_cams_detect >= 2:
            center3D = self.reproTool.reconstructPoint(
                (
                    preds.reshape(self.num_cameras, 2)
                    * (downsampling_scale * 2)
                ).transpose(0, 1),
                maxvals,
            )
            centerHMs = self.reproTool.reprojectPoint(center3D.unsqueeze(0))

            # centerHMs[:, 0] = torch.clamp(
            #     centerHMs[:, 0], self.bbox_hw, img_size[0] - self.bbox_hw
            # )
            # centerHMs[:, 1] = torch.clamp(
            #     centerHMs[:, 1], self.bbox_hw, img_size[1] - self.bbox_hw
            # )

            min_val = torch.full_like(
                centerHMs[:, 0], self.bbox_hw, dtype=torch.float32
            )
            centerHMs[:, 0] = torch.clamp(
                centerHMs[:, 0], min_val, img_sizes[:, 0] - self.bbox_hw
            )
            centerHMs[:, 1] = torch.clamp(
                centerHMs[:, 1], min_val, img_sizes[:, 1] - self.bbox_hw
            )
            centerHMs = centerHMs.int()

            imgs_cropped = torch.zeros(
                (
                    self.num_cameras,
                    3,
                    self.bounding_box_size,
                    self.bounding_box_size,
                ),
                device=torch.device("cuda"),
            )

            for i in range(self.num_cameras):
                ## TODO, change here as well
                # imgs_cropped[i] = imgs[
                #     i,
                #     :,
                #     centerHMs[i, 1]
                #     - self.bbox_hw : centerHMs[i, 1]
                #     + self.bbox_hw,
                #     centerHMs[i, 0]
                #     - self.bbox_hw : centerHMs[i, 0]
                #     + self.bbox_hw,
                # ]

                imgs_cropped[i] = img_tensors[i][
                    :,
                    centerHMs[i, 1]
                    - self.bbox_hw : centerHMs[i, 1]
                    + self.bbox_hw,
                    centerHMs[i, 0]
                    - self.bbox_hw : centerHMs[i, 0]
                    + self.bbox_hw,
                ]

            imgs_cropped = (
                imgs_cropped - self.transform_mean
            ) / self.transform_std

            _, _, points3D, confidences = self.hybridNet(
                imgs_cropped.unsqueeze(0),
                [],
                centerHMs.unsqueeze(0),
                center3D.int().unsqueeze(0),
                cameraMatrices.unsqueeze(0),
                intrinsicMatrices.unsqueeze(0),
                distortionCoefficients.unsqueeze(0),
            )
        else:
            points3D = None
            confidences = None
        return points3D, confidences
