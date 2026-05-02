"""
JARVIS-MoCap (https://jarvis-mocap.github.io/jarvis-docs)
Copyright (c) 2022 Timo Hueser.
https://github.com/JARVIS-MoCap/JARVIS-HybridNet
Licensed under GNU Lesser General Public License v2.1
"""

import os, sys, inspect
import ruamel.yaml
import json
import shutil
from yacs.config import CfgNode as CN
import numpy as np
import streamlit as st


from jarvis.config import cfg
from jarvis.dataset.dataset2D import Dataset2D
from jarvis.dataset.dataset3D import Dataset3D
from jarvis.utils.utils import CLIColors
import jarvis.utils.clp as clp


class ProjectManager:
    """
    Project Manager Class to load and setup projects and find suitable values
    for network parameters.
    """

    def __init__(self):
        self.cfg = None
        current_dir = os.path.dirname(
            os.path.abspath(inspect.getfile(inspect.currentframe()))
        )
        self.parent_dir = os.path.dirname(os.path.dirname(current_dir))

    def load(self, project_name):
        """
        Load an existing project.

        :param project_name: Name of the project
        :type project_name: string
        """
        self.cfg = cfg
        self.cfg.PROJECT_NAME = project_name
        if not (
            os.path.isfile(
                os.path.join(
                    self.parent_dir,
                    self.cfg.PROJECTS_ROOT_PATH,
                    project_name,
                    "config.yaml",
                )
            )
        ):
            clp.error(
                "Project does not exist, change name or create new "
                "Project by calling create_new(...)."
            )
            self.cfg = None
            return False

        cfg.merge_from_file(
            os.path.join(
                self.parent_dir,
                self.cfg.PROJECTS_ROOT_PATH,
                project_name,
                "config.yaml",
            )
        )
        self.cfg.logPaths = CN()
        self.cfg.savePaths = CN()
        for module in ["CenterDetect", "KeypointDetect", "HybridNet"]:
            model_savepath = os.path.join(
                self.parent_dir,
                self.cfg.PROJECTS_ROOT_PATH,
                project_name,
                "models",
                module,
            )
            log_path = os.path.join(
                self.parent_dir,
                self.cfg.PROJECTS_ROOT_PATH,
                project_name,
                "logs",
                module,
            )
            self.cfg.savePaths[module] = model_savepath
            self.cfg.logPaths[module] = log_path
        self.cfg.PARENT_DIR = self.parent_dir
        clp.success(f"Successfully loaded project {project_name}.")
        return True

    def create_new(self, name, dataset2D_path, dataset3D_path=None):
        """
        Create a new project. This sets up the project directory structure and
        initializes the config file values obtained by analyzing the training
        sets provided.

        :param name: Name of the new project
        :type name: string
        :param dataset2D_path: Path to the dataset that is used to train the
                               EffDet cropping and the EffTrack 2D tracking
                               network. This does not have to be the same
                               Dataset that is used for final 3D training.
        :type dataset2D_path: string
        :param dataset3D_path: path to the dataset that is used to train the
                               full 3D network. If None, the config will not try
                               to setup 3D Network parameters.
        :type dataset3D_path: string

        """
        self.cfg = cfg
        if os.path.isfile(
            os.path.join(
                self.parent_dir,
                self.cfg.PROJECTS_ROOT_PATH,
                name,
                "config.yaml",
            )
        ):
            clp.error(
                "Project already exist, change name or delete old " "project."
            )
            self.cfg = None
            return False

        if not os.path.isdir(dataset2D_path):
            clp.error("Dataset2D directory does not exist. Aborting...")
            return False
        if dataset3D_path != None:
            if not os.path.isdir(dataset3D_path):
                clp.error("Dataset3D directory does not exist. Aborting...")
                return False

        self.cfg.PROJECT_NAME = name
        self.cfg.DATASET.DATASET_2D = dataset2D_path
        self.cfg.DATASET.DATASET_3D = dataset3D_path
        os.makedirs(
            os.path.join(self.parent_dir, cfg.PROJECTS_ROOT_PATH, name),
            exist_ok=True,
        )

        self.cfg.logPaths = CN()
        self.cfg.savePaths = CN()

        for module in ["CenterDetect", "KeypointDetect", "HybridNet"]:
            model_savepath = os.path.join(
                self.parent_dir,
                self.cfg.PROJECTS_ROOT_PATH,
                name,
                "models",
                module,
            )
            log_path = os.path.join(
                self.parent_dir,
                self.cfg.PROJECTS_ROOT_PATH,
                name,
                "logs",
                module,
            )
            self.cfg.savePaths[module] = model_savepath
            self.cfg.logPaths[module] = log_path
            os.makedirs(log_path, exist_ok=True)
            os.makedirs(model_savepath, exist_ok=True)
        self._init_dataset2D()
        if dataset3D_path != None:
            self._init_dataset3D()
        self._init_config(name)
        clp.success(f"Project {name} created succesfully.")
        return True

    def get_create_config_interactive(
        self, name, dataset2D_path, dataset3D_path=None
    ):
        st.session_state["creating_project"] = True
        self.cfg = cfg
        self.cfg.PROJECT_NAME = name
        self.cfg.DATASET.DATASET_2D = dataset2D_path
        self.cfg.DATASET.DATASET_3D = dataset3D_path

        if os.path.isfile(
            os.path.join(
                self.parent_dir,
                self.cfg.PROJECTS_ROOT_PATH,
                name,
                "config.yaml",
            )
        ):
            st.error("Project already exists, please choose a different name.")
            return
        if dataset3D_path != None:
            if not os.path.isdir(dataset3D_path):
                st.error("Dataset3D directory does not exist..")
                return
        if not os.path.isdir(dataset2D_path):
            st.error("Dataset2D directory does not exist..")
            return

        dataset2D = Dataset2D(self.cfg, set="train", mode="keypoints")
        suggested_bbox_size = dataset2D.get_dataset_config()
        if dataset3D_path != None:
            dataset3D = Dataset3D(self.cfg, set="train")
            suggestions = dataset3D.get_dataset_config()
        st.title("Project Configuration")
        with st.form("config_form"):
            bbox_size = st.number_input(
                "2D bounding box size (has to be " "divisible by 64):",
                value=suggested_bbox_size,
                min_value=64,
                step=64,
            )
            if dataset3D_path != None:
                sigma_default = suggestions.get("gt_sigma_mm")
                if sigma_default is None:
                    sigma_default = max(1.0, 0.05 * suggestions["bbox"])
                gt_sigma_mm = st.number_input(
                    "GT Gaussian sigma (mm) — auto-estimated from "
                    "triangulation residuals; ~15-25% of target size is "
                    "a good rule of thumb. Set to 0 to keep legacy "
                    "behavior.",
                    value=float(sigma_default),
                    min_value=0.0,
                    step=0.5,
                )
                # Couple grid suggestion to sigma (sigma_voxels ~= 1.5).
                if gt_sigma_mm > 0:
                    grid_from_sigma = max(1, int(round(gt_sigma_mm / 3.0)))
                else:
                    grid_from_sigma = 1
                grid_from_bbox = max(1, suggestions["resolution"])
                grid_default = max(grid_from_sigma, grid_from_bbox)
                grid_spacing = st.number_input(
                    "Grid spacing:",
                    value=grid_default,
                    min_value=1,
                    max_value=1024,
                    step=1,
                )
                bbox_size_3D = st.number_input(
                    "3D tracking Volume size "
                    "(has to be divisible by 4*grid_spacing):",
                    value=suggestions["bbox"],
                    min_value=4,
                    step=4,
                )
            submitted2 = st.form_submit_button("Confirm")
        if submitted2:
            # Auto-round bboxes up to satisfy divisibility, since the form
            # widgets don't re-render their defaults when the user changes
            # grid_spacing. Rounding up preserves the data's extent.
            if bbox_size % 64 != 0:
                bbox_size = int(np.ceil(bbox_size / 64.0)) * 64
                st.info(
                    f"2D bounding box size rounded up to {bbox_size} "
                    f"(next multiple of 64)."
                )
            if dataset3D_path != None:
                if bbox_size_3D % (4 * grid_spacing) != 0:
                    divisor = 4 * grid_spacing
                    bbox_size_3D = (
                        int(np.ceil(bbox_size_3D / divisor)) * divisor
                    )
                    st.info(
                        f"3D bounding box size rounded up to "
                        f"{bbox_size_3D} (next multiple of {divisor})."
                    )
                if grid_spacing > bbox_size_3D:
                    st.error(
                        "Grid spacing can not be bigger than " "bounding box."
                    )
                    return
            os.makedirs(
                os.path.join(self.parent_dir, cfg.PROJECTS_ROOT_PATH, name),
                exist_ok=True,
            )
            self.cfg.KEYPOINTDETECT.BOUNDING_BOX_SIZE = bbox_size
            self.cfg.KEYPOINTDETECT.NUM_JOINTS = dataset2D.num_keypoints[0]
            if dataset3D_path != None:
                self.cfg.HYBRIDNET.ROI_CUBE_SIZE = bbox_size_3D
                self.cfg.HYBRIDNET.GRID_SPACING = grid_spacing
                self.cfg.HYBRIDNET.NUM_CAMERAS = dataset3D.num_cameras
                self.cfg.HYBRIDNET.GT_SIGMA_MM = (
                    None if gt_sigma_mm == 0 else float(gt_sigma_mm)
                )
            self.cfg.logPaths = CN()
            self.cfg.savePaths = CN()
            for module in ["CenterDetect", "KeypointDetect", "HybridNet"]:
                model_savepath = os.path.join(
                    self.parent_dir,
                    self.cfg.PROJECTS_ROOT_PATH,
                    name,
                    "models",
                    module,
                )
                log_path = os.path.join(
                    self.parent_dir,
                    self.cfg.PROJECTS_ROOT_PATH,
                    name,
                    "logs",
                    module,
                )
                self.cfg.savePaths[module] = model_savepath
                self.cfg.logPaths[module] = log_path
                os.makedirs(log_path, exist_ok=True)
                os.makedirs(model_savepath, exist_ok=True)
            self._init_config(name)
            st.session_state["creating_project"] = False
            st.session_state["created_project"] = name
            st.experimental_rerun()

    def get_cfg(self):
        """
        Get configuration handle for the configuration of the current project.
        """
        if self.cfg == None:
            print(
                "No Project loaded yet! Call either load(...) or "
                "create_new(...)."
            )
        return self.cfg

    def get_projects(self):
        return os.listdir(os.path.join(self.parent_dir, "projects"))

    def _get_number_from_user(self, question, default, div=None, bounds=None):
        """
        Get a valid number divisible by div and in range bounds from the user.
        :param question: Question to ask if no selected
        :type question: string
        :param default: Default Value, returned if user selects yes
        :type default: int
        :param div: Number input should be divisble by. Not used if None
        :type div: int
        :param div: Range for input number
        :type bounds: list of two ints. Not used if None
        """
        number = default
        if div == None:
            div = 1
        valid_accepts = ["yes", "Yes", "y", "Y"]
        valid_declines = ["no", "No", "n", "N"]
        got_valid_answer = False
        while not got_valid_answer:
            ans = input()
            if ans in valid_declines:
                got_valid_answer = True
                print(question)
                ans_is_valid = False
                while not ans_is_valid:
                    ans = input()
                    if ans.isdigit() and int(ans) % div == 0:
                        if (
                            bounds == None
                            or int(ans) >= bounds[0]
                            and int(ans) <= bounds[1]
                        ):
                            number = int(ans)
                            ans_is_valid = True
                        else:
                            print(
                                f"Please enter a Number between {bounds[0]} "
                                f"and {bounds[1]}!"
                            )
                    else:
                        print(f"Please enter a Number divisible by {div}!")
            elif ans in valid_accepts:
                got_valid_answer = True
            else:
                print("Please enter either yes or no!")
        return number

    def _init_dataset2D(self):
        dataset2D = Dataset2D(self.cfg, set="train", mode="keypoints")
        suggested_bbox_size = dataset2D.get_dataset_config()
        print("KeypointDetect 2D Configuration:")
        print(
            f"Use suggested Bounding Box size of {suggested_bbox_size} "
            "pixels? (yes/no)"
        )
        q = "Enter custom Bounding Box size, make sure it is divisible by 64:"
        bbox_size = suggested_bbox_size
        bbox_size = self._get_number_from_user(q, suggested_bbox_size, 64)

        # CenterDetect input resolution. Default 320 works for most setups;
        # bump to 448 (or higher) if your animal is small in the frame —
        # red3d2jarvis.py prints a data-driven suggestion at export time.
        cd_default = self.cfg.CENTERDETECT.IMAGE_SIZE
        print(
            f"\nCenterDetect 2D Configuration:"
        )
        print(
            f"Use CenterDetect IMAGE_SIZE of {cd_default}? (yes/no)"
        )
        q = "Enter custom CenterDetect IMAGE_SIZE, make sure it is divisible by 64:"
        cd_image_size = self._get_number_from_user(q, cd_default, 64)

        self.cfg.KEYPOINTDETECT.BOUNDING_BOX_SIZE = bbox_size
        self.cfg.KEYPOINTDETECT.NUM_JOINTS = dataset2D.num_keypoints[0]
        self.cfg.CENTERDETECT.IMAGE_SIZE = cd_image_size
        # Default eval cadence — emit val metrics every 5 epochs.
        self.cfg.CENTERDETECT.VAL_INTERVAL = 5
        self.cfg.KEYPOINTDETECT.VAL_INTERVAL = 5

    def _init_dataset3D(self):
        print("HybridNet 3D Configuration:")
        dataset3D = Dataset3D(self.cfg, set="train")
        suggestions = dataset3D.get_dataset_config()
        bbox_size = suggestions["bbox"]

        # Ask sigma first — it informs the grid choice.
        sigma_suggestion = suggestions.get("gt_sigma_mm")
        sigma_mm = self._prompt_gt_sigma(sigma_suggestion, bbox_size)

        # Suggest grid spacing coupled to sigma. Target sigma_voxels ~= 1.5
        # (each heatmap voxel spans grid_spacing*2 mm, so this gives a
        # well-resolved Gaussian peak). Floor by the legacy bbox-based
        # heuristic so very small-sigma datasets don't get sub-mm grids.
        if sigma_mm is not None and sigma_mm > 0:
            grid_from_sigma = max(1, int(np.round(sigma_mm / 3.0)))
        else:
            grid_from_sigma = 1
        grid_from_bbox = max(1, int(np.round(bbox_size / 85.0)))
        resolution_suggestion = max(grid_from_sigma, grid_from_bbox)

        print(
            f"Use suggested grid spacing of {resolution_suggestion} "
            "mm? (yes/no)"
        )
        q = "Enter custom grid spacing:"
        resolution = self._get_number_from_user(
            q, resolution_suggestion, bounds=[0, 10]
        )
        # Round UP to the next multiple of 4*resolution, matching the
        # behavior of get_dataset_config(). Rounding down here can shrink
        # the bbox below the data's extent if bbox isn't already divisible.
        suggestion_bbox = (
            int(np.ceil(bbox_size / (resolution * 4))) * resolution * 4
        )
        print(
            f"Use suggested 3D Bounding Box size of {suggestion_bbox} "
            "mm? (yes/no)"
        )
        q = (
            f"Enter custom 3D Bounding Box size, make sure it is divisible "
            f"by {resolution*4}:"
        )
        bbox_size = self._get_number_from_user(
            q, suggestion_bbox, resolution * 4
        )

        self.cfg.HYBRIDNET.ROI_CUBE_SIZE = bbox_size
        self.cfg.HYBRIDNET.GRID_SPACING = resolution
        self.cfg.HYBRIDNET.NUM_CAMERAS = dataset3D.num_cameras
        self.cfg.HYBRIDNET.GT_SIGMA_MM = sigma_mm
        # Tiny batch (1) at high LR (default 0.003) destabilizes HybridNet
        # training. Drop to 0.001 by default.
        self.cfg.HYBRIDNET.MAX_LEARNING_RATE = 0.001
        self.cfg.HYBRIDNET.VAL_INTERVAL = 5
        self.cfg.HYBRIDNET.NUM_EPOCHS = 100

    def _prompt_gt_sigma(self, sigma_suggestion, bbox_size):
        """
        Prompt the user for the GT Gaussian sigma (mm). If the auto-estimate
        from leave-one-out triangulation is unavailable, falls back to a
        bbox-fraction heuristic. Returns None to keep the legacy behavior.
        """
        if sigma_suggestion is None:
            sigma_suggestion = max(1.0, 0.05 * bbox_size)
            print(
                f"GT Gaussian sigma: heuristic suggestion "
                f"{sigma_suggestion:.1f} mm (5% of cube size). Use this? "
                f"(yes/no/legacy — 'legacy' keeps the old voxel-coupled "
                f"default.)"
            )
        else:
            print(
                f"GT Gaussian sigma: heuristic suggestion "
                f"{sigma_suggestion:.2f} mm (10% of cube size; pick "
                f"~15-25% of physical target size if you know it). "
                f"Use this? (yes/no/legacy)"
            )
        valid_accepts = ["yes", "Yes", "y", "Y", ""]
        valid_declines = ["no", "No", "n", "N"]
        while True:
            ans = input()
            if ans.strip().lower() == "legacy":
                return None
            if ans in valid_accepts:
                return float(sigma_suggestion)
            if ans in valid_declines:
                print("Enter custom GT sigma in mm (e.g. 5.0):")
                while True:
                    raw = input()
                    try:
                        val = float(raw)
                        if val > 0:
                            return val
                    except ValueError:
                        pass
                    print("Please enter a positive number.")
            print("Please enter yes / no / legacy.")

    def _init_config(self, name):
        config_path = os.path.join(
            self.parent_dir, cfg.PROJECTS_ROOT_PATH, name, "config.yaml"
        )
        shutil.copyfile(
            os.path.join(
                self.parent_dir, "jarvis/config/config_template.yaml"
            ),
            config_path,
        )
        with open(config_path, "r") as stream:
            config_data = ruamel.yaml.load(
                stream, Loader=ruamel.yaml.RoundTripLoader
            )
            self._update_values(config_data, self.cfg)

            if self.cfg.DATASET.DATASET_3D != None:
                dataset_dir = os.path.join(
                    self.cfg.PARENT_DIR,
                    self.cfg.DATASET.DATASET_ROOT_DIR,
                    self.cfg.DATASET.DATASET_3D,
                )
            else:
                dataset_dir = os.path.join(
                    self.cfg.PARENT_DIR,
                    self.cfg.DATASET.DATASET_ROOT_DIR,
                    self.cfg.DATASET.DATASET_2D,
                )
            dataset_json = open(
                os.path.join(dataset_dir, "annotations", "instances_val.json")
            )
            dataset_data = json.load(dataset_json)
            try:
                keypoints = dataset_data["keypoint_names"]
                config_data["KEYPOINT_NAMES"] = keypoints
                skeleton = []
                for component in dataset_data["skeleton"]:
                    skeleton.append(
                        [component["keypointA"], component["keypointB"]]
                    )
                config_data["SKELETON"] = skeleton
            except:
                print("No keypoint names or skeleton defined in this dataset!")

        with open(config_path, "w") as outfile:
            ruamel.yaml.dump(
                config_data, outfile, Dumper=ruamel.yaml.RoundTripDumper
            )

    def _update_values(self, config_dict, cfg):
        for k, v in config_dict.items():
            if isinstance(v, dict):
                self._update_values(v, cfg[k])
            else:
                try:
                    config_dict[k] = cfg[k]
                except:
                    print(k, v)
