# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

import numpy as np
import torch
from typing import Dict, Optional, Sequence

from embodichain.lab.gym.envs import EmbodiedEnv, EmbodiedEnvCfg
from embodichain.lab.gym.utils.registration import register_env
from embodichain.utils import logger
from embodichain.lab.sim.cfg import MarkerCfg

from embodichain.lab.gym.envs.tasks.tableware.base_agent_env import BaseAgentEnv
from .action_bank import (
    ClickBellActionBank,
)


from pathlib import Path

from embodichain.utils.utility import load_config
from robosynchallenge.data.constants import ROBOSYNCHALLENGE_ROOT


__all__ = ["ClickBellEnv", "ClickBellTestEnv", "ClickBellAgentEnv"]



@register_env("ClickBell", max_episode_steps=600)
class ClickBellEnv(EmbodiedEnv):

    def __init__(self, cfg: EmbodiedEnvCfg = None, **kwargs):
        super().__init__(cfg, **kwargs)

        action_config = kwargs.get("action_config", None)

        # 默认保持官方的单动作配置模式。
        self.auto_select_arm = False
        self.arm_action_configs = {}
        self.action_config = action_config

        if (
            isinstance(action_config, dict)
            and action_config.get("mode") == "auto_arm"
        ):
            # 自动选手配置只保存左右动作配置的路径，实际动作图在这里分别加载。
            left_path = Path(action_config["left_config_path"]).expanduser()
            right_path = Path(action_config["right_config_path"]).expanduser()

            # 相对路径统一以 RoboSynChallenge 仓库根目录为基准，避免受启动目录影响。
            if not left_path.is_absolute():
                left_path = ROBOSYNCHALLENGE_ROOT / left_path
            if not right_path.is_absolute():
                right_path = ROBOSYNCHALLENGE_ROOT / right_path

            self.arm_action_configs = {
                "left_arm": load_config(str(left_path)),
                "right_arm": load_config(str(right_path)),
            }
            self.auto_select_arm = True

            # 自动模式将在按钮随机完成后选择其中一个配置，不能把包装配置传给 ActionBank。
            self.action_config = None

            logger.log_info(
                "Loaded left and right click-bell action configs for auto arm selection."
            )

        self._button_pressed = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

    @staticmethod
    def _get_pose_xy(pose) -> np.ndarray:
        """从 4×4 位姿矩阵中读取 xy 坐标。"""

        if isinstance(pose, torch.Tensor):
            pose = pose.detach().cpu().numpy()

        pose = np.asarray(pose)

        # 兼容 [1, 4, 4] 和 [4, 4] 两种位姿数据形状。
        if pose.ndim == 3:
            pose = pose[0]

        return pose[:2, 3]

    def select_press_arm(self) -> str:
        """选择距离按钮更近的手臂。

        公共可达区域采用确定性规则：
        - 哪只手臂基座距离按钮更近，就使用哪只手臂；
        - 两边距离完全相同时，固定选择左臂；
        - 不进行随机选手，避免产生冲突训练标签。
        """

        button_xy = self._get_pose_xy(
            self.affordance_datas["button_pose"]
        )
        left_base_xy = self._get_pose_xy(
            self.affordance_datas["left_arm_base_pose"]
        )
        right_base_xy = self._get_pose_xy(
            self.affordance_datas["right_arm_base_pose"]
        )

        left_distance = np.linalg.norm(button_xy - left_base_xy)
        right_distance = np.linalg.norm(button_xy - right_base_xy)

        # 左臂基座位于正 y、右臂基座位于负 y，因此分界线自然接近 y=0。
        if left_distance <= right_distance:
            return "left_arm"

        return "right_arm"

    def create_demo_action_list(self, *args, **kwargs):
        """根据按钮位置选择左右手，并生成专家动作。"""

        logger.log_info("Create demo action list for ClickBellTask.")

        # 官方单手配置继续使用原来的执行方式。
        if not self.auto_select_arm:
            if self.action_config is None:
                logger.log_error(
                    "No action_config found, please check the configuration."
                )

            self._init_action_bank(
                ClickBellActionBank,
                self.action_config,
            )
            return self.create_expert_demo_action_list(**kwargs)

        # 按钮已经在 reset 阶段随机完成，此时再选择手臂。
        preferred_arm = self.select_press_arm()

        # 首选手臂失败时，尝试另一只手臂。
        fallback_arm = (
            "right_arm" if preferred_arm == "left_arm" else "left_arm"
        )

        for arm_name in (preferred_arm, fallback_arm):
            logger.log_info(f"Trying to click bell with {arm_name}.")

            self._init_action_bank(
                ClickBellActionBank,
                self.arm_action_configs[arm_name],
            )

            action_list = self.create_expert_demo_action_list(**kwargs)

            if action_list is not None:
                # 记录本条示范实际使用的手臂，便于检查数据。
                self.selected_press_arm = arm_name

                logger.log_info(
                    f"Selected press arm: {arm_name}.",
                    color="green",
                )
                return action_list

            logger.log_warning(
                f"Failed to generate trajectory for {arm_name}."
            )

        # 两只手都不能生成有效轨迹，让采集程序重新随机按钮。
        logger.log_warning(
            "Neither arm can reach the button. Resetting episode."
        )
        return None

    def create_expert_demo_action_list(self, **kwargs):
        """
        Create an expert demonstration action list using the action bank.

        This function generates a trajectory based on expert knowledge, mapping joint and end-effector
        states to the required action format for the environment and robot type.

        Args:
            **kwargs: Additional keyword arguments.

        Returns:
            list: A list of actions, each containing joint positions ("qpos").
        """

        if hasattr(self, "action_bank") is False or self.action_bank is None:
            logger.log_error(
                "Action bank is not initialized. Cannot create expert demo action list."
            )

        ret = self.action_bank.create_action_list(
            self, self.graph_compose, self.packages
        )

        if ret is None:
            logger.log_warning("Failed to generate expert demo action list.")
            return None

        # TODO: to be removed, need a unified interface in robot class
        left_arm_joints = self.robot.get_joint_ids(name="left_arm", remove_mimic=True)
        right_arm_joints = self.robot.get_joint_ids(name="right_arm", remove_mimic=True)
        left_eef_joints = self.robot.get_joint_ids(name="left_eef", remove_mimic=True)
        right_eef_joints = self.robot.get_joint_ids(name="right_eef", remove_mimic=True)


        total_traj_num = ret[list(ret.keys())[0]].shape[-1]
        num_active_joints = len(self.active_joint_ids)
        actions = torch.zeros(
            (total_traj_num, self.num_envs, num_active_joints), dtype=torch.float32
        )

        # 建立一个从全局 joint_id 到 active_joint_id 在 action 数组中正确存放位置的映射
        global_to_active_idx = {
            joint_id: active_idx for active_idx, joint_id in enumerate(self.active_joint_ids)
        }

        for key, joints in [
            ("left_arm", left_arm_joints),
            ("left_eef", left_eef_joints),
            ("right_arm", right_arm_joints),
            ("right_eef", right_eef_joints),
        ]:
            if key in ret:
                # TODO: only 1 env supported now
                local_action_data = torch.as_tensor(ret[key].T, dtype=torch.float32)

                # 【修改重点2】：使用映射精准定位它在 action tensor 中的正确位置存放
                for i, joint_id in enumerate(joints):
                    if joint_id in global_to_active_idx:
                        active_idx = global_to_active_idx[joint_id]
                        actions[:, 0, active_idx] = local_action_data[:, i]
        return actions
    def compute_task_state(self, **kwargs):
        button = self.sim.get_articulation("button")
        button_qpos = button.get_qpos()

        # button.urdf uses a single prismatic joint with range [-0.005, 0.0].
        # Treat any detectable displacement as success (with tiny epsilon to avoid numerical noise).
        press_depth = -button_qpos[:, 0]
        movement_threshold = 0.0048
        success = press_depth >= movement_threshold
        # print(f"press_depth: {press_depth}, movement_threshold: {movement_threshold}")
        self._button_pressed |= success
        fail = torch.zeros_like(success, dtype=torch.bool)
        success = torch.zeros_like(fail, dtype=torch.bool)
        return success, fail, {}

    def is_task_success(self, **kwargs) -> torch.Tensor:
        return self._button_pressed
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        obs, info = super().reset(seed=seed, options=options)

        if options is None:
            options = {}
        reset_ids = options.get(
            "reset_ids",
            torch.arange(self.num_envs, dtype=torch.int32, device=self.device),
        )
        self._button_pressed[reset_ids] = False

        return obs, info

@register_env("ClickBellTest", max_episode_steps=600)
class ClickBellTestEnv(ClickBellEnv):
    def compute_task_state(self, **kwargs):
        button = self.sim.get_articulation("button")
        button_qpos = button.get_qpos()

        # button.urdf uses a single prismatic joint with range [-0.005, 0.0].
        # Treat any detectable displacement as success (with tiny epsilon to avoid numerical noise).
        press_depth = -button_qpos[:, 0]
        movement_threshold = 0.004
        success = press_depth >= movement_threshold
        # print(f"press_depth: {press_depth}, movement_threshold: {movement_threshold}")
        self._button_pressed |= success
        fail = torch.zeros_like(success, dtype=torch.bool)

        return success, fail, {}
    def is_task_success(self, **kwargs) -> torch.Tensor:
        return torch.ones_like(self._button_pressed, dtype=torch.bool)

@register_env("ClickBellAgent", max_episode_steps=600)
class ClickBellAgentEnv(BaseAgentEnv, ClickBellEnv):
    def __init__(self, cfg: EmbodiedEnvCfg = None, **kwargs):
        super().__init__(cfg, **kwargs)
        super()._init_agents(**kwargs)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        obs, info = super().reset(seed=seed, options=options)
        super().get_states()
        return obs, info
