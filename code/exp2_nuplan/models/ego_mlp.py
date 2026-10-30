"""Ego-status MLP baseline for experiment 2, as a nuPlan TorchModuleWrapper.

Input: only the ego vehicle's own past states. The model uses nuPlan's AgentsFeatureBuilder ("agents" feature) and
reads only its ego part: the rear-axle pose (x, y, heading) of the present and `past_trajectory_sampling.num_poses`
past frames, expressed in the current ego frame. The other agents in that feature are ignored and no map feature is
built, so, like the ego-status baselines of AD-MLP (Zhai et al., 2023) and NAVSIM's EgoStatusMLP, the network sees
neither the map nor other road users. With the default sampling (4 past poses over 1.5 s, as in nuPlan's
simple_vector_model) the input is exactly the ego branch of the vector baseline.

Output: a nuPlan Trajectory of `future_trajectory_sampling.num_poses` poses (x, y, heading) in the current ego frame
(16 poses at 0.5 s = 8 s), supervised by nuPlan's EgoTrajectoryTargetBuilder. In simulation it runs through nuPlan's
MLPlanner (planner=ml_planner) like any other nuPlan model.
"""
from typing import List

import torch
from torch import nn

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.modeling.types import FeaturesType, TargetsType
from nuplan.planning.training.preprocessing.feature_builders.agents_feature_builder import AgentsFeatureBuilder
from nuplan.planning.training.preprocessing.features.agents import Agents
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
from nuplan.planning.training.preprocessing.target_builders.ego_trajectory_target_builder import (
    EgoTrajectoryTargetBuilder,
)


class EgoStatusMLP(TorchModuleWrapper):
    """MLP from the ego's past poses to its future trajectory (no map, no agents)."""

    def __init__(
        self,
        past_trajectory_sampling: TrajectorySampling,
        future_trajectory_sampling: TrajectorySampling,
        hidden_size: int = 512,
        num_hidden_layers: int = 3,
    ) -> None:
        """
        :param past_trajectory_sampling: sampling of the ego history (num_poses past poses + the present one).
        :param future_trajectory_sampling: sampling of the predicted trajectory.
        :param hidden_size: width of the hidden layers.
        :param num_hidden_layers: number of hidden Linear+ReLU layers.
        """
        super().__init__(
            feature_builders=[AgentsFeatureBuilder(past_trajectory_sampling)],
            target_builders=[EgoTrajectoryTargetBuilder(future_trajectory_sampling)],
            future_trajectory_sampling=future_trajectory_sampling,
        )
        self._num_frames = past_trajectory_sampling.num_poses + 1
        input_size = self._num_frames * Agents.ego_state_dim()
        output_size = future_trajectory_sampling.num_poses * Trajectory.state_size()

        layers: List[nn.Module] = []
        width = input_size
        for _ in range(num_hidden_layers):
            layers += [nn.Linear(width, hidden_size), nn.ReLU()]
            width = hidden_size
        layers.append(nn.Linear(width, output_size))
        self.mlp = nn.Sequential(*layers)

    def forward(self, features: FeaturesType) -> TargetsType:
        """
        :param features: {"agents": Agents}; only Agents.ego (list of [num_frames, 3] tensors) is used.
        :return: {"trajectory": Trajectory} with data of shape [batch, num_poses, 3].
        """
        agents: Agents = features["agents"]
        ego = torch.stack([e.float() for e in agents.ego])  # [batch, num_frames, 3]
        batch_size = ego.shape[0]
        predictions = self.mlp(ego.reshape(batch_size, -1))
        return {"trajectory": Trajectory(data=predictions.view(batch_size, -1, Trajectory.state_size()))}
