"""PLUTO without post-processing (exp2 variant; pluto@b9964b6 has no config switch for this).

PlutoPlanner._run_planning_once (official setting) does, after the network forward pass:
  1. _trim_candidates: keep the top-k (candidate_max_num=20) candidates by learned probability, softmax them,
     append the reference-free trajectory with fixed probability 0.25;
  2. TrajectoryEvaluator.evaluate (src/post_processing/trajectory_evaluator.py): LQR/bicycle forward simulation of
     every candidate and PDM-style rule-based scoring against the network's agent predictions
     (multiplicative: no at-fault collision, drivable area, driving direction; weighted: progress 5, TTC 5,
     speed limit 2, comfort 2);
  3. final score = rule_based_score + learning_based_score_weight (0.3 in config/planner/pluto_planner.yaml)
     * learned probability, argmax;
  4. EmergencyBrake.brake_if_emergency (src/post_processing/emergency_brake.py): if the chosen candidate's
     time-to-at-fault-collision <= max(v/2.4 + 0.5, 3.0) s and v <= 8 m/s, output a braking trajectory instead.

This subclass turns all four off: it outputs the network's own `output_trajectory`, i.e. the candidate with the
highest learned probability over (reference line x mode) with padded reference lines masked, or the
reference-free trajectory when no reference line exists. Feature building, model and checkpoint are unchanged.

Use with pluto's own planner config:  planner=pluto_planner
  planner.pluto_planner._target_=exp2tools.pluto_no_post.PlutoPlannerNoPost
(requires the pluto repo root on PYTHONPATH, because pluto's code is the top-level package `src`).
"""
import numpy as np
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory

from src.planners.ml_planner_utils import global_trajectory_to_states
from src.planners.pluto_planner import PlutoPlanner


class PlutoPlannerNoPost(PlutoPlanner):
    """PlutoPlanner that returns the highest-probability network trajectory without rule-based post-processing."""

    def _run_planning_once(self, current_input):
        ego_state = current_input.history.ego_states[-1]

        planner_feature = self._planner_feature_builder.get_features_from_simulation(current_input, self._initialization)
        planner_feature_torch = planner_feature.collate([planner_feature.to_feature_tensor()]).to_device(self.device)
        out = self._planner.forward(planner_feature_torch.data)

        # (80, 3) x, y, heading in the ego (rear-axle) frame
        traj = out["output_trajectory"][0].cpu().numpy().astype(np.float64)

        # ego frame -> global frame (same transform as PlutoPlanner._trim_candidates)
        origin = ego_state.rear_axle.array
        angle = ego_state.rear_axle.heading
        rot_mat = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
        traj[..., :2] = np.matmul(traj[..., :2], rot_mat) + origin
        traj[..., 2] += angle

        return InterpolatedTrajectory(
            global_trajectory_to_states(
                global_trajectory=traj,
                ego_history=current_input.history.ego_states,
                future_horizon=len(traj) * self._step_interval,
                step_interval=self._step_interval,
                include_ego_state=False,
            )
        )
