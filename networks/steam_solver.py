import numpy as np
import torch
import cpp.build.SteamSolver as steamcpp

class SteamSolver:
    """
    Simplified STEAM solver wrapper for use with RadarTransformer correspondences.
    Handles one pair of scans (previous & current).
    """

    def __init__(self, config):
        self.gpuid = config["gpuid"]
        self.window_size = 2

        steam_cfg = config["steam"]
        self.solver_cpp = steamcpp.SteamSolver(steam_cfg["time_step"], self.window_size)
        qc_diag = np.array(steam_cfg["qc_diag"]).reshape(6, 1)
        self.solver_cpp.setQcInv(qc_diag)

        if steam_cfg.get("use_ransac", False):
            self.solver_cpp.useRansac()
            self.solver_cpp.setRansacVersion(steam_cfg.get("ransac_version", 1))
        if steam_cfg.get("use_ctsteam", False):
            self.solver_cpp.useCTSteam()

        T_sv = np.eye(4, dtype=np.float32)
        if "ex_translation_vs_in_s" in steam_cfg:
            T_sv[:3, 3] = steam_cfg["ex_translation_vs_in_s"]
        if "ex_rotation_sv" in steam_cfg:
            T_sv[:3, :3] = np.array(steam_cfg["ex_rotation_sv"]).reshape(3, 3)
        self.solver_cpp.setExtrinsicTsv(T_sv)

        self.solver_cpp.setZeroVelPriorFlag(steam_cfg.get("zero_vel_prior", False))
        self.solver_cpp.setVelPriorFlag(steam_cfg.get("vel_prior", False))

    def optimize(
        self,
        keypoint_coords,  # (1, N, 3)
        pseudo_coords,    # (1, N, 3)
        match_weights,       # (1, N)
        time_tgt, time_src,  # scalar timestamps [1]
        t_ref_tgt, t_ref_src # scalar refs [1]
    ):
        """
        Compute SE(2)/SE(3) relative motion using STEAM between two radar frames.

        Returns:
            R_pred: torch.Tensor [1, 3, 3]
            t_pred: torch.Tensor [1, 3, 1]
        """
        # Convert tensors to numpy
        pts_tgt_3d = keypoint_coords[0].detach().cpu().numpy()   # N×3
        pts_src_3d = pseudo_coords[0].detach().cpu().numpy()     # N×3
        w = match_weights[0].detach().cpu().numpy().reshape(-1, 1)

        #covs = [np.eye(3, dtype=np.float32) * float(max(1e-3, 1 - wi)) for wi in w]
        alpha = 3.0
        covs = [np.eye(3, dtype=np.float32) * np.exp(alpha * (1.0 - float(wi))) for wi in w]

        num_src = pts_src_3d.shape[0]
        num_tgt = pts_tgt_3d.shape[0]

        timestamps1 = np.linspace(t_ref_src, t_ref_tgt, num_src, dtype=np.int64)
        timestamps2 = np.linspace(t_ref_src, t_ref_tgt, num_tgt, dtype=np.int64)
        t_refs = [t_ref_src, t_ref_tgt]

        self.solver_cpp.resetTraj()
        self.solver_cpp.setMeas(
            [pts_tgt_3d.astype(np.float32)],  # list of 1 (N×3)
            [pts_src_3d.astype(np.float32)],  # list of 1 (N×3)
            [np.stack(covs).astype(np.float32)],  # list of 1 (N×3×3)
            [timestamps2], [timestamps1], t_refs
        )

        self.solver_cpp.optimize()

        pose = np.zeros((1, 2, 4, 4), dtype=np.float32)
        self.solver_cpp.getPoses(pose[0])
        T = pose[0, 1]

        R_pred = T[:3, :3]
        t_pred = T[:3, 3:4]

        return (
            torch.from_numpy(R_pred).unsqueeze(0).to(self.gpuid),
            torch.from_numpy(t_pred).unsqueeze(0).to(self.gpuid)
        )
