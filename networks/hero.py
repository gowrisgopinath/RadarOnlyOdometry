import torch
import torch.nn as nn
from transformer_models import RadarDeepMatcher
from networks.steam_solver import SteamSolver
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment

class HERO(nn.Module):
    """
    Modified HERO that uses a pretrained RadarTransformer to compute point correspondences
    and passes them into the STEAM solver for trajectory estimation.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gpuid = config['gpuid']

        self.radar_transformer = RadarDeepMatcher(input_size=config['radar_transformer']['input_size'])
        checkpoint = torch.load(config['radar_transformer']['checkpoint'], map_location='cpu')
        self.radar_transformer.load_state_dict(checkpoint)
        self.radar_transformer.to(self.gpuid)
        self.radar_transformer.eval()

        self.solver = SteamSolver(config)

    def forward(self, batch):
        
        pc1 = batch['pc_prev'].to(self.gpuid)
        pc2 = batch['pc_curr'].to(self.gpuid)
        timestamps = batch['timestamps']
        t_ref = batch['t_ref']

        def to_int_list(x):
            out = []
            if isinstance(x, (list, tuple)):
                for e in x:
                    if torch.is_tensor(e):
                        out.append(int(e.item()))
                    else:
                        out.append(int(e))
            elif torch.is_tensor(x):
                out = [int(e.item()) for e in x.flatten()]
            else:
                out = list(map(int, np.array(x).flatten().tolist()))
            return out

        timestamps = to_int_list(timestamps)
        t_ref = to_int_list(t_ref)
        print(timestamps)

        pc1 = pc1.squeeze(0)
        pc2 = pc2.squeeze(0)
        
        X = torch.cat([pc1.flatten(), pc2.flatten()], dim=0).unsqueeze(0).unsqueeze(0)
        print(pc1.shape)

        with torch.no_grad():
            affinity = self.radar_transformer(X).squeeze(0)  # (N, N)

        pc1 = pc1.cpu().numpy()
        pc2 = pc2.cpu().numpy()
        pc1_size = np.max(np.count_nonzero(pc1[1:, :], axis=0), axis=0)
        pc2_size = np.max(np.count_nonzero(pc2[1:, :], axis=0), axis=0)
        pc1_valid = pc1[1:pc1_size, :]
        pc2_valid = pc2[1:pc2_size, :]

        affinity = affinity[1:pc1_size, 1:pc2_size]

        #affinity = torch.sigmoid(affinity)
        '''fig, ax = plt.subplots()
        vmax = np.percentile(affinity, 99)
        vmin = np.percentile(affinity, 1)
        im = ax.imshow(affinity, cmap='jet', vmin=vmin, vmax=vmax)

        fig.colorbar(im)
        ax.set_title(f'Affinity matrix')
        plt.show()'''

        #affinity = torch.softmax(affinity, dim=-1)
        #match_confidence, indices = torch.max(affinity, dim=-1)
        '''match_confidence = torch.relu(match_confidence)  # ensure nonnegative
        max_val = match_confidence.max().clamp(min=1e-6)
        min_val = match_confidence.min()
        match_confidence = (match_confidence - min_val) / (max_val - min_val + 1e-8)'''

        '''valid_mask = match_confidence > 0.5
        idx_pc1 = torch.arange(pc1_valid.shape[0], device=self.gpuid)[valid_mask]
        idx_pc2 = indices[valid_mask]
        match_weights = match_confidence[valid_mask]'''
        affinity = torch.softmax(affinity, dim=-1)
        affinity_np = affinity.detach().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(affinity_np, maximize=True)
        idx_pc1 = torch.as_tensor(row_ind, device=self.gpuid)
        idx_pc2 = torch.as_tensor(col_ind, device=self.gpuid)
        match_weights = affinity[idx_pc1, idx_pc2]

        top_n = 210
        if match_weights.numel() > top_n:
            topk_vals, topk_idx = torch.topk(match_weights, k=top_n)
            idx_pc1 = idx_pc1[topk_idx]
            idx_pc2 = idx_pc2[topk_idx]
            match_weights = topk_vals
        '''topk_vals, topk_idxs = torch.topk(match_confidence, k=min(top_n, match_confidence.numel()))
        idx_pc1 = torch.arange(pc1_valid.shape[0], device=self.gpuid)[topk_idxs]
        idx_pc2 = indices[topk_idxs]
        match_weights = topk_vals'''

        '''max_val = match_weights.max().clamp(min=1e-6)
        min_val = match_weights.min()
        match_weights = (match_weights - min_val) / (max_val - min_val + 1e-8)'''

        if idx_pc1.numel() < 3:
            return {
                'R': torch.eye(3, device=self.gpuid),
                't': torch.zeros(3, 1, device=self.gpuid),
                'match_weights': torch.zeros(1, device=self.gpuid)
            }
        pc1_valid = torch.from_numpy(pc1_valid).to(self.gpuid)
        pc2_valid = torch.from_numpy(pc2_valid).to(self.gpuid)
        keypoint_coords = pc1_valid[idx_pc1][:, :3]
        pseudo_coords = pc2_valid[idx_pc2][:, :3]
        print('keypoint_coords', keypoint_coords)
        print('pseudo_coords', pseudo_coords)
        print('match_weights', match_weights)

        w = match_weights.detach().cpu().numpy().reshape(-1, 1)

        print("Matched points shape:", keypoint_coords.shape)
        print("Weights mean:", float(w.mean()))

        R_pred, t_pred = self.solver.optimize(
            keypoint_coords.unsqueeze(0),
            pseudo_coords.unsqueeze(0),
            torch.from_numpy(w).to(self.gpuid).unsqueeze(0),  # back to tensor
            timestamps[1],
            timestamps[0],
            t_ref[1],
            t_ref[0]
        )

        return {'R': R_pred, 't': t_pred, 'match_weights': torch.from_numpy(w).to(self.gpuid)}
