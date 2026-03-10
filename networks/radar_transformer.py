# Copyright (C) 2024 Jan Michalczyk, Control of Networked Systems, University
# of Klagenfurt, Austria.
#
# All rights reserved.
#
# This software is licensed under the terms of the BSD-2-Clause-License with
# no commercial use allowed, the full terms of which are made available
# in the LICENSE file. No license in patents is granted.
#
# You can contact the author at <jan.michalczyk@aau.at>

from torch import nn
import torch
import numpy as np
import sys
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
np.set_printoptions(threshold=sys.maxsize)

EMBEDDINGS_DIM = 128
NUM_EPOCHS = 30
MINIBATCH_SIZE = 8
FEATURES_PER_POINT = 5

class PointNet(nn.Module):
    def __init__(self, emb_dims=EMBEDDINGS_DIM):
        super(PointNet, self).__init__()
        self.conv1 = nn.Conv1d(FEATURES_PER_POINT, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(emb_dims)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        return x

class PointNetFusion(nn.Module):
    def __init__(self, emb_dims=EMBEDDINGS_DIM):
        super(PointNetFusion, self).__init__()
        # x,y,z branch
        self.loc_conv1 = nn.Conv1d(3, 4, kernel_size=1, bias=False)
        self.loc_conv2 = nn.Conv1d(4, 16, kernel_size=1, bias=False)
        self.loc_conv3 = nn.Conv1d(16, 64, kernel_size=1, bias=False)

        # rcs branch
        self.rcs_conv1 = nn.Conv1d(1, 4, kernel_size=1, bias=False)
        self.rcs_conv2 = nn.Conv1d(4, 16, kernel_size=1, bias=False)
        self.rcs_conv3 = nn.Conv1d(16, 64, kernel_size=1, bias=False)

        # doppler branch
        self.vel_conv1 = nn.Conv1d(1, 4, kernel_size=1, bias=False)
        self.vel_conv2 = nn.Conv1d(4, 16, kernel_size=1, bias=False)
        self.vel_conv3 = nn.Conv1d(16, 64, kernel_size=1, bias=False)

        self.bn_loc1, self.bn_loc2, self.bn_loc3 = nn.BatchNorm1d(4), nn.BatchNorm1d(16), nn.BatchNorm1d(64)
        self.bn_rcs1, self.bn_rcs2, self.bn_rcs3 = nn.BatchNorm1d(4), nn.BatchNorm1d(16), nn.BatchNorm1d(64)
        self.bn_vel1, self.bn_vel2, self.bn_vel3 = nn.BatchNorm1d(4), nn.BatchNorm1d(16), nn.BatchNorm1d(64)

        self.global_conv1 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.bn_global1 = nn.BatchNorm1d(128)
        self.global_conv2 = nn.Conv1d(128, emb_dims, kernel_size=1, bias=False)
        self.bn_global2 = nn.BatchNorm1d(emb_dims)

    def forward(self, x):
        l = x[:, 0:3, :]
        r = x[:, 3:4, :]
        v = x[:, 4:5, :]

        f1_l = F.relu(self.bn_loc1(self.loc_conv1(l)))
        f1_r = F.relu(self.bn_rcs1(self.rcs_conv1(r)))
        f1_v = F.relu(self.bn_vel1(self.vel_conv1(v)))
        f1_pooled = torch.mean(torch.stack([f1_l, f1_r, f1_v], dim=1), dim=1)

        f2_l = F.relu(self.bn_loc2(self.loc_conv2(f1_pooled)))
        f2_r = F.relu(self.bn_rcs2(self.rcs_conv2(f1_r)))
        f2_v = F.relu(self.bn_vel2(self.vel_conv2(f1_v)))
        f2_pooled = torch.mean(torch.stack([f2_l, f2_r, f2_v], dim=1), dim=1)

        f3_l = F.relu(self.bn_loc3(self.loc_conv3(f2_pooled)))
        f3_r = F.relu(self.bn_rcs3(self.rcs_conv3(f2_r)))
        f3_v = F.relu(self.bn_vel3(self.vel_conv3(f2_v)))
        PF = torch.mean(torch.stack([f3_l, f3_r, f3_v], dim=1), dim=1)

        gf1 = F.relu(self.bn_global1(self.global_conv1(PF)))
        gf2 = F.relu(self.bn_global2(self.global_conv2(gf1)))
        return gf2

class Criterion(nn.Module):
    def __init__(self, num_points_per_pointcloud, device):
        super().__init__()
        self.half_input = num_points_per_pointcloud
        self.loss_fn = nn.CrossEntropyLoss()
        self.device = device

    def _get_loss(self, matches_indices_list, phi_phi):
        target = torch.zeros(
            phi_phi.shape[0], phi_phi.shape[1], dtype=torch.int64, device=self.device)
        for batch_idx, matches in enumerate(matches_indices_list):
            # Assign column indices to target at rows indices locations.
            target[batch_idx, matches[0]] = matches[1]
        loss = self.loss_fn(phi_phi, target)
        return loss

    def _calculate_matches_indices(self, X, y, pc1_size, pc2_size):
        points_indices = []
        for batch_idx in range(X.shape[0]):
            # Costs is a matrix where rows are points from the first transformed pointcloud
            # and cols are points from the second pointcloud.
            # Account for padded zero at the begining of each pointcloud.
            costs = torch.cdist(y[batch_idx, :pc1_size[batch_idx], :],
                                X[batch_idx, (self.half_input + 1):(self.half_input + 1 + pc2_size[batch_idx]), :], p=2)
            with torch.no_grad():
                row_ind, col_ind = linear_sum_assignment(
                    costs.cpu())
                idx_to_keep = np.ones(row_ind.shape, dtype=bool)
                # Check the euclidean cost and remove if above the threshold.
                for idx, (row, col) in enumerate(zip(row_ind.tolist(), col_ind.tolist())):
                    # 0.5 best.
                    if costs[row, col] > 0.6:
                        idx_to_keep[idx] = False
            row_ind = torch.as_tensor(
                row_ind[idx_to_keep], dtype=torch.int64, device=self.device)
            col_ind = torch.as_tensor(col_ind[idx_to_keep],
                                      dtype=torch.int64, device=self.device)
            # increment all entries.
            points_indices.append(torch.vstack((row_ind + 1, col_ind + 1)))
        return points_indices

    def _calculate_non_padded_pc_sizes(self, X):
        pc1_size, _ = torch.max(torch.count_nonzero(
            X[:, :self.half_input, :], dim=1), axis=1)
        pc2_size, _ = torch.max(torch.count_nonzero(
            X[:, self.half_input:, :], dim=1), axis=1)
        return pc1_size, pc2_size

    def _get_gt_matches(self, X, y):
        pc1_size, pc2_size = self._calculate_non_padded_pc_sizes(X)
        matches_indices = self._calculate_matches_indices(
            X, y, pc1_size, pc2_size)
        return matches_indices

    def _reshape_into_inputs(self, X, y):
        old_X_shape = X.shape
        X = torch.reshape(
            X, (old_X_shape[0], 2 * self.half_input, FEATURES_PER_POINT))
        old_y_shape = y.shape
        y = torch.reshape(
            y, (old_y_shape[0], self.half_input - 1, FEATURES_PER_POINT))
        return X, y

    def forward(self, phi_phi, X, y):
        # Reshape inputs.
        X, y = self._reshape_into_inputs(X, y)
        matches_indices_list = self._get_gt_matches(X, y)
        # In matches_indices the row are indexes of points in second pointcloud
        # and cols are indices of points in the transformed first pointcloud (GT).
        loss = self._get_loss(matches_indices_list, phi_phi)
        return loss


class DoubleTransformerBlock(nn.Module):
    def __init__(self):
        super(DoubleTransformerBlock, self).__init__()
        # Encoder.
        self.tranformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=EMBEDDINGS_DIM, nhead=4, dropout=0.3)
        self.transformer_encoder = nn.TransformerEncoder(
            self.tranformer_encoder_layer, num_layers=1)
        # Decoder.
        self.transformer_decoder_layer = nn.TransformerDecoderLayer(
            d_model=EMBEDDINGS_DIM, nhead=4, dropout=0.3)
        self.transformer_decoder = nn.TransformerDecoder(
            self.transformer_decoder_layer, num_layers=1)

    def forward(self, embeddings_pc1, embeddings_pc2):
        encoder_output = self.transformer_encoder(embeddings_pc1)
        decoder_output = self.transformer_decoder(
            embeddings_pc2, encoder_output)
        return decoder_output


class RadarDeepMatcher(nn.Module):
    def __init__(self, input_size, feature_fusion=False):
        super(RadarDeepMatcher, self).__init__()
        if feature_fusion:
            self.embedding_layer = PointNetFusion()
        else:
            self.embedding_layer = PointNet()
        # input_size is two concatenated pointclouds zero-padded. 1 accounts for the
        # "non-matched" class.
        self.input_size = input_size
        self.num_entries_in_point = FEATURES_PER_POINT
        self.num_points = self.input_size // self.num_entries_in_point
        # MLP encoder layer no 1 - size num_points_in_both_pc2 x num_entries_in_point.
        self.num_points_per_pointcloud = self.num_points // 2
        self.num_points_per_pointcloud_squared = self.num_points_per_pointcloud * \
            self.num_points_per_pointcloud
        self.real_num_points_per_pointclouds = self.num_points_per_pointcloud - 2
        self.first_layer_mlp = nn.Sequential(
            nn.Linear(self.num_entries_in_point,
                      EMBEDDINGS_DIM),
            nn.ReLU()
        )
        # Transformers.
        self.transformer_pc1 = DoubleTransformerBlock()
        self.transformer_pc2 = DoubleTransformerBlock()

    def forward(self, X):
        """X is a batch of examples.
        X = [[[ example1 ]],
             [[ example2 ]],
               ... ]]].
            (batch_size, 1, input_size).
        """
        # 1. Reshape the input to fit the first layer.
        old_shape = X.shape
        X = torch.reshape(
            X, (old_shape[0], self.num_points, self.num_entries_in_point))
        X = X.transpose(1, 2).contiguous()
        # output - [batch_size, num_points, embeddings_size] - embeddings.
        output = self.embedding_layer(X)
        # output = self.first_layer_mlp(X)
        output = output.transpose(1, 2).contiguous()
        # 2. Two transformer modules taking as inputs embedded pc1 and pc2.
        half_input = int(output.shape[1] / 2)
        phi_pc1 = self.transformer_pc1(
            output[:, :half_input, :], output[:, half_input:, :])
        phi_pc2 = self.transformer_pc2(
            output[:, half_input:, :], output[:, :half_input, :])
        # 3. Sum the transformer outputs with the original embeddings.
        phi_pc1 = phi_pc1 + output[:, :half_input, :]
        phi_pc2 = phi_pc2 + output[:, half_input:, :]
        # 4. Build the affinities matrix.
        #phi_pc1 = F.normalize(phi_pc1, p=2, dim=-1)
        #phi_pc2 = F.normalize(phi_pc2, p=2, dim=-1)
        phi_phi = torch.bmm(
            phi_pc1, torch.transpose(phi_pc2, 1, 2))
        #phi_phi = (phi_phi + 1.0) / 2.0 # restricting entries of affinity matrix to be in [0,1] range
        return phi_phi
