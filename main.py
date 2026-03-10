#! /usr/bin/env python3

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

import sys
import numpy as np
import hdf5_dataloader
import networks.radar_transformer
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
import time
import argparse
from pathlib import Path
import csv
import re
torch.set_printoptions(precision=20)
torch.set_default_dtype(torch.float32)
np.set_printoptions(threshold=sys.maxsize)

PLOT = True
SAVE = True

def parse_epoch_from_filename(path):
    match = re.search(r"epoch(\d+)", str(path))
    return int(match.group(1)) if match else 0

def train_one_epoch(dataloader, model, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    losses_per_batch = []
    print("Training RadarTransformer.")
    for batch_index, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        # Compute prediction and loss.
        phi_phi = model(X)
        #phi_phi_scaled = phi_phi*5.0
        loss = criterion(phi_phi, X, y)
        # Backpropagation.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        losses_per_batch.append(loss.item())

    print("Epoch loss average: {:.4f}".format(
        running_loss / len(dataloader)))
    print('-' * 10)
    return running_loss / len(dataloader), losses_per_batch


def evaluate_model(dataloader, model, criterion, device):
    model.eval()
    running_loss = 0.0
    losses_per_batch = []
    print('Evaluating RadarTransformer.')
    for batch_index, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        phi_phi = model(X)
        #phi_phi_scaled = phi_phi*5.0
        loss = criterion(phi_phi, X, y)
        running_loss += loss.item()
        losses_per_batch.append(loss.item())

    print("Val loss average: {:.4f}".format(running_loss / len(dataloader)))
    print('-' * 10)
    return running_loss / len(dataloader), losses_per_batch

def main(args):
    torch.manual_seed(42)
    # Process commandline args.
    model_output_dir_as_string = args.model_output_dir or "saved_models_features"

    # Prepare dataloaders.
    train_data = hdf5_dataloader.HDF5Dataset(
        "./extended_data_features/train/pointclouds.hdf5", "./extended_data_features/train/labels.hdf5")
    val_data = hdf5_dataloader.HDF5Dataset(
        "./extended_data_features/val/pointclouds.hdf5", "./extended_data_features/val/labels.hdf5")

    val_dataloader = DataLoader(
        val_data, batch_size=radar_transformer.MINIBATCH_SIZE, shuffle=True)
    train_dataloader = DataLoader(
        train_data, batch_size=radar_transformer.MINIBATCH_SIZE, shuffle=True)

    model = radar_transformer.RadarDeepMatcher(train_data.get_input_length(), feature_fusion=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for param in model.parameters():
        param.requires_grad = True
    params = [p for p in model.parameters() if p.requires_grad]

    criterion = radar_transformer.Criterion(
        model.num_points_per_pointcloud, device)
    model.to(device)
    criterion.to(device)

    optimizer = torch.optim.Adam(params, lr=0.0001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=200, gamma=0.1)

    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if ckpt_path.exists():
            start_epoch = parse_epoch_from_filename(ckpt_path)
            print(f"Resuming training from checkpoint: {ckpt_path} epoch {start_epoch}")

            checkpoint = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint)
            print("Model weights loaded successfully")
        else:
            print(f"Checkpoint not found at {ckpt_path}, training from scratch")
            start_epoch = 0
    else:
        print("Training from scratch")
        start_epoch = 0

    # Training.
    train_losses = []
    val_losses = []
    for i in range(start_epoch, start_epoch+radar_transformer.NUM_EPOCHS):
        print("Epoch {} / {}".format(i, start_epoch+radar_transformer.NUM_EPOCHS - 1))
        print('-' * 20)
        start = time.time()
        train_loss, train_losses_per_batch = train_one_epoch(
            train_dataloader, model, criterion, optimizer, device)
        train_losses.append(train_loss)
        # train_losses.extend(train_losses_per_batch)
        val_loss, val_losses_per_batch = evaluate_model(
            val_dataloader, model, criterion, device)
        val_losses.append(val_loss)
        # test_losses.extend(test_losses_per_batch)
        lr_scheduler.step()
        stop = time.time()
        print("Time per epoch: {}".format(stop - start))

        if (i + 1) % 5 == 0 :
            ckpt_path = Path(model_output_dir_as_string) / f"RadarTransformer_epoch{i+1}.ptm"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    # Saving model.
    model_output_dir = Path(model_output_dir_as_string + "/" + "RadarTransformer_" +
                            time.strftime("%d%b%Y_%H%M%S") + ".ptm")
    model_output_dir.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_output_dir)
    print(f"Model saved as: {model_output_dir}")

    if PLOT:
        # Plot train/val losses.
        plt.figure()
        plt.plot(train_losses, label="Train")
        plt.plot(val_losses, label="Val")
        plt.legend()
        plt.xlabel("Epochs")
        plt.title("Average losses")
        plt.savefig(str(model_output_dir).split(".ptm")[0] + "_losses.png")
        #plt.show()
    if SAVE:
        with open("losses.csv", 'w', newline='') as losses_file:
            writer = csv.writer(losses_file, quoting=csv.QUOTE_ALL)
            writer.writerow(train_losses)
            writer.writerow(val_losses)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_output_dir",
                        help="Folder where to store the model")
    parser.add_argument("--checkpoint",
                        help="Path to pretrained model (saved_models/RadarTransformer_epoch30.ptm)")
    args = parser.parse_args()
    main(args)
