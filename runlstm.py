#!/usr/bin/env python
# coding: utf-8

# In[2]:


from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch import optim
import torch

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import metrics

import random
import math
from typing import Tuple, List, Sequence
import warnings

import os

import numpy as np
import pandas as pd
import tensorflow as tf
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

matplotlib.rcParams['figure.figsize'] = (10, 5)

# Set random seed for reproducibility
seed = 42
round_interv = 0.75
random.seed(a=seed)
np.random.seed(seed=seed)
torch.manual_seed(seed=seed)

# Set compute device to GPU if available or CPU otherwise
device = ("cuda" if torch.cuda.is_available() else "cpu")
# Define columns of interest
dfs = pd.read_excel("Ultimate_sheets.xlsx", sheet_name=None) # We set sheet_name to None so that pd loads all sheets

# Add the sheet number to each dataframe
for key in dfs.keys():
    dfs[key]["Sheet"] = key

# Loaded sheets are stored in a dict, so we need to concat them
data = pd.concat(objs=dfs.values(), ignore_index=True)


cols_to_keep = ["Winner", "Player1_Damaged%", "Player2_Damaged%","Player_1_Max","Player_2_Max",	"Match_Progress","Round_Progression", "Sheet"]

data = data[cols_to_keep]
def standardization(df, column_name):
    mean_value = df[column_name].mean()
    std_value = df[column_name].std()
    df[column_name] = (df[column_name] - mean_value) / std_value

# Apply standardization to 'Salary' column
standardization(data, 'Player_1_Max')
standardization(data, 'Player_2_Max')
# Get indices of round starts in the data.
# We assume that we are dealing with a new round whenever Round_Progression is equal to 0
round_start_idx = np.where(data["Round_Progression"] == 0)[0]

# Add the length of the dataset as the last index
# so that we can properly iterate over all rounds
round_start_idx = np.append(arr=round_start_idx, values=data.shape[0])
round_start_idx = round_start_idx[:-1]


def get_round_slices(
        data: pd.DataFrame,
        round_frac: float) -> Tuple[
            List[np.ndarray],
            List[List[int]],
            np.ndarray[int],
            int,
            List[str]
        ]:
    """
    Get slices of rounds from data along with the label for the winner. For each round,
    a certain number of the initial data points is obtained, specified
    by round_frac. Indices of obtained rounds are kept for standardization purposes.
    Sheets are stored as well to simplify splitting into training, validation, and test sets later.

    Args:
        data (pd.DataFrame):
            DataFrame object to obtain round data points from.
        round_frac (float):
            The initial fraction of the rounds to obtain.

    Returns:
        (data_points, labels, idx_for_std, max_seq_len, sheets) (Tuple[List[np.ndarray], List[int], np.array[int], int, List[str]]):
            Sliced data points, labels, indices of selected data points, the maximum sequence length
            encountered in the data, and the sheet numbers for each round.
    """

    assert 0.0 <= round_frac <= 1.0, "Percentage must be in the interval [0.0, 1.0]."

    if round_frac == 1.0:
        warnings.warn(message="""Current fraction of timesteps per round to be retained is 1.0, or 100%.
                      This represents a high risk of data leakage as the model will have direct access
                      to the outcome of each round. Consider lowering the percentage.
                      """)

    # Lists for storing data points for rounds and targets
    data_points = []
    targets = []

    # Variable for keeping track of the longest sequence
    max_seq_len = 0

    # List for keeping track of indices
    # that will be kept for training.
    # This will be used to standardize the data later
    idx_for_std = []

    # List for keeping track of sheets
    sheets = []

    # Dict for mapping player names to integers
    player_mapping = {"Player_1": 0, "Player_2": 1}

    # Iterate over each round
    for i in range(1, len(round_start_idx)):
        # Get the start indices of the current and next round
        curr_round_start = round_start_idx[i - 1]
        next_round_start = round_start_idx[i]

        # Compute length of current round
        curr_round_len = next_round_start - curr_round_start

        # Get the ending index for chosen fraction of data points
        # and store it if it's longer than max_seq_len
        seq_len = round(curr_round_len * round_frac)

        if seq_len > max_seq_len:
            max_seq_len = seq_len

        # Store data points for current sequence.
        # Only damage-related columns are kept
        seq_end = curr_round_start + seq_len
        data_points.append(
            data.iloc[curr_round_start:seq_end].drop(
            [ "Winner", "Sheet"], axis=1).to_numpy())

        # Obtain and store the sheet number for current round
        sheet = data.iloc[curr_round_start]["Sheet"]
        sheets.append(sheet)

        # Obtain the winner of the current round
        # and map them to an integer target
        winner = data.iloc[curr_round_start]["Winner"]
        targets.append(player_mapping[winner])

        # Keep the indices of selected data points for current round
        indices = np.arange(curr_round_start, curr_round_start + seq_len)
        idx_for_std.append(indices)

    return data_points, targets, idx_for_std, max_seq_len, sheets

def compute_label_proportions(labels: List[int]) -> np.ndarray:
    """
    Compute label proportions from provided label list.

    Args:
        labels (List[int]):
            Integer labels in a list.

    Returns:
        label_counts (np.ndarray):
            A NumPy array containing the percentages of each label
            stored at the index corresponding to the label.
    """

    label_counts = np.unique(labels, return_counts=True)[1] # Return value is a tuple, we need the second element with value counts
    label_counts = label_counts/label_counts.sum() # Divide value counts by total

    return label_counts

class RoundDataset(Dataset):
    def __init__(
            self,
            data: List[np.ndarray],
            labels: List[int],
            means: Tuple[float, float],
            stds: Tuple[float, float],
            max_seq_len: int):
        """
        Dataset for Street Fighter matches.

        Args:
            data (List[np.ndarray]):
                List containing unpadded NumPy arrays of varying lengths,
                each representing a separate game round.
            labels (List[int]):
                The labels for the game matches in data.
            means (Tuple[float, float]):
                A tuple containing the means of each feature in data.
            std (Tuple[float, float]):
                A tuple containing the standard deviations of each feature in data.
            max_seq_len (int):
                The maximum sequence length in the data
                that all data points should be padded to.
        """
        super(RoundDataset, self).__init__()
        self.data = data
        self.means = means
        self.stds = stds
        self.labels = labels
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(
            self,
            idx: int) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Obtain a single sample from the dataset.

        Args:
            idx (int):
                The index of the sample to obtain:

        Returns:
            (data, label, seq_len) (Tuple[np.ndarray, float, int]):
                The padded sample, the corresponding label, and the length of the sequence before padding.
        """

        # Obtain sample, standardize it, and then pad it
        data = self.data[idx].copy()
        data[:, 0] = (data[:, 0] - self.means[0]) / self.stds[0]
        data[:, 1] = (data[:, 1] - self.means[1]) / self.stds[1]

        seq_len = data.shape[0]
        data = np.pad(
            array=data,
            pad_width=((0, self.max_seq_len - seq_len), (0, 0)), # Pad axis 0, which corresponds to the time step dimension
            constant_values=0).astype(np.float32) # Fill -1 as padding

        # Obtain label as float32
        label = np.float32(self.labels[idx])

        return data, label, seq_len

class LSTMModel(nn.Module):
    def __init__(
            self,
            in_dim: int = 2,
            hidden_dim: int = 8,
            bidirectional: bool = True,
            dropout: float = 0.3
            ):
        """
        LSTM classifier.

        Args:
            in_dim (int):
                The dimension of the model's input. Default: 2.
            hidden_dim (int):
                Dimension of the hidden layer. Default: 4.
            bidirectional (bool):
                Whether the LSTM layer is bidirectional. Default: True.
            dropout (float):
                Dropout rate for LSTM output. Default: 0.25.
        """

        super(LSTMModel, self).__init__()

        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=bidirectional)
        self.drop = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(
            in_features=hidden_dim * (bidirectional + 1),
            out_features=1)

    def forward(
            self,
            x: torch.Tensor,
            seq_lens: torch.Tensor,
            pack_sequences: bool = True) -> torch.Tensor:
            """
            Forward pass through the LSTM classifier.

            Args:
                x (torch.Tensor):
                    Tensor of shape (batch_size, seq_len, in_dim)
                    containing the input data to the model.
                seq_lens (torch.Tensor):
                    A list of the sequence lengths for samples in x before padding.
                pack_sequences (bool):
                    Whether inputs should be packed. Default: True
            """

            # Pack sequences
            if pack_sequences:
                x = nn.utils.rnn.pack_padded_sequence(
                     input=x,
                     lengths=seq_lens,
                     batch_first=True,
                     enforce_sorted=False)

            out, _ = self.lstm(x)

            # Unpack LSTM output if it's been packed
            if pack_sequences:
                out, _ = nn.utils.rnn.pad_packed_sequence(
                     sequence=out,
                     batch_first=True) # (batch_size, max_seq_len_in_batch, hidden_size)

            # Apply dropout, average data over time steps,
            # and feed to classifier layer
            out = self.drop(out) # (batch_size, max_seq_len_in_batch, hidden_size)
            out = self.classifier(out.mean(dim=1)) # (batch_size, hidden_size)

            return out

def train_step(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Training step with gradient update on a batch of inputs.

    Args:
        inputs (torch.Tensor):
            Tensor with inputs to model.
        targets (torch.Tensor):
            Tensor with target outputs for the inputs.

    Returns:
        (outs, loss) (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
            Output logits and loss for batch.
    """

    # Zero gradients computed on previous inputs
    optimizer.zero_grad()

    # Perform a forward pass and compute loss
    outs = model(x=inputs.to(device), **kwargs)
    loss = loss_fn(input=outs.cpu(), target=targets.unsqueeze(1)) # Add dimension of size 1 to targets

    # Apply gradients
    loss.backward()
    optimizer.step()

    return outs.cpu().detach(), loss

def val_or_test_step(
                inputs: torch.Tensor,
                targets: torch.Tensor,
                **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Validation or test step on a batch of inputs.

    Args:
        inputs (torch.Tensor):
            Tensor with inputs to model.
        targets (torch.Tensor):
            Tensor with target outputs for the inputs.

    Returns:
        (outs, loss) (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
            Output logits and loss for batch.
    """

    with torch.no_grad(): # Disable gradient computation to save on compute
        # Perform a forward pass and compute loss
        outs = model(x=inputs.to(device), **kwargs)
        loss = loss_fn(input=outs.cpu(), target=targets.unsqueeze(1)) # Add dimension of size 1 to targets

    return outs.cpu().detach(), loss

def train_epoch(
        train_dl: DataLoader,
        val_dl: DataLoader) -> Tuple[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            ]:
    """
    One training epoch.

    Args:
        train_dl (DataLoader):
            Train dataloader for weight updates.
        val_dl (DataLoader):
            Val dataloader for checking generalization.

    Returns:
        (train_outs, val_outs) (Tuple[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            ]):
        A tuple of two tuples, each containing output logits, targets, and loss
        for the train and validation sets.
    """

    # Lists for storing predictions and targets
    train_epoch_preds = []
    train_epoch_targets = []

    val_epoch_preds = []
    val_epoch_targets = []

    # Variable for keeping track of training loss
    running_loss = 0.0

    # Training loop
    model.train()
    for i, (inputs, targets, seq_lens) in enumerate(train_dl):
        # Update model weights on batch
        train_preds, loss = train_step(
            inputs=inputs,
            targets=targets,
            **{"seq_lens": seq_lens, "pack_sequences": True})

        # Store predictions and targets
        train_epoch_preds.append(train_preds)
        train_epoch_targets.append(targets)

        # Account for the loss of current batch
        running_loss += loss.item()

    # Concat predictions for later use
    train_epoch_preds = torch.cat(train_epoch_preds)
    train_epoch_targets = torch.cat(train_epoch_targets)

    # Compute train loss for epoch as the average of losses for all batches
    train_loss = running_loss / (i + 1)



    # Reset running_Loss for use with the validation set
    running_loss = 0.0

    # Validation loop
    model.eval()
    for i, (inputs, targets, seq_lens) in enumerate(val_dl):
        # Compute logits and validation loss
        val_preds, loss = val_or_test_step(
            inputs=inputs,
            targets=targets,
            **{"seq_lens": seq_lens, "pack_sequences": True})

        # Store predictions and targets
        val_epoch_preds.append(val_preds)
        val_epoch_targets.append(targets)

        # Account for the loss of current batch
        running_loss += loss.item()

    # Concat predictions for later use
    val_epoch_preds = torch.cat(val_epoch_preds)
    val_epoch_targets = torch.cat(val_epoch_targets)

    # Compute validation loss for epoch
    val_loss = running_loss / (i + 1)

    # Pack outputs
    train_outs = (train_epoch_preds, train_epoch_targets, train_loss)
    val_outs = (val_epoch_preds, val_epoch_targets, val_loss)

    return (train_outs, val_outs, model)
import os
batch_size = 64
count = 1
TA_tr = {}
TA_te = {}
TC_tr = {}
TC_te = {}
GT_tr = {}
GT_te = {}
PD_tr = {}
PD_te = {}
LO_tr = {}
LO_te = {}
for i in [2,3,4,5,6]:
    # Split the data into rounds and obtain the initial portion
  # of data points from each, specified by round_frac
  data_points, labels, idx_for_std, max_seq_len, sheets = get_round_slices(
      data=data,
      round_frac=0.75)

  # Make sure that the retrieved data points match
  # values at the source at provided indices
  data_c = data.drop([ "Winner", "Sheet"], axis=1)
  cond = [(data_points[i] == data_c.iloc[idx_for_std[i]].values).all() for i in range(len(data_points))]
  assert np.all(cond), "Not all retrieved data points exactly match the source data."

  # Split data into training, validation, and test sets,
  # making sure that there is no overlap between sheets in the sets
  train_end = np.where(np.array(sheets) == "Sheet_8")[0][0]
  test_start = np.where(np.array(sheets) == "Sheet_{}".format(i))[0][0]
  test_end = np.where(np.array(sheets) == "Sheet_{}".format(i+3))[0][0]

  #assert test_end > train_end, "Test set begins before training set ends"

  #train_data_points = data_points[:train_end]
  test_data_points = data_points[test_start:test_end]
  train_data_points = data_points[:test_start] + data_points[test_end:]

  test_labels = labels[test_start:test_end]
  mask = ~np.isin(labels, test_labels)
  train_labels = labels[:test_start] + labels[test_end:]

  # Compute means and stds on the TRAIN SET
  idx_for_std = np.concatenate(idx_for_std[:train_end]) # We need to concat the indices because they were stored in separate lists per round

  mean_1 = data["Player1_Damaged%"].iloc[idx_for_std].mean()
  mean_2 = data["Player2_Damaged%"].iloc[idx_for_std].mean()

  std_1 = data["Player1_Damaged%"].iloc[idx_for_std].std()
  std_2 = data["Player2_Damaged%"].iloc[idx_for_std].std()

  print(f"{len(train_data_points)} samples in training set")
  print(f"{len(test_data_points)} samples in test set")

  # Compute and display class distribution in data
  class_counts = compute_label_proportions(labels)
  train_class_counts = compute_label_proportions(train_labels)
  test_class_counts = compute_label_proportions(test_labels)

  print(f"Label proportions for overall data - label 0: {class_counts[0]:.4f}, label 1: {class_counts[1]:.4f}")
  print(f"Label proportions for train set - label 0: {train_class_counts[0]:.4f}, label 1: {train_class_counts[1]:.4f}")
  print(f"Label proportions for test set - label 0: {test_class_counts[0]:.4f}, label 1: {test_class_counts[1]:.4f}")
  print('$'*75)

  # Create dataloaders for each split
  train_ds = RoundDataset(
      data=train_data_points,
      labels=train_labels,
      means=(mean_1, mean_2),
      stds=(std_1, std_2),
      max_seq_len=max_seq_len)
  train_dl = DataLoader(
      dataset=train_ds,
      batch_size=batch_size,
      shuffle=True)

  test_ds = RoundDataset(
      data=test_data_points,
      labels=test_labels,
      means=(mean_1, mean_2),
      stds=(std_1, std_2),
      max_seq_len=max_seq_len)
  test_dl = DataLoader(
      dataset=test_ds,
      batch_size=batch_size,
      shuffle=False)

  # Initialize model and move to device
  model = LSTMModel(in_dim=train_ds[0][0].shape[1])

  model.to(device=device)

  loss_fn = nn.BCEWithLogitsLoss()
  optimizer = optim.Adam(
      params=model.parameters(),
      lr=0.001,
      weight_decay=1e-4)
  # Lists for storing various metrics

  # Losses
  train_losses = []
  test_losses = []

  # ROC AUC
  train_aucs = []
  test_aucs = []

  # Accuracies
  train_accuracies = []
  test_accuracies = []

  # Training loop
  highest = 0

  for epoch in range(500):
      # Run one training epoch
      train_outs, test_outs, model = train_epoch(
          train_dl=train_dl,
          val_dl=test_dl)

      # Unpack training and validation outputs
      train_epoch_preds, train_epoch_targets, train_loss = train_outs
      test_epoch_preds, test_epoch_targets, test_loss = test_outs

      # Convert logits to probabilities
      train_epoch_preds = torch.sigmoid(train_epoch_preds)
      test_epoch_preds = torch.sigmoid(test_epoch_preds)


      # Store metrics for train and val splits
      train_losses.append(train_loss)
      train_accuracies.append(metrics.accuracy_score(
          y_true=train_epoch_targets,
          y_pred=train_epoch_preds >= 0.5))
      train_auc = metrics.roc_auc_score(
          y_true=train_epoch_targets,
          y_score=train_epoch_preds)
      train_aucs.append(train_auc)

      test_losses.append(test_loss)
      test_accuracies.append(metrics.accuracy_score(
          y_true=test_epoch_targets,
          y_pred=test_epoch_preds >= 0.5))

      test_acc = metrics.accuracy_score(y_true=test_epoch_targets,y_pred=test_epoch_preds >= 0.5)

      test_auc = metrics.roc_auc_score(
          y_true=test_epoch_targets,
          y_score=test_epoch_preds)
      test_aucs.append(test_auc)

      # Display epoch metrics
      print(f"Training Loss for epoch {epoch + 1}: ", train_loss)
      print(f"Test Loss for epoch {epoch + 1}: ", test_loss)
      print('Test Accuracy:-', metrics.accuracy_score(y_true=test_epoch_targets,y_pred=test_epoch_preds >= 0.5))
      print('$'*100)

      if(test_acc>highest):
        print(test_loss)
        highest = test_acc
        TA_tr['model_check_point{}.pth'.format(count)] = train_accuracies
        TC_tr['model_check_point{}.pth'.format(count)] = train_aucs
        LO_tr['model_check_point{}.pth'.format(count)] = train_losses
        TA_te['model_check_point{}.pth'.format(count)] = test_accuracies
        TC_te['model_check_point{}.pth'.format(count)] = test_aucs
        LO_te['model_check_point{}.pth'.format(count)] = test_losses
        if os.path.exists('model_check_point{}.pth'.format(count)):
          os.remove('model_check_point{}.pth'.format(count))
        torch.save(model.state_dict(), 'model_check_point{}.pth'.format(count))

  test_epoch_pred = []

  # Instead, directly use the sigmoid outputs as continuous predictions
  test_epoch_pred_continuous = test_epoch_preds.detach().numpy()

  dfh = pd.DataFrame()
  dfh['Ground Gruth'] = list(test_epoch_targets)
  dfh['Prediction'] = test_epoch_pred_continuous
  dfh.to_csv('0.75_LSTM_KFold_{}.csv'.format(count))
  count = count + 1
# Display metrics after training
print(f"Training Loss: ", train_loss)
print(f"Test Loss: ", test_loss)
print(f"ROC AUC score for train set: {metrics.roc_auc_score(train_epoch_targets, train_epoch_preds):.4f}")
print(f"ROC AUC score for test set: {metrics.roc_auc_score(test_epoch_targets, test_epoch_preds):.4f}")


# In[ ]:




