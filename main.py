import os

import numpy as np
import torch
import torch.optim as optim

from load_and_plot_model import load_and_plot_model
from train_multiple_rnns import train_multiple_rnns
from train_rnn import MyRNN, train

# Define the hyperparameter search space
hidden_dims = [32, 64, 128, 256]
# hidden_dims = [64]
initialization_types = ["random"]
# initialization_types = ["random", "uniform"]
g_values = [0.1, 0.5, 0.9]
# g_values = [0.5]
# g_values = [1]
# nonlinearities = ['tanh']
nonlinearities = ['tanh']
# tasks = ['ready_set_go']
tasks = ['ready_set_go', 'delay_discrimination', 'flip_flop', 'integrator']

# Create a folder to save weights
weights_folder = "rnn_weights"
os.makedirs(weights_folder, exist_ok=True)

# Training settings
epochs = 5000
batch_size = 32
learning_rate = 1e-3

hyperparameters = {'hidden_dims': hidden_dims, 'initialization_types': initialization_types, 'g_values': g_values,
                   'nonlinearities': nonlinearities, 'tasks': tasks, 'epochs': epochs, 'batch_size': batch_size,
'learning_rate': learning_rate, 'max_retries': 3, 'weights_folder': weights_folder}

# Assuming task_dataset_class is the class implementing the tasks (from your earlier task dataset implementation)
train_multiple_rnns(**hyperparameters)
