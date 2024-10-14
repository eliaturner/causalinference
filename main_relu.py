import torch
import torch.optim as optim
import os
import numpy as np
from train_rnn import MyRNN, train

from load_and_plot_model import load_and_plot_model
from train_multiple_rnns import train_multiple_rnns

# Define the hyperparameter search space
hidden_dims = [32, 64, 128, 256]
# hidden_dims = [64]
initialization_types = ["random"]
# initialization_types = ["random", "uniform"]
g_values = [0.1, 0.5, 0.9]
# g_values = [0.5]
# g_values = [1]
# nonlinearities = ['tanh']
nonlinearities = ['relu']
# tasks = ['ready_set_go']
tasks = ['ready_set_go', 'delay_discrimination', 'integrator']
# tasks = ['ready_set_go', 'delay_discrimination', 'flip_flop', 'integrator']

# Create a folder to save weights
weights_folder = "rnn_weights"
os.makedirs(weights_folder, exist_ok=True)

# Maximum number of retries if training doesn't converge

# Training settings
epochs = 2000
batch_size = 32
learning_rate = 1e-3

hyperparameters = {'hidden_dims': hidden_dims, 'initialization_types': initialization_types, 'g_values': g_values,
                   'nonlinearities': nonlinearities, 'tasks': tasks, 'epochs': epochs, 'batch_size': batch_size,
'learning_rate': learning_rate, 'max_retries': 3, 'weights_folder': weights_folder}



# Training script





# Example usage: Plot for a specific set of hyperparameters and tasks

# Assuming task_dataset_class is the class implementing the tasks (from your earlier task dataset implementation)
train_multiple_rnns(**hyperparameters)
# load_and_plot_model(weights_folder, hidden_dim=64, init_type="random", g=1, nonlinearity='tanh', tasks=tasks)
