import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import numpy as np

# Importing the RNN class and task dataset implementation
from train_rnn import MyRNN, train
from rnntaskdataset import RNNTaskDataset

# Define the hyperparameter search space
hidden_dims = [32, 64, 128, 256]
initialization_types = ["random", "uniform"]
g_values = [0.5, 1.0, 1.5]
nonlinearities = ['tanh', 'relu']
tasks = ['ready_set_go', 'delay_discrimination', 'flip_flop', 'evidence_accumulation']

# Create a folder to save weights
weights_folder = "rnn_weights"
os.makedirs(weights_folder, exist_ok=True)

# Maximum number of retries if training doesn't converge
max_retries = 3

# Training settings
device = 'cuda' if torch.cuda.is_available() else 'cpu'
epochs = 1000
batch_size = 32
learning_rate = 1e-4


# Helper function to check convergence
def has_converged(loss_array, threshold=1e-2):
    if len(loss_array) > 5:
        return abs(loss_array[-1] - loss_array[-5]) < threshold
    return False


# Training script
def train_multiple_rnns(task_dataset_class):
    for hidden_dim in hidden_dims:
        for init_type in initialization_types:
            for g in g_values:
                for nonlinearity in nonlinearities:
                    for task in tasks:
                        retries = 0
                        converged = False

                        # Load the task dataset
                        task_dataset = task_dataset_class(n_trials=200, time=150, n_channels=2)
                        x, y = getattr(task_dataset, task)()
                        x_tensor = torch.tensor(x, dtype=torch.float32)
                        y_tensor = torch.tensor(y, dtype=torch.float32)
                        dataset = TensorDataset(x_tensor, y_tensor)
                        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

                        # Retry training if it doesn’t converge
                        while not converged and retries < max_retries:
                            print(
                                f"Training: hidden_dim={hidden_dim}, init_type={init_type}, g={g}, nonlinearity={nonlinearity}, task={task}")

                            # Create the RNN model with the current hyperparameters
                            initialize_uniform = (init_type == "uniform")
                            initialize_normal = (init_type == "random")
                            model = MyRNN(input_dim=x.shape[2], hidden_dim=hidden_dim, output_dim=y.shape[2],
                                          nonlinearity=nonlinearity, initialize_uniform=initialize_uniform,
                                          initialize_normal=initialize_normal, g=g).to(device)

                            # Train the model
                            train_loss_array, _ = train(model, train_loader, criterion='mse', device=device,
                                                        lr=learning_rate, epochs=epochs)

                            # Check if the model has converged
                            converged = has_converged(train_loss_array)

                            if not converged:
                                retries += 1
                                print(f"Training failed to converge. Retrying... ({retries}/{max_retries})")
                            else:
                                print(f"Training converged successfully!")

                                # Save the final weights
                                save_path = os.path.join(weights_folder,
                                                         f"weights_{hidden_dim}_{init_type}_g{g}_{nonlinearity}_{task}.pt")
                                torch.save(model.state_dict(), save_path)
                                print(f"Model weights saved to: {save_path}")


# Assuming task_dataset_class is the class implementing the tasks (from your earlier task dataset implementation)
train_multiple_rnns(RNNTaskDataset)
