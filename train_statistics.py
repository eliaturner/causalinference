import torch
import os
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from rnntaskdataset import RNNTaskDataset  # Assuming your task dataset class
from train_rnn import MyRNN  # Assuming the RNN model implementation
from train_multiple_rnns import MaskedMSELoss  # Assuming your custom loss function

# Define the universal test set for all tasks
universal_test_dataset_class = RNNTaskDataset
n_test_trials = 50  # Define the number of test trials for the test set
test_time = 100  # Define the length of the test time
test_batch_size = 32

# Prepare a table to store the results
columns = ['hidden_dim', 'init_type', 'g', 'nonlinearity', 'task', 'mse_loss']
results_table = pd.DataFrame(columns=columns)

# Define your hyperparameter space
hidden_dims = [32, 64, 128, 256]
initialization_types = ["random"]
g_values = [0.1, 0.5, 0.9]
nonlinearities = ['tanh', 'relu']
tasks = ['ready_set_go', 'delay_discrimination', 'flip_flop', 'integrator']
weights_folder = "rnn_weights"

# Define device
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Load the test dataset for all tasks
def get_test_data(task):
    task_dataset = universal_test_dataset_class(n_trials=n_test_trials, time=test_time, n_channels=2)
    x, y = getattr(task_dataset, task)()  # Generate x and y for the task
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    return DataLoader(TensorDataset(x_tensor, y_tensor), batch_size=test_batch_size), x_tensor, y_tensor


# Evaluation function
def evaluate_model(model, test_loader, criterion):
    model.eval()
    running_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred, _ = model(X_batch)
            loss = criterion(y_pred, y_batch)
            running_loss += loss.item()
    return running_loss / len(test_loader)


# Load weights, evaluate, and store results
for hidden_dim in hidden_dims:
    for init_type in initialization_types:
        for g in g_values:
            for nonlinearity in nonlinearities:
                for task in tasks:
                    # Generate the filenames for the saved models
                    successful_weights_filename = f"weights_{hidden_dim}_{init_type}_g{g}_{nonlinearity}_{task}.pt"
                    failed_weights_filename = f"best_failed_weights_{hidden_dim}_{init_type}_g{g}_{nonlinearity}_{task}.pt"
                    successful_weights_path = os.path.join(weights_folder, successful_weights_filename)
                    failed_weights_path = os.path.join(weights_folder, failed_weights_filename)

                    # Check which weight file exists
                    weights_path = None
                    if os.path.exists(successful_weights_path):
                        weights_path = successful_weights_path
                        print(f"Evaluating successful model for {successful_weights_filename}")
                    elif os.path.exists(failed_weights_path):
                        weights_path = failed_weights_path
                        print(f"Evaluating failed model for {failed_weights_filename}")

                    if weights_path:
                        # Load the universal test data for the task
                        test_loader, x_tensor, y_tensor = get_test_data(task)

                        # Dynamically determine input and output dimensions based on the test set
                        input_dim = x_tensor.shape[2]  # Number of input channels
                        output_dim = y_tensor.shape[2]  # Number of output channels

                        # Create the RNN model with the inferred dimensions
                        model = MyRNN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                                      nonlinearity=nonlinearity,
                                      initialize_uniform=(init_type == "uniform"),
                                      initialize_normal=(init_type == "random"), g=g).to(device)

                        # Load the weights, handle CPU-only machines
                        if device == 'cpu':
                            model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
                        else:
                            model.load_state_dict(torch.load(weights_path))

                        # Define the loss function (MaskedMSELoss)
                        criterion = MaskedMSELoss()

                        # Evaluate the model
                        mse_loss = evaluate_model(model, test_loader, criterion)

                        mse_loss = round(mse_loss, 5)

                        # Store the results in the table
                        results_table = results_table.append({
                            'hidden_dim': hidden_dim,
                            'init_type': init_type,
                            'g': g,
                            'nonlinearity': nonlinearity,
                            'task': task,
                            'mse_loss': mse_loss
                        }, ignore_index=True)
                    else:
                        print(f"No weights found for {hidden_dim}, {init_type}, g={g}, {nonlinearity}, task={task}")

# Save the results to a CSV file
results_table.to_csv('rnn_evaluation_results.csv', index=False)

print("Evaluation complete. Results saved to rnn_evaluation_results.csv")
