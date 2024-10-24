import torch
import os
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from rnntaskdataset import RNNTaskDataset  # Assuming your task dataset class
from train_rnn import MyRNN  # Assuming the RNN model implementation
from train_multiple_rnns import MaskedMSELoss  # Assuming your custom loss function

# Define the universal test set for all tasks
universal_test_dataset_class = RNNTaskDataset
n_test_trials = 50  # Define the number of test trials for the test set
test_time = 100  # Define the length of the test time
test_batch_size = 32
figure_folder = "rnn_figures"
os.makedirs(figure_folder, exist_ok=True)

# Prepare a table to store the results
columns = ['hidden_dim', 'init_type', 'g', 'nonlinearity', 'task', 'mse_loss']
results_table = pd.DataFrame(columns=columns)

# Define your hyperparameter space
hidden_dims = [32, 64, 128, 256]
initialization_types = ["random"]  # Now using only random initialization
g_values = [0.1, 0.5, 0.9]
nonlinearities = ['tanh', 'relu']
tasks = ['ready_set_go', 'delay_discrimination', 'flip_flop', 'integrator']
weights_folder = "rnn_weights_2"

# Define device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the test dataset once for all tasks
task_test_data = {}
task_test_data_extrapolation = {}

hyperparameters = {'hidden_dims': hidden_dims, 'initialization_types': initialization_types, 'g_values': g_values,
                   'nonlinearities': nonlinearities, 'tasks': tasks, 'weights_folder': weights_folder}


def get_test_data(task, extrapolation=False):
    if extrapolation:
        time = 100 + 200
        task_data = task_test_data_extrapolation
    else:
        time = 100
        task_data = task_test_data
    if task not in task_data:
        task_dataset = universal_test_dataset_class(n_trials=n_test_trials, time=time, n_channels=2)
        x, y = getattr(task_dataset, task)()  # Generate x and y for the task
        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        task_test_data[task] = (
        DataLoader(TensorDataset(x_tensor, y_tensor), batch_size=test_batch_size), x_tensor, y_tensor)
    return task_test_data[task]


# Plot function to visualize input, desired output, and actual output for 3 trials
def plot_results(x, y_true, y_pred, task, loss, save_name):
    num_trials = 3  # Show results for 3 trials
    time = np.arange(x.shape[1])

    fig, axs = plt.subplots(num_trials, 1, figsize=(10, 6), sharex=True)
    fig.suptitle(f"{task}: Desired vs Actual Output, MSE {loss:.4f}", fontsize=16)

    for i in range(num_trials):
        axs[i].plot(time, y_true[i, :, 0], label="Desired Output", color='blue')  # Correctly using the 2D format
        axs[i].plot(time, y_pred[i, :, 0], label="Actual Output", linestyle='dashed', color='red')
        axs[i].plot(time, x[i, :, 0], label="Input", linestyle='dotted', color='green')
        if x.shape[2] > 1:
            axs[i].plot(time, x[i, :, 1], linestyle='dashdot', color='green')
        axs[i].legend()
        axs[i].set_ylabel("Value")

    axs[-1].set_xlabel("Time Steps")
    plt.tight_layout()
    plt.savefig(os.path.join(figure_folder, save_name))
    plt.close()


# Evaluation function
def evaluate_model(model, test_loader, criterion):
    model.eval()
    running_loss = 0
    all_y_pred = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred, _ = model(X_batch)
            loss = criterion(y_pred, y_batch)
            running_loss += loss.item()
            all_y_pred.append(y_pred.cpu().numpy())

    all_y_pred = np.concatenate(all_y_pred, axis=0)
    return running_loss / len(test_loader), all_y_pred


def evaluate_model_noise(model, test_loader, criterion):
    model.eval()
    running_loss = 0
    all_y_pred = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred, _ = model(X_batch, inject_noise=True)
            loss = criterion(y_pred, y_batch)
            running_loss += loss.item()
            all_y_pred.append(y_pred.cpu().numpy())

    all_y_pred = np.concatenate(all_y_pred, axis=0)
    return running_loss / len(test_loader), all_y_pred


def get_weight_path():
    # Generate the filenames for the saved models
    successful_weights_filename = f"weights_{hidden_dim}_random_g{g}_{nonlinearity}_{task}.pt"
    failed_weights_filename = f"best_failed_weights_{hidden_dim}_random_g{g}_{nonlinearity}_{task}.pt"
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

    return weights_path


normal_loss = []
extrapolation_loss = []

# Load weights, evaluate, and store results
for hidden_dim in hidden_dims:
    for g in g_values:
        for nonlinearity in nonlinearities:
            for task in tasks:
                weights_path = get_weight_path()

                if weights_path:
                    # Load the universal test data for the task (load x, y once)
                    test_loader, x_tensor, y_tensor = get_test_data(task)
                    test_loader_extrapolation, x_tensor_extrapolation, y_tensor_extrapolation = get_test_data(task, extrapolation=True)

                    # Dynamically determine input and output dimensions based on the test set
                    input_dim = x_tensor.shape[2]  # Number of input channels
                    output_dim = y_tensor.shape[2]  # Number of output channels

                    # Create the RNN model with the inferred dimensions
                    model = MyRNN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                                  nonlinearity=nonlinearity, initialize_uniform=False,
                                  initialize_normal=True, g=g).to(device)

                    # Load the weights, handle CPU-only machines
                    if device == 'cpu':
                        model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
                    else:
                        model.load_state_dict(torch.load(weights_path))

                    # Define the loss function (MaskedMSELoss)
                    criterion = MaskedMSELoss()

                    # Evaluate the model
                    # mse_loss, y_pred = evaluate_model(model, test_loader, criterion)
                    y_pred, _ = model(x_tensor, inject_noise=False)
                    mse_loss = criterion(y_pred, y_tensor).item()

                    ### step 1: Evaluate the model with loss extrapolation
                    # mse_loss_extrapolation, y_pred_extrapolation = evaluate_model(model, test_loader_extrapolation, criterion)
                    y_pred_extrapolation, _ = model(x_tensor_extrapolation, inject_noise=False)
                    mse_loss_extrapolation = criterion(y_pred_extrapolation, y_tensor_extrapolation).item()

                    ## step 2: Evaluate with lesion
                    from train_rnn import simulate_with_lesions
                    mse_lesion = simulate_with_lesions(model, x_tensor, y_tensor, 0.01)

                    ## step 3: Evaluate with noise
                    # mse_loss_noise, y_pred_noise = evaluate_model_noise(model, test_loader, criterion)
                    y_pred_noise, _ = model(x_tensor, inject_noise=True)
                    mse_loss_noise = criterion(y_pred_noise, y_tensor).item()

                    # Store the results in the table
                    results_table = results_table.append({
                        'hidden_dim': hidden_dim,
                        'init_type': 'random',
                        'g': g,
                        'nonlinearity': nonlinearity,
                        'task': task,
                        'mse_loss': np.log(mse_loss),
                        'loss_extrapolation': np.log(mse_loss_extrapolation),
                        'loss_noisy': np.log(mse_loss_noise),
                        'loss_lesion': np.log(mse_lesion),
                    }, ignore_index=True)

                    normal_loss.append(np.log(mse_loss))
                    extrapolation_loss.append(np.log(mse_loss_extrapolation))

                else:
                    print(f"No weights found for {hidden_dim}, random, g={g}, {nonlinearity}, task={task}")

# # Save the results to a CSV file
# results_table.to_csv('rnn_evaluation_results.csv', index=False)

import seaborn as sns
import matplotlib.pyplot as plt

# Create a folder to save plots if it doesn't exist
loss_plots_folder = "loss_plots"
os.makedirs(loss_plots_folder, exist_ok=True)

# Melt the DataFrame to prepare for plotting (for easier Seaborn usage)
loss_data = results_table.melt(
    id_vars=['hidden_dim', 'init_type', 'g', 'nonlinearity', 'task'],
    value_vars=['mse_loss', 'loss_extrapolation', 'loss_noisy', 'loss_lesion'],
    var_name='Loss Type',
    value_name='Loss Value'
)

# Plot loss distribution using violin plots
plt.figure(figsize=(12, 6))
sns.violinplot(data=loss_data, x='Loss Type', y='Loss Value', inner='box', scale='width')
# plt.yscale('log')  # Use log scale for better visualization of distributions
plt.title('Distribution of Losses Across Hyperparameter Combinations')
plt.ylabel('Log Loss Value')
plt.xlabel('Loss Type')
plt.tight_layout()
plt.savefig(os.path.join(loss_plots_folder, 'loss_violin_plot.png'))
# plt.show()

# Optionally, create box plots for comparison
plt.figure(figsize=(12, 6))
sns.boxplot(data=loss_data, x='Loss Type', y='Loss Value')
# plt.yscale('log')
plt.title('Box Plot of Losses Across Hyperparameter Combinations')
plt.ylabel('Log Loss Value')
plt.xlabel('Loss Type')
plt.tight_layout()
plt.savefig(os.path.join(loss_plots_folder, 'loss_box_plot.png'))
# plt.show()
