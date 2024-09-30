import matplotlib.pyplot as plt
import torch
device = 'cpu'
from train_rnn import MyRNN
import os
from rnntaskdataset import RNNTaskDataset

# Function to load a trained model and plot its output vs. desired output for a few examples
def load_and_plot_model(weights_folder, hidden_dim, init_type, g, nonlinearity, tasks):

    # Loop over all tasks
    for task in tasks:
        # Load the task dataset with a few trials (2 trials)
        task_dataset = RNNTaskDataset(n_trials=2, time=150, n_channels=2)
        x, y = getattr(task_dataset, task)()  # Get the input and target outputs for the task
        x_tensor = torch.tensor(x, dtype=torch.float32).to(device)

        # Load the trained model weights
        model = MyRNN(input_dim=x.shape[2], hidden_dim=hidden_dim, output_dim=y.shape[2],
                      nonlinearity=nonlinearity, initialize_uniform=(init_type == "uniform"),
                      initialize_normal=(init_type == "random"), g=g).to(device)

        weights_path = os.path.join(weights_folder, f"weights_{hidden_dim}_{init_type}_g{g}_{nonlinearity}_{task}.pt")

        # Load weights if they exist
        if os.path.exists(weights_path):
            model.load_state_dict(torch.load(weights_path))
            model.eval()

            # Pass the input examples through the model
            with torch.no_grad():
                y_pred, _ = model(x_tensor)
                y_pred = y_pred.cpu().numpy()

            # Plot the model's predictions vs. true outputs for each trial
            for trial in range(x.shape[0]):  # We have 2 trials
                plt.figure(figsize=(10, 5))
                plt.plot(y[trial].flatten(), label='Desired Output', color='blue')
                plt.plot(y_pred[trial].flatten(), label='Model Output', linestyle='dashed', color='red')
                plt.title(f'Task: {task}, Trial: {trial}')
                plt.xlabel('Time Steps')
                plt.ylabel('Output')
                plt.legend()
                plt.show()

        else:
            print(
                f"Weights not found for: hidden_dim={hidden_dim}, init_type={init_type}, g={g}, nonlinearity={nonlinearity}, task={task}")

