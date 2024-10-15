# Importing the RNN class and task dataset implementation

import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from rnntaskdataset import RNNTaskDataset
from train_rnn import MyRNN


# Helper function to check convergence
# def has_converged(loss_array, threshold=1e-2):
#     if len(loss_array) > 5:
#         return abs(loss_array[-1] - loss_array[-5]) < threshold
#     return False
class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, y_pred, y_true):
        # Create a mask that is 1 where y_true is not nan and 0 where y_true is nan
        mask = ~torch.isnan(y_true)

        # Apply the mask to y_pred and y_true
        y_pred_masked = y_pred[mask]
        y_true_masked = y_true[mask]

        # Compute MSE only for non-nan values
        loss = nn.functional.mse_loss(y_pred_masked, y_true_masked)
        return loss



# Helper function to check convergence
def has_converged(loss_array, threshold=1e-4):
    return min(loss_array) < threshold



def train_multiple_rnns(**hyperparameters):
    hidden_dims = hyperparameters["hidden_dims"]
    initialization_types = hyperparameters["initialization_types"]
    g_values = hyperparameters["g_values"]
    nonlinearities = hyperparameters["nonlinearities"]
    tasks = hyperparameters["tasks"]
    epochs = hyperparameters["epochs"]
    task_dataset_class = RNNTaskDataset
    batch_size = hyperparameters["batch_size"]
    learning_rate = hyperparameters["learning_rate"]
    max_retries = hyperparameters["max_retries"]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    if device == 'cuda' and nonlinearities[0] == 'relu':
        torch.cuda.set_device(0)
    if device == 'cuda' and nonlinearities[0] == 'tanh':
        torch.cuda.set_device(1)
    weights_folder = hyperparameters["weights_folder"]

    for hidden_dim in hidden_dims:
        for init_type in initialization_types:
            for g in g_values:
                for nonlinearity in nonlinearities:
                    for task in tasks:
                        retries = 0
                        converged = False
                        best_trial_loss = float('inf')  # Track the best loss across all trials
                        best_trial_weights = None  # Track the best weights across all trials

                        # Load the task dataset
                        task_dataset = task_dataset_class(n_trials=200, time=100, n_channels=2)
                        x, y = getattr(task_dataset, task)()
                        x_tensor = torch.tensor(x, dtype=torch.float32)
                        y_tensor = torch.tensor(y, dtype=torch.float32)
                        dataset = TensorDataset(x_tensor, y_tensor)
                        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

                        # Retry training if it doesn’t converge
                        while not converged and retries < max_retries:
                            print(f"Training: hidden_dim={hidden_dim}, init_type={init_type}, g={g}, nonlinearity={nonlinearity}, task={task}")

                            # Create the RNN model with the current hyperparameters
                            initialize_uniform = (init_type == "uniform")
                            initialize_normal = (init_type == "random")
                            model = MyRNN(input_dim=x.shape[2], hidden_dim=hidden_dim, output_dim=y.shape[2],
                                          nonlinearity=nonlinearity, initialize_uniform=initialize_uniform,
                                          initialize_normal=initialize_normal, g=g).to(device)

                            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                            criterion_fn = MaskedMSELoss()

                            best_loss = float('inf')
                            best_weights = None
                            epochs_since_improvement = 0
                            current_lr = learning_rate
                            loss_history = []  # Store losses for plotting

                            for epoch in range(epochs):
                                model.train()
                                running_loss = 0

                                for X_batch, y_batch in train_loader:
                                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                                    optimizer.zero_grad()
                                    y_pred, _ = model(X_batch)
                                    loss = criterion_fn(y_pred, y_batch)
                                    loss.backward()
                                    optimizer.step()

                                    running_loss += loss.item()

                                avg_loss = running_loss / len(train_loader)
                                loss_history.append(avg_loss)  # Record loss for this epoch

                                # Print the current loss at each epoch
                                if epoch % 20 == 0:
                                    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

                                # Check if current loss is better than the best loss
                                if avg_loss < best_loss:
                                    best_loss = avg_loss
                                    best_weights = model.state_dict()  # Save the best weights for this retry
                                    epochs_since_improvement = 0
                                    print(f"New best loss: {best_loss:.4f} at epoch {epoch+1}")
                                else:
                                    epochs_since_improvement += 1

                                # If no improvement for 50 epochs, reset weights and reduce learning rate
                                if epochs_since_improvement > 100:
                                    print(f"No improvement for 50 epochs. Reverting to best weights and reducing learning rate.")
                                    model.load_state_dict(best_weights)  # Restore best weights
                                    current_lr /= 2  # Reduce learning rate
                                    optimizer = optim.Adam(model.parameters(), lr=current_lr)
                                    epochs_since_improvement = 0

                                    if current_lr < 1e-5:
                                        print("Learning rate too low. Stopping training.")
                                        break

                                # Optionally check for convergence
                                if has_converged(loss_history):
                                    converged = True
                                    print(f"Converged at epoch {epoch+1} with best loss: {best_loss:.4f}")
                                    break

                            # Retry if the model didn't converge
                            if not converged:
                                retries += 1
                                print(f"Training failed to converge. Retrying... ({retries}/{max_retries})")

                                # Keep track of the best weights and loss from the failed trial
                                if best_loss < best_trial_loss:
                                    best_trial_loss = best_loss
                                    best_trial_weights = best_weights
                            else:
                                print(f"Training converged successfully!")

                                # Save the best weights
                                save_path = os.path.join(weights_folder, f"weights_{hidden_dim}_{init_type}_g{g}_{nonlinearity}_{task}.pt")
                                torch.save(best_weights, save_path)
                                print(f"Best model weights saved to: {save_path}")
                                break  # Exit the retry loop after successful convergence

                        # If all retries fail, save the best weights across all failed trials
                        if not converged:
                            print(f"All retries failed. Saving the best model from failed trials with loss {best_trial_loss:.4f}.")
                            save_path = os.path.join(weights_folder, f"best_failed_weights_{hidden_dim}_{init_type}_g{g}_{nonlinearity}_{task}.pt")
                            torch.save(best_trial_weights, save_path)
                            print(f"Best failed trial weights saved to: {save_path}")