import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np

import torch
import torch.nn as nn

from masked_MSE_loss import MaskedMSELoss


class MyRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, bias=True, dropout=0, nonlinearity='tanh', 
                 initialize_uniform=False, initialize_normal=False, g=1.0):
        super(MyRNN, self).__init__()
        
        self.hidden_dim = hidden_dim

        # Define RNN and output layer
        self.rnn = nn.RNN(input_dim, hidden_dim, bias=bias, nonlinearity=nonlinearity, batch_first=True, dropout=dropout)
        self.out = nn.Linear(hidden_dim, output_dim, bias=bias)
        
        # Initialize weights
        if initialize_normal:
            wrec = g * np.random.normal(loc=0, scale=1/np.sqrt(hidden_dim), size=(hidden_dim, hidden_dim))
            self.rnn.weight_hh_l0 = nn.Parameter(torch.from_numpy(wrec).float())

        if initialize_uniform:
            wrec = g * np.random.uniform(low=-1/np.sqrt(hidden_dim), high=1/np.sqrt(hidden_dim), size=(hidden_dim, hidden_dim))
            self.rnn.weight_hh_l0 = nn.Parameter(torch.from_numpy(wrec).float())

        self.noise_std = 0.01

    # def forward(self, x, hidden=None):
    #     out, hidden = self.rnn(x, hidden)
    #     out = self.out(out)
    #     return out, hidden

    def forward(self, x, hidden=None, inject_noise=False):
        out_list = []
        seq_len = x.size(1)  # Get the sequence length

        for t in range(seq_len):
            # Get the input for the current time step
            x_t = x[:, t].unsqueeze(1)

            # Step forward through the RNN
            out_t, hidden = self.rnn(x_t, hidden)

            # Inject noise into the hidden state if enabled
            if inject_noise and self.noise_std > 0:
                noise = torch.randn_like(hidden) * self.noise_std
                hidden = hidden + noise

            out_t = self.out(out_t)  # Compute output for this step
            out_list.append(out_t)

        # Concatenate outputs for all time steps
        out = torch.cat(out_list, dim=1)

        return out, hidden


def train(model, train_loader, val_loader=None, criterion='mse', device='cpu', lr=1e-3, epochs=100, log=True):
    model.to(device)
    train_loss_array = []
    val_loss_array = []

    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Set criterion based on input
    if criterion == 'mse':
        criterion_fn = MaskedMSELoss()
    elif criterion == 'ce':
        criterion_fn = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported criterion: {criterion}")
    
    start_time = time.time()
    
    for ep in range(epochs):  
        model.train()
        running_loss = 0
        
        for batch, (X, y_true) in enumerate(train_loader):
            X, y_true = X.to(device), y_true.to(device)
            y_pred, _ = model(X)  # Forward pass

            optimizer.zero_grad()
            loss = criterion_fn(y_pred, y_true)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss_array.append(running_loss / len(train_loader))

        if val_loader:
            val_loss = eval_model(model, val_loader, criterion_fn, device)
            val_loss_array.append(val_loss)

        if log:
            print(f'Epoch {ep + 1}',
                    f'Train Loss: {train_loss_array[-1]:.6f}')
                    # f'Val Loss: {val_loss_array[-1]:.4f}',
                    # f'Time: {time.time() - start_time:.1f}s')

        if train_loss_array[-1] < 1e-4:
            break

    return train_loss_array, val_loss_array


def eval_model(model, val_loader, criterion, device='cpu'):
    model.eval()
    running_loss = 0

    with torch.no_grad():
        for X, y_true in val_loader:
            X, y_true = X.to(device), y_true.to(device)
            y_pred, _ = model(X)  # Forward pass
            loss = criterion(y_pred, y_true)
            running_loss += loss.item()

    return running_loss / len(val_loader)


def simulate_with_lesions(model, X, y, percentage_of_lesions, criterion='mse', device='cpu'):
    # Move model and data to the appropriate device
    model.to(device)
    X, y_true = X.to(device), y.to(device)

    # Define the criterion function based on the input
    if criterion == 'mse':
        criterion_fn = MaskedMSELoss()
    elif criterion == 'ce':
        criterion_fn = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported criterion: {criterion}")

    # Store the original state_dict (weights) of the model to restore later
    original_state_dict = model.state_dict()

    try:
        # Clone the current recurrent weights to modify
        original_weights = model.rnn.weight_hh_l0.data.clone()

        # Calculate the number of weights to set to zero
        num_weights = original_weights.numel()
        num_lesions = int(percentage_of_lesions * num_weights)

        # Randomly select indices to lesion (set to zero)
        lesion_indices = np.random.choice(num_weights, num_lesions, replace=False)

        # Flatten the weights, set selected indices to zero
        flattened_weights = original_weights.view(-1)
        flattened_weights[lesion_indices] = 0

        # Assign the modified weights to the RNN
        model.rnn.weight_hh_l0.data = flattened_weights.view_as(original_weights)

        # Evaluate the model with lesioned weights
        model.eval()
        with torch.no_grad():
            y_pred, _ = model(X)
            loss = criterion_fn(y_pred, y_true)

    finally:
        # Restore the original state_dict to return the model to its original state
        model.load_state_dict(original_state_dict)

    return loss.item()


