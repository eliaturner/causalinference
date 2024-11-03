import numpy as np
import pandas as pd
from itertools import product

# Define hyperparameter ranges
hidden_dims = [16, 32, 64, 128, 256, 512]  # Hidden dimensions
g_values = [0.1, 0.5, 0.9]  # Connectivity strength
nonlinearities = ['tanh', 'relu']  # Activation functions

# Generate all combinations of hyperparameters
combinations = list(product(hidden_dims, g_values, nonlinearities))

# Fine-tuned coefficients to achieve balanced probabilities
beta_0 = -1.0  # Intercept (shift logit down)
beta_1 = 0.2   # Coefficient for relu activation
beta_2 = 0.1   # Coefficient for connectivity strength
beta_3 = 0.005 # Coefficient for hidden dimensions

# Calculate treatment probabilities
data = []
for hidden_dim, g, activation in combinations:
    # Encode activation function as 0 (tanh) or 1 (relu)
    activation_code = 1 if activation == 'relu' else 0

    # Calculate the logit
    logit = beta_0 + beta_1 * activation_code + beta_2 * g + beta_3 * hidden_dim
    print(f"Logit: {logit}")  # Optional: print to inspect logit values

    # Convert logit to probability using the sigmoid function
    prob_treatment = 1 / (1 + np.exp(-logit))

    # Store results
    data.append([hidden_dim, g, activation, prob_treatment])

# Create a DataFrame
df = pd.DataFrame(data, columns=["Hidden_Dim", "Connectivity_Strength",
                                 "Activation_Function", "Prob_Treatment"])

# Check for bias and distribution
print("Mean Probability of Treatment:", df["Prob_Treatment"].mean())
print("Probability Distribution:\n", df["Prob_Treatment"].describe())

# Save the DataFrame
df.to_csv("treatment_probabilities_without_task.csv", index=False)

# Display a sample of the DataFrame
print(df.head())
