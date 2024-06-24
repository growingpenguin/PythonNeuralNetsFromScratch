# Feed Forward Neural Network Example
# Note: This script demonstrates initializing a simple model, making predictions, and calculating loss to train the network.

# Initialize weight (w) with an arbitrary value; this represents the slope in a simple linear model.
w = 0.1

# Data setup: Input values and their corresponding target values.
inputs = [1, 2, 3, 4]
targets = [2, 4, 6, 8]  # Target outputs for the training data

def predict(input_value):
    """
    Make a prediction using a simple linear model.

    Args:
    input_value (float): The input value to the model.

    Returns:
    float: The prediction result, calculated as the product of the weight and the input.
    """
    return w * input_value  # Calculate prediction as weight multiplied by input.

# Generate predictions for each input in the training dataset
pred = [predict(i) for i in inputs]

# Calculate errors as the difference between target values and predictions
errors = [t - p for p, t in zip(pred, targets)]

# Calculate the cost (mean squared error) as the average of these errors
cost = sum(errors) / len(targets)

# Output the model's weight and the calculated cost to understand how well the network is performing.
print(f"Weight: {w:.2f}, Cost: {cost:.2f}")
# The cost represents the average error across all samples. The goal is to reduce the cost as close to zero as possible, indicating better performance.
