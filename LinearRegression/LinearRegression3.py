# Feed Forward Neural Network Example
# This script demonstrates initializing a simple model, making predictions, and calculating loss to train the network.

# Initialize weight (w) with an arbitrary value. This represents the slope in a simple linear model.
w = 0.1
learning_rate = 0.1  # Set a learning rate to adjust the model's weight during the training process.

# Data setup: Define input values and their corresponding target values.
inputs = [1, 2, 3, 4]
targets = [2, 4, 6, 8]  # Define the target outputs for the training data.

def predict(input_value):
    """
    Predict the output using a simple linear model.

    Args:
    input_value (float): The input value for the model.

    Returns:
    float: The output result, calculated as the product of the weight and the input.
    """
    return w * input_value  # Multiply the input by the weight to make a prediction.

# Training loop to adjust the weight based on the model's performance.
for _ in range(25):  # Iteratively adjust weights 25 times.
    # Observe the behavior of the cost function over iterations to assess training effectiveness.
    # The cost should ideally decrease towards zero as the model learns from the training data.
    pred = [predict(i) for i in inputs]  # Compute predictions for each input value.

    errors = [t - p for p, t in zip(pred, targets)]  # Calculate the difference between targets and predictions.

    cost = sum(errors) / len(targets)  # Compute the mean error across all samples as the cost.

    print(f"Weight: {w:.2f}, Cost: {cost:.2f}")  # Output the current weight and cost for monitoring.

    w += learning_rate * cost  # Update the weight by adding a scaled cost, aiming to minimize future costs.

# Testing the trained network with new inputs to evaluate performance.
test_inputs = [5, 6]
test_targets = [10, 12]
pred = [predict(i) for i in test_inputs]  # Generate predictions for the test inputs.

# Output the results for the test inputs and compare them with the targets.
for i, t, p in zip(test_inputs, test_targets, pred):
    print(f"input:{i}, target:{t}, pred:{p:.4f}")
# The predictions should closely match the targets if the training was successful.

# Concluding Note:
# This example is straightforward due to its simplicity. For more complex scenarios involving larger datasets or more features,
# additional techniques such as batch processing and advanced optimization algorithms are recommended.
# Backpropagation in more intricate networks adjusts multiple weights simultaneously based on the error gradient.
