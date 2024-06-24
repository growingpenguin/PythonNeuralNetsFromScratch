# Feed Forward Neural Network Example
# This script demonstrates initializing a simple model, making predictions, and calculating loss to train the network.

# Initialize weight (w) with an arbitrary value, representing the slope in a simple linear model.
w = 0.1
learning_rate = 0.1  # Set an arbitrary learning rate to adjust the model's weight during training.

# Data setup: Define input values and their corresponding target values.
inputs = [1, 2, 3, 4]
targets = [2, 4, 6, 8]  # Specify target outputs for the training data.


def predict(input_value):
    """
    Make a prediction using a simple linear model.

    Args:
    input_value (float): The input value to the model.

    Returns:
    float: The prediction result, calculated as the product of the weight and the input.
    """
    return w * input_value  # Perform the prediction operation by multiplying the input by the model's weight.


# Training loop to adjust weights based on the model's performance.
for _ in range(25):  # Iterate through the training process 25 times to adjust weights.
    '''
    Observe the behavior of the cost function over iterations to assess training effectiveness.
    Expectation: Cost should approach zero as the model learns from the training data.
    Observation: Cost values fluctuate and show an oscillating pattern, indicating unstable learning.
    Issue: Weight adjustments may be too aggressive.
    Solution: Consider reducing the learning rate for more gradual updates, potentially improving stability.
    '''
    # Generate predictions for each input.
    pred = [predict(i) for i in inputs]

    # Calculate errors as the difference between actual targets and predictions.
    errors = [t - p for p, t in zip(pred, targets)]

    # Compute the mean squared error cost as the average of these squared differences.
    cost = sum(errors) / len(targets)  # Cost represents the average error across all samples.

    # Output the current weight and cost for monitoring the training progress.
    print(f"Weight: {w:.2f}, Cost: {cost:.2f}")

    # Update the weight based on the cost, aiming to reduce the cost in subsequent iterations.
    w += learning_rate * cost  # Adjust the weight to minimize the cost.

    # Insights: Smaller weight adjustments can lead to a more consistent decrease in cost.
    # As the network becomes more complex, the mechanism of backpropagation will adjust multiple weights.

# Additional notes:
# - This example works efficiently even with 1000 iterations due to its simplicity.
# - In more complex scenarios, other strategies like batch processing and advanced optimization algorithms may be required.
# - Backpropagation in more complex networks involves adjusting multiple weights based on gradients calculated for each layer.
