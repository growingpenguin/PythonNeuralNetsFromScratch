# Feed Forward Neural Network Example
# This script demonstrates initializing a simple model, making predictions, and calculating loss to train the network.
'''
Theme: Bias
The goal of linear regression is to fit a regression line as best as we can to the point in the diagram
Previously, we assumed regression line intercepts the y-axis at zero
=> Challenge: Usually not the case. Data points can be anywhere in the diagram
Only changing the slope, the line will never fit the points as best as it can
'''
# Initialize weight (w) with an arbitrary value, representing the slope in a simple linear model.
w = 0.1
b = 0.3 # Arbitrary value of bias
learning_rate = 0.1  # Set an arbitrary learning rate to adjust the model's weight during training.

# Data setup: Define input values and their corresponding target values.
inputs = [1, 2, 3, 4]
# targets = [2, 4, 6, 8]  # Specify target outputs for the training data.
targets = [12, 14, 16, 18]  # Moved the target points up (Increased target values by 10)


def predict(input_value):
    """
    Make a prediction using a simple linear model.

    Args:
        input_value (float): The input value to the model.

    Returns:
        float: The prediction result, calculated as the product of the weight and the input.
    """
    return w * input_value + b  # Perform the prediction operation by multiplying the input by the model's weight.

# Training loop to adjust weights based on the model's performance.
for _ in range(100):  # Iterate through the training process 10 times to adjust weights.
    '''
    Observe the behavior of the cost function over iterations to assess training effectiveness.
    Expectation: Cost should approach zero as the model learns from the training data.
    '''
    pred = [predict(i) for i in inputs]  # Generate predictions for each input.

    # Calculate squared errors as the difference between actual targets and predictions.
    errors = [(p - t) ** 2 for p, t in zip(pred, targets)]

    # Compute the mean squared error (MSE) cost as the average of these squared errors.
    cost = sum(errors) / len(targets)  # Cost represents the average error across all samples.

    # Output the current weight and cost for monitoring the training progress.
    print(f"Weight: {w:.2f}, Bias:{b:.2f}, Cost: {cost:.2f}")

    errors_d = [2 * (p - t) for p, t in zip(pred, targets)] # Calculate derivatives of the error
    weight_d = [e * i for e, i in zip(errors_d, inputs)] # Multiply each error derivative by input to calculate weight adjustments
    bias_d = [e * 1 for e in errors_d] # bias deltas, bias_d is copy of errors_d

    # Update the weight based on the cost, aiming to reduce the cost in subsequent iterations.
    w -= learning_rate * sum(weight_d) / len(weight_d)  # Adjust the weight to minimize the cost.
    b -= learning_rate * sum(bias_d) / len(bias_d)

# Testing the trained network with new inputs to evaluate performance.
test_inputs = [5, 6]
# test_targets = [10, 12]
test_targets = [20, 22] # Moved the target points up (Increased target values by 10)
pred = [predict(i) for i in test_inputs]  # Generate predictions for the test inputs.

# Output the results for the test inputs and compare them with the targets.
for i, t, p in zip(test_inputs, test_targets, pred):
    print(f"input:{i}, target:{t}, pred:{p:.4f}")
# The predictions should closely match the targets if the training was successful.

'''
Bias?
Bias is just like another input to the neuron
Bias input is always one
Bias influences the y-intercept by changing the prediction function 
w * i => w * i + b
Result
Used bias to move the regression line up, now the cost goes down 
With 100 epochs, the prediction is pretty accurate
'''
'''
Cost Function:
(((w * i) + b) - t ) ** 2
Whole Process:
(1) Feed Forward 
w * i + b
Calculate the predictions for the neuron for all training data
(2) Cost
Calculate the cost over all the training data
(3) Back Propagation 
Calculate the Error Derivatives
Calculate the Weight Delta & Bias Delta 
Update the weight & bias

'''