# Feed Forward Neural Network Example
# Note: This example does a bad job at predicting because the network hasn't been trained.

w = 0.1  # Initialize weight (w) with an arbitrary value; this represents the slope in a simple linear model.

def predict(input_value):
    """
    Make a prediction using a simple linear model.

    Args:
    input_value (float): The input value to the model.

    Returns:
    float: The prediction result, calculated as the product of the weight and the input.
    """
    return w * input_value  # Calculate prediction as weight multiplied by input.

# Test the model by making a prediction with an input value of 2
predicted_value = predict(2)
print(predicted_value)  # Output the prediction, which will print: 0.2

# Commentary:
# The current output is 0.2, but the target value is 4.
# This discrepancy indicates that the model's prediction is far off from the desired output.
