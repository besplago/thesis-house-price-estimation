"""Utilities for the regression notebook."""
import numpy as np
import matplotlib.pyplot as plt

def plot_regression_results(model_name, y_test, y_pred):
    """Plot the results of a regression model."""
    # Plotting the test set results
    plt.scatter(y_test, y_pred)

    # Calculate residuals
    residuals = y_pred - y_test

    # Calculate distances from the perfect fit line
    distances = np.abs(y_test - y_pred)

    # Define color gradient based on distances
    colors = distances / np.max(distances)  # Normalize distances to range [0, 1]
    # colors = plt.cm.RdYlGn_r(colors)  # Reverse the colormap: green (furthest), red (closest)

    # Plot true values vs predictions with color gradient
    plt.scatter(y_test, y_pred, c=colors)
    plt.xlabel('True values')
    plt.ylabel('Predictions')
    # Plot the perfect fit line
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], c='r')
    # Name the perfect fit line
    plt.title(f'True values vs Predictions ({model_name})')
    plt.colorbar(label='Distance from Diagonal')
    plt.legend(['Test values', 'Perfect fit'])
    plt.show()

    # Plot residuals
    plt.scatter(y_pred, residuals, c=colors)
    plt.hlines(y=0, xmin=y_test.min(), xmax=y_test.max(), colors='r')
    plt.title(f'Residual plot ({model_name})')
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.colorbar(label='Distance from Diagonal')
    plt.legend(['Residuals', 'Perfect fit'])
    plt.show()
