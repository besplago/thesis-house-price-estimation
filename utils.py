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


def regression_stats(y_test, y_pred):
    """Calculate regression statistics."""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')
    print(f'Mean Absolute Error: {mae}')
    print(f'R^2: {r2}')
    return None

def remove_outliers(df, column, z_score_threshold=2):
    """Remove outliers from a DataFrame based on a z-score threshold."""
    from scipy import stats
    z_scores = np.abs(stats.zscore(df[column]))
    df_no_outliers = df[z_scores < z_score_threshold]
    return df_no_outliers


def print_DF_price_stats(df): 
    # Get the number of rows and columns
    num_datapoints = df.shape[0]
    mean_price = df['price'].mean()
    median_price = df['price'].median()
    min_price = df['price'].min()
    max_price = df['price'].max()
    std_price = df['price'].std()
    print(f"Number of datapoints: {num_datapoints}")
    print(f"Mean price: {mean_price}")
    print(f"Median price: {median_price}")
    print(f"Min price: {min_price}")
    print(f"Max price: {max_price}")
    print(f"Standard deviation of price:{std_price}")
    return None


