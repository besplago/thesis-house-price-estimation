"""Utilities for the regression notebook."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd


########## REGRESSION UTILS ##########
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
    plt.xlabel("True values")
    plt.ylabel("Predictions")
    # Plot the perfect fit line
    try:
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], c="r")
    except:
        print("")
    # Name the perfect fit line
    plt.title(f"True values vs Predictions ({model_name})")
    plt.colorbar(label="Distance from Diagonal")
    plt.legend(["Test values", "Perfect fit"])
    # plt.gcf().set_size_inches(3, 3)

    plt.show()

    # Plot residuals
    plt.scatter(y_pred, residuals, c=colors)
    try:
        plt.hlines(y=0, xmin=y_test.min(), xmax=y_test.max(), colors="r")
    except:
        print("")
    plt.title(f"Residual plot ({model_name})")
    plt.xlabel("Predicted values")
    plt.ylabel("Residuals")
    plt.colorbar(label="Distance from Diagonal")
    plt.legend(["Residuals", "Perfect fit"])
    # plt.gcf().set_size_inches(3, 3)
    plt.show()
    return None


def plot_regression_results_outliers(
    model_name: str, y_test: np.ndarray, y_pred: np.ndarray, n_outliers: int = 5
):
    """
    Plot the results of a regression model with outliers highlighted.

    Args:
        model_name: str, name of the model
        y_test: array, true values
        y_pred: array, predicted values
        n_outliers: int, number of outliers to highlight
    """
    # Plotting the test set results
    plt.scatter(y_test, y_pred)

    # Calculate residuals
    residuals = y_pred - y_test

    # Calculate distances from the perfect fit line
    distances = np.abs(y_test - y_pred)

    # Color everything grey
    colors = np.full_like(distances, fill_value="grey", dtype="object")

    # Highlight the n_outliers largest outliers
    outlier_indices = np.argsort(-distances)[:n_outliers]
    for idx in outlier_indices:
        # Update the colors to highlight the outliers
        colors[idx] = "red"

    # Plot true values vs predictions with color gradient
    plt.scatter(y_test, y_pred, c=colors, edgecolors="k")
    plt.xlabel("True values")
    plt.ylabel("Predictions")
    # Plot the perfect fit line
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], c="r")
    # Name the perfect fit line
    plt.title(
        f"True values vs Predictions ({model_name})\n{n_outliers} largest outliers highlighted"
    )
    plt.legend(["Test values", "Perfect fit"])
    plt.show()

    # Plot residuals
    plt.scatter(y_pred, residuals, c=colors, edgecolors="k")
    plt.hlines(y=0, xmin=y_test.min(), xmax=y_test.max(), colors="r")
    plt.xlabel("Predicted values")
    plt.ylabel("Residuals")
    plt.title(
        f"Residual plot ({model_name})\n{n_outliers} largest outliers highlighted"
    )
    plt.legend(["Residuals", "Perfect fit"])
    plt.show()


def plot_regression_stats(y_test, y_pred):
    """Calculate regression statistics."""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    r2 = r2_score(y_test, y_pred)
    percentage_error = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    print(f"R^2: {r2}")
    print(f"Mean Absolute Error: {mae}")
    print(f"Mean Percentage Error: {percentage_error}")
    print(f"Mean Squared Error: {mse}")

    return None


def regression_stats(y_test: np.ndarray, y_pred: np.ndarray) -> tuple:
    """Calculate regression statistics."""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    r2 = r2_score(y_test, y_pred)
    percentage_error = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    return r2, mae, percentage_error, mse


def plot_feature_importance(model, X_train):
    """Plot the feature importances of a model."""
    try:
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.bar(range(X_train.shape[1]), importances[indices])
        plt.xticks(range(X_train.shape[1]), indices)
        plt.title("Feature Importances")
        # Add column names to x-axis
        plt.xticks(range(X_train.shape[1]), X_train.columns[indices], rotation=90)
        # Add percentage labels to y-axis
        plt.yticks(np.arange(0, 1.1, step=0.1))
        plt.ylabel("Importance")
        # Add percentages to the bars
        for index, value in enumerate(importances[indices]):
            plt.text(index, value, f"{value:.2f}", ha="center", va="bottom")
        plt.gcf().set_size_inches(3, 3)
        plt.show()
    except:
        print(f"Could not plot feature importances for model {model}")
    return None


def eval_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    # check if multidimensional
    if len(y_pred.shape) > 1:
        y_pred = y_pred.flatten()
    if y_pred[0] < 800000:
        y_pred = np.exp(y_pred)
        y_test = np.exp(y_test)
    plot_regression_stats(y_test, y_pred)
    # plot_regression_results("Model", y_test, y_pred)
    plot_feature_importance(model, x_test)
    return None


########## DATAFRAME UTILS  ##########
def flatten_columns(df, column_name):
    """Flatten a column of lists into a DataFrame."""
    try:
        df = pd.concat([df, df[column_name].apply(pd.Series)], axis=1)
        # Set the column names of the new columns
        df = df.drop(columns=[column_name])
        # Turn the bow-column names into strings
        df.columns = df.columns.astype(str)
        # MAdd column names, based of column_name + index
        # df.columns = [column_name + '_' + str(col) for col in df.columns]
    except:
        print(f"Could not flatten column {column_name}")
    return df


def prepare_features(df):
    df = df[(df["postal_code"] >= 1000) & (df["postal_code"] <= 2920)]
    df = df[df["type"] == "ejerlejlighed"]
    df = df.drop(columns=["address"])
    df = df.drop(columns=["type"])
    df["basement_size"].fillna(0, inplace=True)
    df["year_rebuilt"].fillna(df["year_built"], inplace=True)
    df["energy_label"] = df["energy_label"].astype("category").cat.codes
    df["postal_code"] = df["postal_code"].astype("category").cat.codes

    scaler = StandardScaler()
    df[
        [
            "size",
            "rooms",
            "year_built",
            "year_rebuilt",
            "basement_size",
            "energy_label",
            "postal_code",
        ]
    ] = scaler.fit_transform(
        df[
            [
                "size",
                "rooms",
                "year_built",
                "year_rebuilt",
                "basement_size",
                "energy_label",
                "postal_code",
            ]
        ]
    )
    # scaler = StandardScaler()
    # df['price'] = scaler.fit_transform(df[['price']])

    df.dropna(inplace=True)
    df = remove_outliers(df, "price")
    df = df.dropna()
    return df


########## OTHER UTILS  ##########
def remove_outliers(df, column, z_score_threshold=3):
    """Remove outliers from a DataFrame based on a z-score threshold."""
    from scipy import stats

    z_scores = np.abs(stats.zscore(df[column]))
    df_no_outliers = df[z_scores < z_score_threshold]
    return df_no_outliers


def print_DF_price_stats(df):
    # Get the number of rows and columns #USE THE ONE BELOW INSTEAD
    num_datapoints = df.shape[0]
    mean_price = df["price"].mean()
    median_price = df["price"].median()
    min_price = df["price"].min()
    max_price = df["price"].max()
    std_price = df["price"].std()
    print(f"Number of datapoints: {num_datapoints}")
    print(f"Mean price: {mean_price}")
    print(f"Median price: {median_price}")
    print(f"Min price: {min_price}")
    print(f"Max price: {max_price}")
    print(f"Standard deviation of price:{std_price}")
    return None


def describe_df(df):
    display(df.describe())
    return None


def plot_distributions(df):
    for column in df.columns:
        if column == "image_floorplan":
            continue
        try:
            if column == "price":
                # Set intervals to millions 10 bins
                plt.hist(df[column], bins=90)
                # Cap the x-axis to 9 million
                plt.xlim(0, 9000000)
            else:
                plt.hist(df[column])
            plt.title(column)
            plt.show()
        except:
            print(f"Could not plot {column}")
    return None


########## MODELS  ##########
def RF(x_train, y_train, x_test, y_test):
    from sklearn.ensemble import RandomForestRegressor

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    # Plot the results
    print("Results")
    plot_regression_stats(y_test, y_pred)
    plot_regression_results("RF", y_test, y_pred)
    # Feature importance
    plot_feature_importance(model, x_train)
    return model


def SVC(x_train, y_train, x_test, y_test):
    from sklearn.svm import SVC

    model = SVC(kernel="linear")
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    # Plot the results
    print("Results")
    plot_regression_stats(y_test, y_pred)
    plot_regression_results("SVC", y_test, y_pred)
    # Feature importance
    # feature_importance(model, x_train)
    return model


def XGB(x_train, y_train, x_test, y_test):
    from xgboost import XGBRegressor

    model = XGBRegressor()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    # Plot the results
    print("Results")
    plot_regression_stats(y_test, y_pred)
    plot_regression_results("XGB", y_test, y_pred)
    # Feature importance
    plot_feature_importance(model, x_train)
    return model


##########Error Tracing: ###############
# Find the top 10 values with highest error
def top_n_worst(y_test, y_pred, dataset, images, n):
    """
    Find the top 5 values with highest error
    Locate them in the dataset.
    Foreach of the top 5 values, find the 3 nearest neighbors
    """
    # Calculate residuals
    residuals = y_pred - y_test
    # Calculate distances from the perfect fit line
    distances = np.abs(y_test - y_pred)
    # Find the top 5 values with highest error
    top_n_idx = np.argsort(distances)[-n:]
    # Locate them in the dataset
    top_n = dataset.iloc[top_n_idx]
    for idx in top_n_idx:
        print("True", y_test[idx])
        print("Predicted", y_pred[idx])
        print("Residual", residuals[idx])
        print(
            "Predicted",
            y_pred[idx],
            "True",
            y_test[idx],
            "Residual",
            residuals[idx],
            "Distance",
            distances[idx],
        )
        # Find the 3 nearest neighbors
        top_i_features = dataset.iloc[idx]
        display(top_i_features.to_frame().T)
        # display the image
        img = images[idx]
        plt.imshow(img)
        plt.show()
    return None


def top_n_best(y_test, y_pred, dataset, n):
    """
    Find the top 5 values with lowest error
    Locate them in the dataset.
    Foreach of the top 5 values, find the 3 nearest neighbors
    """
    # Calculate residuals
    residuals = y_pred - y_test
    # Calculate distances from the perfect fit line
    distances = np.abs(y_test - y_pred)
    # Find the top 5 values with lowest error
    top_n_idx = np.argsort(distances)[:n]
    # Locate them in the dataset
    top_n = dataset.iloc[top_n_idx]
    # Foreach of the top 5 values, find the 3 nearest neighbors

    for idx in top_n_idx:
        print("Predicted", y_pred[idx], "True", y_test[idx], "Residual", residuals[idx])
        # Find the 3 nearest neighbors
        top_i = dataset.iloc[idx]
        display(top_i.to_frame().T)
        # display the image
        plt.imshow(top_i["image_floorplan"])

        plt.show()
    return None
