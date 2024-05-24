import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import json
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.cluster import KMeans

from PIL import Image
import tensorflow as tf
from tqdm import tqdm
import seaborn as sns


###### LOADING DATA ###########
class House:
    """Class to represent a house"""

    def __init__(self, **kwargs):
        # Textual Data
        self.__dict__.update(kwargs)


def load_jpg_and_json(folder_path: str) -> tuple[dict, np.array]:
    files = os.listdir(folder_path)
    jpg_file_path = None
    json_file_path = None

    # Find "0.jpg" and "data.json" specifically
    for file in files:
        if file == "0.jpg":
            jpg_file_path = os.path.join(folder_path, file)
        elif file == "data.json":
            json_file_path = os.path.join(folder_path, file)

    # Load the jpg
    image_data = cv2.imread(jpg_file_path)
    if image_data is None:
        raise Exception(f"Error loading image {jpg_file_path}")
    # Load the json
    try:
        with open(json_file_path, "r", encoding="utf-8") as file:
            json_data = json.load(file)
    except Exception as e:
        raise Exception(f"Error loading json {json_file_path}: {e}")
    return json_data, image_data

def load_houses(folder_path: str, max_houses: int = None):
    houses = []
    count = 0  # Counter to track the number of loaded houses
    errors: dict = {}
    for folder in os.listdir(folder_path):
        if max_houses is not None and count >= max_houses:
            break  # Stop loading houses if the maximum number is reached
        try:
            json_data, jpg = load_jpg_and_json(os.path.join(folder_path, folder))
            house = House(**json_data, image_floorplan=jpg)
            houses.append(house)
            count += 1
        except Exception as e:
            error_str = str(e)
            if error_str in errors:
                errors[error_str] += 1
            else:
                errors[error_str] = 1
            continue
    if errors:
        print("Errors encountered while loading houses:")
        for error, count in errors.items():
            print(f"{error}: {count} times")
    return houses

def load_data_and_images(folder_path, include_floorplan, include_images, max_houses):
    data = []
    iter = 0
    for subfolder, _, files in os.walk(folder_path):
        if iter == max_houses:
            break
        else:
            if subfolder == folder_path:
                continue  # Skip the main folder itself

            # Extract data from JSON file
            try:
                with open(os.path.join(subfolder, "data.json"), "r") as f:
                    json_data = json.load(f)
            except:
                print(f"Error loading JSON file for {subfolder}")
                # Go to the next subfolder
                continue

            # Extract image paths
            floorplan_path = os.path.join(subfolder, "0.jpg")
            image_paths = [
                os.path.join(subfolder, f"{i}.jpg")
                for i in range(1, len(files) - 2 if len(files) > 1 else 1)
            ]
            # Read floorplan image (assuming BGR color space)
            if include_floorplan:
                try:
                    floorplan = cv2.imread(floorplan_path)
                except FileNotFoundError:
                    print(f"Floorplan image not found for {subfolder}")
                    floorplan = None

            # Load other image paths as a NumPy array
            if include_images:
                try:
                    images = [cv2.imread(path) for path in image_paths]
                except:
                    print(f"Error loading images for {subfolder}")
                    images = None

            # Create dictionary with extracted data
            data_dict = {
                "url": json_data["url"],
                "address": json_data["address"],
                "postal_code": json_data["postal_code"],
                "type": json_data["type"],
                "price": json_data["price"],
                "size": json_data["size"],
                "basement_size": json_data["basement_size"],
                "rooms": json_data["rooms"],
                "year_built": json_data["year_built"],
                "year_rebuilt": json_data["year_rebuilt"],
                "energy_label": json_data["energy_label"],
                "image_floorplan": floorplan if include_floorplan else None,
                "images": images if include_images else None,
            }
            # Append data dictionary to the list
            data.append(data_dict)
        iter += 1

    # Create DataFrame from the list of dictionaries
    df = pd.DataFrame(data)

    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    # Drop "url" column
    df = df.drop(columns=["url", "address"])

    # Set the basement_size to 0 if it is NaN
    df["basement_size"] = df["basement_size"].fillna(0)

    # Set the year_rebuilt to year_built if it is NaN
    df["year_rebuilt"] = df["year_rebuilt"].fillna(
        df["year_built"]
    )

    # Encode the features: postal_code, type, energy_class
    df["postal_code"] = df["postal_code"].astype("category").cat.codes
    df["type"] = df["type"].astype("category").cat.codes
    df["energy_label"] = df["energy_label"].astype("category").cat.codes

    # Drop rows with "longitude" or "latitude" set to 0
    try: 
        df = df[(df["lng"] != 0) & (df["lat"] != 0)]

        # Turn "lat" and "lng" to floats
        df["lat"] = df["lat"].astype(float)
        df["lng"] = df["lng"].astype(float)
    except:
        pass

    try: 
        df = df[(df['longitude'] != 0) & (df['latitude'] != 0)]
        df['latitude'] = df['latitude'].astype(float)
        df['longitude'] = df['longitude'].astype(float)
    except:
        pass

    # Drop NaN values
    df = df.dropna()

    return df

def remove_outliers(df, column, z_score_threshold=3):
    """Remove outliers from a DataFrame based on a z-score threshold."""
    from scipy import stats

    z_scores = np.abs(stats.zscore(df[column]))
    df_no_outliers = df[z_scores < z_score_threshold]
    return df_no_outliers

def data_to_df(
    folder_paths: list[str], preprocess: bool = True, rm_outliers: bool = False
) -> list[pd.DataFrame]:
    """
    Load data from multiple folders and returns the DataFrames
    """
    dfs = []

    for folder_path in folder_paths:
        houses = load_houses(folder_path)
        data = []

        for house in tqdm(houses, desc=f"Processing {folder_path}"):
            data.append(house.__dict__)

        df = pd.DataFrame(data)
        dfs.append(df)

    if preprocess:
        dfs = [preprocess_data(df) for df in tqdm(dfs, desc="Preprocessing")]

    if rm_outliers:
        print("Removing outliers...")
        print(f"Datapoints before: {sum([len(df) for df in dfs])}")
        dfs = [remove_outliers(df, "price", 3) for df in tqdm(dfs, desc="Removing outliers")]
        print(f"Datapoints after: {sum([len(df) for df in dfs])}")

    return dfs



###### IMAGE PREPROCESSING ######
def convert_to_grayscale(images: np.array) -> np.array:
    gray_images = np.array(
        [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]
    )
    # reshape the images
    gray_images = np.array(
        [image.reshape(image.shape[0], image.shape[1], 1) for image in gray_images]
    )
    #
    rgb_images = np.array(
        [cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) for image in gray_images]
    )
    return gray_images

def convert_to_grayscale_rgb(images: np.array) -> np.array:
    gray_images = np.array(
        [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]
    )
    # reshape the images
    gray_images = np.array(
        [image.reshape(image.shape[0], image.shape[1], 1) for image in gray_images]
    )
    #
    rgb_images = np.array(
        [cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) for image in gray_images]
    )
    return rgb_images

def threshold_images(images: np.array) -> np.array:
    image_shape = images[0].shape
    if len(image_shape) == 3:  # RGB
        images = convert_to_grayscale(images)
        thresholded_images = np.array(
            [cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)[1] for image in images]
        )
        # Convert back to RGB
        thresholded_images = np.array(
            [cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) for image in thresholded_images]
        )
    else:  # GrayScale
        thresholded_images = np.array(
            [cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)[1] for image in images]
        )

    return thresholded_images

def resize_images(df, column_name: str, width: int, height: int) -> np.array:
    """
    Resize the images in a DataFrame to a specific width and height
    """
    resized_images = np.array(
        [
            cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)
            for image in df[column_name]
        ]
    )
    return resized_images

def preprocess_images(
    df: pd.DataFrame,
    column_name: str,
    width: int,
    height: int,
    resize: bool,
    gray_scale: bool,
    threshhold: bool,
) -> np.array:
    images = df[column_name]
    if resize:
        images = resize_images(df, column_name, width, height)
    if gray_scale:
        images = convert_to_grayscale(images)
    if threshhold:
        images = threshold_images(images)
    return images



###### MODEL TRAINING #######
def set_gpu():
    """
    Set the GPU to be used by the model
    """
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        # Set the GPU to be used
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
            tf.config.experimental.set_visible_devices(gpus[0], "GPU")
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)
    else:
        print("No GPU available")

def set_cpu():
    """
    Set the CPU to be used by the model
    """
    print("Setting CPU")
    tf.config.set_visible_devices([], "GPU")


########## REGRESSION UTILS ##########
def regression_stats(y_test: np.ndarray, y_pred: np.ndarray) -> tuple:
    """Calculate regression statistics."""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    r2 = r2_score(y_test, y_pred)
    percentage_error = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    return r2, mae, percentage_error, mse

def save_expected_predicted(test_prices, test_predictions, img_dir):
    #Set X and Y axis to [0, 9.000.000]
    #plt.xlim(0, 9999999)
    #plt.ylim(0, 9999999)
    plt.scatter(test_prices, test_predictions)
    plt.xlabel("Expected Price")
    plt.ylabel("Predicted Price")
    plt.title("Expected vs Predicted Price")
    try: 
        plt.plot([min(test_prices), max(test_prices)], [min(test_prices), max(test_prices)], color='red')
    except:
        pass
    plt.savefig(f"{img_dir}/expected_vs_predicted.png")
    plt.close()

def save_residuals(test_prices, test_predictions, img_dir):
    residuals = test_prices - test_predictions.reshape(-1)
    plt.scatter(test_predictions, residuals)
    try:
        plt.hlines(y=0, xmin=test_prices.min(), xmax=test_prices.max(), colors="r")
    except:
        pass
    plt.xlabel("Expected Price")
    plt.ylabel("Residuals")
    plt.title("Residuals")
    plt.savefig(f"{img_dir}/residuals.png")
    plt.close()

def get_saliency_map(model, image):
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    image = image.astype(np.float32)
    image = tf.convert_to_tensor(image)
    with tf.GradientTape() as tape:
        tape.watch(image)
        prediction = model(image)
    gradients = tape.gradient(prediction, image)
    gradients = tf.squeeze(gradients)
    gradients = tf.reduce_max(gradients, axis=-1)
    gradients = gradients.numpy()
    gradients = (gradients - np.min(gradients)) / (np.max(gradients) - np.min(gradients))
    return gradients

def save_worst_best_predictions(model, test_predictions, test_prices, test_images, img_dir):
    residuals = test_prices - test_predictions.reshape(-1)
    distances = np.abs(test_prices - test_predictions.reshape(-1))
    worst_predictions = np.argsort(distances)[-8:]
    best_predictions = np.argsort(distances)[:8]
    test_images = np.array(test_images)
    for i, idx in enumerate(worst_predictions):
        image = test_images[idx]
        price = test_prices[idx]
        prediction = test_predictions[idx]
        residual = residuals[idx]
        plt.imshow(image)
        textstr = '\n'.join((
            f"Price: {price}",
            f"Predicted Price: {prediction}",
            f"Residual: {residual}"
        ))
        plt.text(0.01, 0.99, textstr, fontsize=10, transform=plt.gcf().transFigure, verticalalignment='top')
        plt.axis("off")
        plt.savefig(f"{img_dir}/worst_{i}.png")
        plt.close()
        
        saliency_map = get_saliency_map(model, image)
        plt.imshow(saliency_map, cmap="hot")
        plt.axis("off")
        plt.savefig(f"{img_dir}/worst_saliency_map_{i}.png")
        plt.close()
        
    for i, idx in enumerate(best_predictions):
        image = test_images[idx]
        price = test_prices[idx]
        prediction = test_predictions[idx]
        residual = residuals[idx]
        plt.imshow(image)
        textstr = '\n'.join((
            f"Price: {price}",
            f"Predicted Price: {prediction}",
            f"Residual: {residual}"
        ))
        plt.text(0.01, 0.99, textstr, fontsize=10, transform=plt.gcf().transFigure, verticalalignment='top')
        plt.axis("off")
        plt.savefig(f"{img_dir}/best_{i}.png")
        plt.close()
        saliency_map = get_saliency_map(model, image)
        plt.imshow(saliency_map, cmap="hot")
        plt.axis("off")
        plt.savefig(f"{img_dir}/best_saliency_map_{i}.png")
        plt.close()

def save_features_importance(feature_importance, img_dir):
    #sort the feature_importance dict by value
    feature_importance = {k: v for k, v in sorted(feature_importance.items(), key=lambda item: item[1], reverse=True)}
    #add percentages to the bars
    plt.bar(feature_importance.keys(), feature_importance.values())
    #plt.bar_label = feature_importance.values()
    plt.title('Feature Importance')
    #Remove y-labels
    plt.ylabel('')
    plt.xticks(rotation=90)
    #Zoom out so that text is visible 
    plt.subplots_adjust(bottom=0.4)
    plt.savefig(f"{img_dir}/feature_importance.png")
    plt.close()

def save_worst_best(test_predictions, test_prices, test_features, model_dir):
    #Find the best predictions, and worst predictions. 
    #Save them in two dataframes. Save a latex of the dataframe in a txt-file 
    residuals = test_prices - test_predictions.reshape(-1)
    distances = np.abs(test_prices - test_predictions.reshape(-1))
    worst_predictions = np.argsort(distances)[-8:]
    best_predictions = np.argsort(distances)[:8]
    
    test_features_ = pd.DataFrame(test_features).copy()
    test_features_["Price"] = test_prices
    test_features_["Predicted Price"] = test_predictions
    test_features_["Residual"] = residuals
    test_features_['Absolute Distances'] = distances
    test_features_ = test_features_.sort_values(by="Absolute Distances", ascending=False)
    worst_df = test_features_.head(8)
    best_df = test_features_.tail(8)
    #save worst and best as latex in txt-file 
    worst_df.to_latex(f"{model_dir}/worst_predictions.txt")
    best_df.to_latex(f"{model_dir}/best_predictions.txt")
    
def save_reconstuctions(AE, test_predictions, test_prices, test_images, model_dir):
    n = 10
    reconstruction_errors = AE.calculate_ssim(test_images)
    best5 = np.argsort(reconstruction_errors)[:n]
    worst5 = np.argsort(reconstruction_errors)[::-1][:n]
    print(best5)
    print(worst5)

    #Calcuate and plot the correlation between residuals and reconstruction errors
    residuals = test_prices - test_predictions.reshape(-1)
    correlation = np.corrcoef(residuals, reconstruction_errors)
    plt.scatter(residuals, reconstruction_errors)
    plt.xlabel("Residuals")
    plt.ylabel("Reconstruction Errors")
    plt.title("Residuals vs Reconstruction Errors")
    textstr = '\n'.join((
        f"Correlation: {correlation[0][1]}"
    ))
    plt.text(0.01, 0.99, textstr, fontsize=10, transform=plt.gcf().transFigure, verticalalignment='top')
    plt.savefig(f"{model_dir}/Reconstruction_Correlation.png")
    plt.close()



    for i in range(n):
        idx = best5[i]
        image = test_images[idx]
        encoded_img = AE.encode(np.expand_dims(image, axis=0))
        decoded_img = AE.decode(encoded_img)
        encoded_img = np.squeeze(encoded_img)
        decoded_img = np.squeeze(decoded_img)
        #Turn decoded_img into intergers
        decoded_img = decoded_img.astype(int)
        fix, ax = plt.subplots(1, 2, figsize=(15, 5))
        ax[0].imshow(image)
        ax[0].set_title("Original Image")
        ax[1].imshow(decoded_img)
        ax[1].set_title("Reconstructed Image")
        #Set overall title as the price vs. the predicted price
        price = test_prices[idx]
        predicted_price = test_predictions[idx]
        textstr = '\n'.join((
            f"Price: {price}",
            f"Predicted Price: {predicted_price}"
        ))
        plt.text(0.01, 0.99, textstr, fontsize=10, transform=plt.gcf().transFigure, verticalalignment='top')
        plt.savefig(f"{model_dir}/best_reconstruction_{i}.png")
        plt.close()
        

    for i in range(n):
        idx = worst5[i]
        image = test_images[idx]
        encoded_img = AE.encode(np.expand_dims(image, axis=0))
        decoded_img = AE.decode(encoded_img)
        encoded_img = np.squeeze(encoded_img)
        decoded_img = np.squeeze(decoded_img)
        #Turn decoded_img into intergers
        decoded_img = decoded_img.astype(int)
        fix, ax = plt.subplots(1, 2, figsize=(15, 5))
        ax[0].imshow(image)
        ax[0].set_title("Original Image")
        ax[1].imshow(decoded_img)
        ax[1].set_title("Reconstructed Image")
        price = test_prices[idx]
        predicted_price = test_predictions[idx]
        textstr = '\n'.join((
            f"Price: {price}",
            f"Predicted Price: {predicted_price}"
        ))
        plt.text(0.01, 0.99, textstr, fontsize=10, transform=plt.gcf().transFigure, verticalalignment='top')
        plt.savefig(f"{model_dir}/worst_reconstruction_{i}.png")
        plt.close()



































# def plot_regression_results(model_name, y_test, y_pred):
#     """Plot the results of a regression model."""
#     # Plotting the test set results
#     plt.scatter(y_test, y_pred)

#     # Calculate residuals
#     residuals = y_pred - y_test

#     # Calculate distances from the perfect fit line
#     distances = np.abs(y_test - y_pred)

#     # Define color gradient based on distances
#     colors = distances / np.max(distances)  # Normalize distances to range [0, 1]
#     # colors = plt.cm.RdYlGn_r(colors)  # Reverse the colormap: green (furthest), red (closest)

#     # Plot true values vs predictions with color gradient
#     plt.scatter(y_test, y_pred, c=colors)
#     plt.xlabel("True values")
#     plt.ylabel("Predictions")
#     # Plot the perfect fit line
#     try:
#         plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], c="r")
#     except:
#         print("")
#     # Name the perfect fit line
#     plt.title(f"True values vs Predictions ({model_name})")
#     plt.colorbar(label="Distance from Diagonal")
#     plt.legend(["Test values", "Perfect fit"])
#     # plt.gcf().set_size_inches(3, 3)

#     plt.show()

#     # Plot residuals
#     plt.scatter(y_pred, residuals, c=colors)
#     try:
#         plt.hlines(y=0, xmin=y_test.min(), xmax=y_test.max(), colors="r")
#     except:
#         print("")
#     plt.title(f"Residual plot ({model_name})")
#     plt.xlabel("Predicted values")
#     plt.ylabel("Residuals")
#     plt.colorbar(label="Distance from Diagonal")
#     plt.legend(["Residuals", "Perfect fit"])
#     # plt.gcf().set_size_inches(3, 3)
#     plt.show()
#     return None


# def plot_regression_results_outliers(
#     model_name: str, y_test: np.ndarray, y_pred: np.ndarray, n_outliers: int = 5
# ):
#     """
#     Plot the results of a regression model with outliers highlighted.

#     Args:
#         model_name: str, name of the model
#         y_test: array, true values
#         y_pred: array, predicted values
#         n_outliers: int, number of outliers to highlight
#     """
#     # Plotting the test set results
#     plt.scatter(y_test, y_pred)

#     # Calculate residuals
#     residuals = y_pred - y_test

#     # Calculate distances from the perfect fit line
#     distances = np.abs(y_test - y_pred)

#     # Color everything grey
#     colors = np.full_like(distances, fill_value="grey", dtype="object")

#     # Highlight the n_outliers largest outliers
#     outlier_indices = np.argsort(-distances)[:n_outliers]
#     for idx in outlier_indices:
#         # Update the colors to highlight the outliers
#         colors[idx] = "red"

#     # Plot true values vs predictions with color gradient
#     plt.scatter(y_test, y_pred, c=colors, edgecolors="k")
#     plt.xlabel("True values")
#     plt.ylabel("Predictions")
#     # Plot the perfect fit line
#     plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], c="r")
#     # Name the perfect fit line
#     plt.title(

#         f"True values vs Predictions ({model_name})\n{n_outliers} largest outliers highlighted"
#     )
#     plt.legend(["Test values", "Perfect fit"])
#     plt.show()

#     # Plot residuals
#     plt.scatter(y_pred, residuals, c=colors, edgecolors="k")
#     plt.hlines(y=0, xmin=y_test.min(), xmax=y_test.max(), colors="r")
#     plt.xlabel("Predicted values")
#     plt.ylabel("Residuals")
#     plt.title(
#         f"Residual plot ({model_name})\n{n_outliers} largest outliers highlighted"
#     )
#     plt.legend(["Residuals", "Perfect fit"])
#     plt.show()


# def plot_regression_stats(y_test, y_pred):
#     """Calculate regression statistics."""
#     from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#     mse = mean_squared_error(y_test, y_pred)
#     mae = mean_absolute_error(y_test, y_pred)
#     mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
#     r2 = r2_score(y_test, y_pred)
#     percentage_error = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
#     print(f"R^2: {r2}")
#     print(f"Mean Absolute Error: {mae}")
#     print(f"Mean Percentage Error: {percentage_error}")
#     print(f"Mean Squared Error: {mse}")

#     return None
# def plot_feature_importance(model, X_train):
#     """Plot the feature importances of a model."""
#     try:
#         importances = model.feature_importances_
#         indices = np.argsort(importances)[::-1]
#         plt.bar(range(X_train.shape[1]), importances[indices])
#         plt.xticks(range(X_train.shape[1]), indices)
#         plt.title("Feature Importances")
#         # Add column names to x-axis
#         plt.xticks(range(X_train.shape[1]), X_train.columns[indices], rotation=90)
#         # Add percentage labels to y-axis
#         plt.yticks(np.arange(0, 1.1, step=0.1))
#         plt.ylabel("Importance")
#         # Add percentages to the bars
#         for index, value in enumerate(importances[indices]):
#             plt.text(index, value, f"{value:.2f}", ha="center", va="bottom")
#         plt.gcf().set_size_inches(3, 3)
#         plt.show()
#     except:
#         print(f"Could not plot feature importances for model {model}")
#     return None


# def eval_model(model, x_test, y_test):
#     y_pred = model.predict(x_test)
#     # check if multidimensional
#     if len(y_pred.shape) > 1:
#         y_pred = y_pred.flatten()
#     if y_pred[0] < 800000:
#         y_pred = np.exp(y_pred)
#         y_test = np.exp(y_test)
#     plot_regression_stats(y_test, y_pred)
#     # plot_regression_results("Model", y_test, y_pred)
#     plot_feature_importance(model, x_test)
#     return None

# ########## DATAFRAME UTILS  ##########
# def flatten_columns(df, column_name):
#     """Flatten a column of lists into a DataFrame."""
#     try:
#         df = pd.concat([df, df[column_name].apply(pd.Series)], axis=1)
#         # Set the column names of the new columns
#         df = df.drop(columns=[column_name])
#         # Turn the bow-column names into strings
#         df.columns = df.columns.astype(str)
#         # MAdd column names, based of column_name + index
#         # df.columns = [column_name + '_' + str(col) for col in df.columns]
#     except:
#         print(f"Could not flatten column {column_name}")
#     return df


# def prepare_features(df):
#     df = df[(df["postal_code"] >= 1000) & (df["postal_code"] <= 2920)]
#     df = df[df["type"] == "ejerlejlighed"]
#     df = df.drop(columns=["address"])
#     df = df.drop(columns=["type"])
#     df["basement_size"].fillna(0, inplace=True)
#     df["year_rebuilt"].fillna(df["year_built"], inplace=True)
#     df["energy_label"] = df["energy_label"].astype("category").cat.codes
#     df["postal_code"] = df["postal_code"].astype("category").cat.codes

#     scaler = StandardScaler()
#     df[
#         [
#             "size",
#             "rooms",
#             "year_built",
#             "year_rebuilt",
#             "basement_size",
#             "energy_label",
#             "postal_code",
#         ]
#     ] = scaler.fit_transform(
#         df[
#             [
#                 "size",
#                 "rooms",
#                 "year_built",
#                 "year_rebuilt",
#                 "basement_size",
#                 "energy_label",
#                 "postal_code",
#             ]
#         ]
#     )
#     # scaler = StandardScaler()
#     # df['price'] = scaler.fit_transform(df[['price']])

#     df.dropna(inplace=True)
#     df = remove_outliers(df, "price")
#     df = df.dropna()
#     return df

# ########## OTHER UTILS  ##########

# def print_DF_price_stats(df):
#     # Get the number of rows and columns #USE THE ONE BELOW INSTEAD
#     num_datapoints = df.shape[0]
#     mean_price = df["price"].mean()
#     median_price = df["price"].median()
#     min_price = df["price"].min()
#     max_price = df["price"].max()
#     std_price = df["price"].std()
#     print(f"Number of datapoints: {num_datapoints}")
#     print(f"Mean price: {mean_price}")
#     print(f"Median price: {median_price}")
#     print(f"Min price: {min_price}")
#     print(f"Max price: {max_price}")
#     print(f"Standard deviation of price:{std_price}")
#     return None


# def describe_df(df):
#     display(df.describe())
#     return None


# def plot_distributions(df):
#     for column in df.columns:
#         if column == "image_floorplan":
#             continue
#         try:
#             if column == "price":
#                 # Set intervals to millions 10 bins
#                 plt.hist(df[column], bins=90)
#                 # Cap the x-axis to 9 million
#                 plt.xlim(0, 9000000)
#             else:
#                 plt.hist(df[column])
#             plt.title(column)
#             plt.show()
#         except:
#             print(f"Could not plot {column}")
#     return None


# ########## MODELS  ##########
# def RF(x_train, y_train, x_test, y_test):
#     from sklearn.ensemble import RandomForestRegressor

#     model = RandomForestRegressor(n_estimators=100, random_state=42)
#     model.fit(x_train, y_train)
#     y_pred = model.predict(x_test)
#     # Plot the results
#     print("Results")
#     plot_regression_stats(y_test, y_pred)
#     plot_regression_results("RF", y_test, y_pred)
#     # Feature importance
#     plot_feature_importance(model, x_train)
#     return model


# def SVC(x_train, y_train, x_test, y_test):
#     from sklearn.svm import SVC

#     model = SVC(kernel="linear")
#     model.fit(x_train, y_train)
#     y_pred = model.predict(x_test)
#     # Plot the results
#     print("Results")
#     plot_regression_stats(y_test, y_pred)
#     plot_regression_results("SVC", y_test, y_pred)
#     # Feature importance
#     # feature_importance(model, x_train)
#     return model


# def XGB(x_train, y_train, x_test, y_test):
#     from xgboost import XGBRegressor

#     model = XGBRegressor()
#     model.fit(x_train, y_train)
#     y_pred = model.predict(x_test)
#     # Plot the results
#     print("Results")
#     plot_regression_stats(y_test, y_pred)
#     plot_regression_results("XGB", y_test, y_pred)
#     # Feature importance
#     plot_feature_importance(model, x_train)
#     return model


# ##########Error Tracing: ###############
# # Find the top 10 values with highest error
# def top_n_worst(y_test, y_pred, dataset, images, n):
#     """
#     Find the top 5 values with highest error
#     Locate them in the dataset.
#     Foreach of the top 5 values, find the 3 nearest neighbors
#     """
#     # Calculate residuals
#     residuals = y_pred - y_test
#     # Calculate distances from the perfect fit line
#     distances = np.abs(y_test - y_pred)
#     # Find the top 5 values with highest error
#     top_n_idx = np.argsort(distances)[-n:]
#     # Locate them in the dataset
#     top_n = dataset.iloc[top_n_idx]
#     for idx in top_n_idx:
#         print("True", y_test[idx])
#         print("Predicted", y_pred[idx])
#         print("Residual", residuals[idx])
#         print(
#             "Predicted",
#             y_pred[idx],
#             "True",
#             y_test[idx],
#             "Residual",
#             residuals[idx],
#             "Distance",
#             distances[idx],
#         )
#         # Find the 3 nearest neighbors
#         top_i_features = dataset.iloc[idx]
#         display(top_i_features.to_frame().T)
#         # display the image
#         img = images[idx]
#         plt.imshow(img)
#         plt.show()
#     return None


# def top_n_best(y_test, y_pred, dataset, n):
#     """
#     Find the top 5 values with lowest error
#     Locate them in the dataset.
#     Foreach of the top 5 values, find the 3 nearest neighbors
#     """
#     # Calculate residuals
#     residuals = y_pred - y_test
#     # Calculate distances from the perfect fit line
#     distances = np.abs(y_test - y_pred)
#     # Find the top 5 values with lowest error
#     top_n_idx = np.argsort(distances)[:n]
#     # Locate them in the dataset
#     top_n = dataset.iloc[top_n_idx]
#     # Foreach of the top 5 values, find the 3 nearest neighbors

#     for idx in top_n_idx:
#         print("Predicted", y_pred[idx], "True", y_test[idx], "Residual", residuals[idx])
#         # Find the 3 nearest neighbors
#         top_i = dataset.iloc[idx]
#         display(top_i.to_frame().T)
#         # display the image
#         plt.imshow(top_i["image_floorplan"])

#         plt.show()
#     return None


# def get_saliency_map(model, image):
#     with tf.GradientTape() as tape:
#         image = tf.convert_to_tensor(image, dtype=tf.float32)
#         image = tf.expand_dims(image, axis=0)
#         tape.watch(image)
#         predictions = model(image)

#     # Compute gradients of the output with respect to the input image
#     gradient = tape.gradient(predictions, image)
    
#     # Take absolute value of gradients to get saliency map
#     saliency_map = tf.abs(gradient)
    
#     # Reshape saliency map
#     saliency_map = tf.reshape(saliency_map, image.shape[1:])  # Remove batch dimension
    
#     # Normalize between 0 and 1
#     saliency_map /= tf.reduce_max(saliency_map)

#     # Set color channels to 0
#     saliency_map = saliency_map[:, :, 0]
#     return saliency_map


# def get_saliency_maps(model, images: np.ndarray):
#     #set_gpu()
#     #model = tf.keras.models.load_model(f"./{MODEL_NAME}")
#     saliency_maps = []
#     for image in images:
#         with tf.GradientTape() as tape:
#             image = tf.convert_to_tensor(image, dtype=tf.float32)
#             image = tf.expand_dims(image, axis=0)
#             tape.watch(image)
#             predictions = model(image)

#         # Compute gradients of the output with respect to the input image
#         gradient = tape.gradient(predictions, image)
        
#         # Take absolute value of gradients to get saliency map
#         saliency_map = tf.abs(gradient)
        
#         # Reshape saliency map
#         saliency_map = tf.reshape(saliency_map, image.shape[1:])  # Remove batch dimension
        
#         # Normalize between 0 and 1
#         saliency_map /= tf.reduce_max(saliency_map)

#         # Set color channels to 0
#         saliency_map = saliency_map[:, :, 0]

#         saliency_maps.append(saliency_map)

#     return saliency_maps

# def plot_saliency_maps(images):
#     fig, axes = plt.subplots(n_images, 2, figsize=(10, 5 * n_images))
#     saliency_maps = get_saliency_maps(images)

#     if len(images) > 1:
#         for i, image in enumerate(images):
#             # Plot the original image
#             axes[i, 0].imshow(image)
#             axes[i, 0].set_title("Original Image")
#             axes[i, 0].axis("off")
            
#             # Plot the saliency map
#             axes[i, 1].imshow(saliency_maps[i], cmap="plasma")
#             axes[i, 1].set_title("Saliency Map")
#             axes[i, 1].axis("off")
#     else:
#         # Plot the original image
#         axes[0].imshow(images[0])
#         axes[0].set_title("Original Image")
#         axes[0].axis("off")
        
#         # Plot the saliency map
#         axes[1].imshow(saliency_maps[0], cmap="plasma")
#         axes[1].set_title("Saliency Map")
#         axes[1].axis("off")

#     plt.show()






