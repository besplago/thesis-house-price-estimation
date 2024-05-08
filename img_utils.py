import os
import json
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from utils import *
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
from PIL import Image
import tensorflow as tf
from tqdm import tqdm


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


# Loads both JSON, floorplan and images
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




###### MODEL EVALUATION LINEAR ######
def print_scores_prices(real_prices, predicted_prices):
    """
    Returns the accuracy, precision, recall and f1-score of the model
    """
    print(f"R2 score: {r2_score(real_prices, predicted_prices):.2f}")
    print(f"Mean Absolute Error: {mae(real_prices, predicted_prices):.2f}")
    print(f"Mean Squared Error: {mse(real_prices, predicted_prices):.2f}")


def plot_predictions(test_images, actual_prices, predicted_prices):
    """
    Plot the predictions of the model
    """
    for i, (image, label, prediction) in enumerate(
        zip(test_images, actual_prices[0:9], predicted_prices[0:9])
    ):
        plt.subplot(3, 3, i + 1)
        plt.imshow(image)
        plt.title(f"Real: {label}\nPredicted: {prediction[0]:.0f}")
        plt.axis("off")
    plt.show()


def plot_hist(hist):
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()
    return None


