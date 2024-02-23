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

import tensorflow as tf


###### LOADING DATA ###########
class House: 
  def __init__(self, address, postal_code, type, real_price, 
                size, basement_size, rooms, year_built, 
                year_rebuilt, energy_label, image_floorplan): 
    
    #Textual Data 
    self.address = address  
    self.postal_code = postal_code
    self.type = type
    self.price = real_price
    self.size = size
    self.basement_size = basement_size
    self.rooms = rooms
    self.year_built = year_built
    self.year_rebuilt = year_rebuilt
    self.energy_label = energy_label

    #Image Data 
    self.image_floorplan = image_floorplan
    
    #Predictions 
    self.predicted_price = None


def load_jpg_and_json(folder_path:str) -> (dict, np.ndarray):
  files = os.listdir(folder_path)
  jpg_file_path = None
  json_file_path = None

  # Find the jpg and json file in the folder
  for filename in files:
    if filename.endswith(".jpg"):
      jpg_file_path = os.path.join(folder_path, filename)
    elif filename.endswith(".json"):
      json_file_path = os.path.join(folder_path, filename)

  # Load the jpg
  image_data = cv2.imread(jpg_file_path)
  # Load the json
  with open(json_file_path, "r", encoding="utf-8") as file:
    json_data = json.load(file)

  if image_data is None:
    raise Exception(f"Error loading image {jpg_file_path}")
  if json_data is None:
    raise Exception(f"Error loading json {json_file_path}")

  return json_data, image_data

def create_house_instance(json_data, jpg): 
  address = json_data["address"]
  postal_code = json_data["postal_code"]
  type = json_data["type"]
  price = json_data["price"]
  size = json_data["size"]
  basement_size = json_data["basement_size"]
  rooms = json_data["rooms"]
  year_built = json_data["year_built"]
  year_rebuilt = json_data["year_rebuilt"] if json_data["year_rebuilt"] else None
  energy_label = json_data["energy_label"]
  image_floorplan = jpg

  house = House(address, postal_code, type, price, 
                size, basement_size, rooms, year_built, 
                year_rebuilt, energy_label, image_floorplan)
  return house

def load_houses(folder_path: str, max_houses: int = None):
    houses = []
    count = 0  # Counter to track the number of loaded houses
    for folder in os.listdir(folder_path):
        if max_houses is not None and count >= max_houses:
            break  # Stop loading houses if the maximum number is reached
        try:
            json_data, jpg = load_jpg_and_json(os.path.join(folder_path, folder))
            house = create_house_instance(json_data, jpg)
            houses.append(house)
            count += 1
        except Exception as e:
            continue
            #print(f"Error loading house {folder}: {e}")
    return houses

#If we want to work with a DF 
def data_to_DF(folder_path:str, max_houses)-> pd.DataFrame:
  houses = load_houses(folder_path, max_houses)
  data = []
  for house in houses:
    data.append([house.address, house.postal_code, house.type, house.price, 
                house.size, house.basement_size, house.rooms, house.year_built, 
                house.year_rebuilt, house.energy_label, house.image_floorplan])
  df = pd.DataFrame(data, columns = ["address", "postal_code", "type", "price", 
                "size", "basement_size", "rooms", "year_built", 
                "year_rebuilt", "energy_label", "image_floorplan"])
  return df


###### FEATURE PROCESSING ######
def preprocces_data(df: pd.DataFrame)-> pd.DataFrame:
  """
  Preprocess the data.
  """
  #df = df.drop(columns=["address"])
  #Feature Columns
  df['basement_size'] = df["basement_size"].fillna(0)
  df['year_rebuilt'] = df['year_rebuilt'].where(~df['year_rebuilt'].isna(), df['year_built']).astype(int)
  #df['type'] = df['type'].astype('category').cat.codes
  df['energy_label'] = df['energy_label'].astype('category').cat.codes
  #data.dropna(inplace=True)

  #Image Columns 
  #df['image_floorplan'] = df['image_floorplan'].apply(convert_to_grayscale)
  #Optimal: use ImageGenerator to augment the images#
  
  #Adding Labels 
  #df = (label_low_med_high(df, onehot=True))

  #Add a column that holds the image resolution
  df['image_resolution'] = df['image_floorplan'].apply(lambda x: x.shape)
  return df



##### LABEL ENCODING ######
def prices_to_n_labels(all_prices: np.array, prices: np.array, n_labels: int)-> np.array:
  """
  Convert the prices to n_labels

  Args:
    all_prices(`np.array`): All the prices
    prices(`np.array`): The prices we want to convert
    n_labels(`int`): The number of labels we want to convert the prices to

  Returns:
    `np.array`: The prices converted to n_labels
  """

  #Calculate the quantiles
  quantiles = [np.quantile(all_prices, i/n_labels) for i in range(1, n_labels)]
  print(quantiles)

  labels = [0 if price < quantiles[0] else n_labels-1 if price > quantiles[-1] else np.argmax([price < quantile for quantile in quantiles]) for price in prices]
  encoder = OneHotEncoder(sparse_output=False)
  one_hot_labels = encoder.fit_transform(np.array(labels).reshape(-1, 1))
  return one_hot_labels

def price_categories(all_prices, prices)-> np.array:
   # Calculate quantiles
  low_quantile = np.quantile(all_prices, 0.33)
  #print(low_quantile)
  high_quantile = np.quantile(all_prices, 0.66)
  #print(high_quantile)
  labels = [0 if price < low_quantile else 1 if price < high_quantile else 2 for price in prices]
  encoder = OneHotEncoder(sparse_output=False)
  one_hot_labels = encoder.fit_transform(np.array(labels).reshape(-1, 1))
  return one_hot_labels

def binary_labels(df: pd.DataFrame) -> pd.DataFrame:
  """
  Add labels-column to the data-points, based on prices. Simply version
  """
  mean = df["price"].mean()
  df["label"] = df["price"].apply(lambda x: 0 if x > mean else 1)
  return df

def normal_distribution_label(data: pd.DataFrame, num_labels:int)-> pd.DataFrame:
  """
  Add labels-column to the data-points, based on prices
  Creates labels based on a normal distribution around the data.
  That is, we have more labels the closer we are to the mean price, and less the further away we are.
  Return a data-Frame with the labels and the label codes. 
  """
  #We want to predict the price of the house 
  min = data['price'].min()
  first_quan = data['price'].quantile(0.25)
  mean = data['price'].mean()
  third_quan = data['price'].quantile(0.75)
  max = data['price'].max()
  #Create a normal distribution of the labels
  f1 = np.linspace(0, min, round(num_labels*0.023))
  f2 = np.linspace(min, first_quan, round(num_labels*0.14))
  f3 = np.linspace(first_quan, mean, round(num_labels*0.34))
  f4 = np.linspace(mean, third_quan, round(num_labels*0.34))
  f5 = np.linspace(third_quan,max, round(num_labels*0.14))
  f6 = np.linspace(max, max*2, round(num_labels*0.023))
  potential_labels = np.concatenate((f1, f2, f3, f4, f5, f6))

  #Create the label codes
  label_codes = [(i, label) for i, label in enumerate(potential_labels)]
  
  #Create the labels
  price_labels = []
  price_bracket = []
  for price in data['price']:
    diff = abs(potential_labels - price)
    index = np.argmin(diff)
    price_labels.append(index)
    left = potential_labels[index-1] if index > 0 else potential_labels[index]
    right = potential_labels[index+1] if index < len(potential_labels)-1 else potential_labels[index]
    price_bracket.append((left, right))

  data['label'] = price_labels
  data['price_bracket'] = price_bracket
  return data, label_codes

def label_low_med_high(df: pd.DataFrame, onehot:bool)-> pd.DataFrame:
  """
  Add labels-column to the data-points, based on prices. THREE labels: low, medium, high
  """
  price_ranges = {
    "low": (0,df['price'].quantile(0.33)),
    "med": (df['price'].quantile(0.33), df['price'].quantile(0.66)),
    "high": (df['price'].quantile(0.66), df['price'].max()), 
  }

  def label(price): 
    if price >= price_ranges['low'][0] and price<= price_ranges['low'][1]: 
      return 0
    elif price >= price_ranges['med'][0] and price <= price_ranges['med'][1]:
      return 1
    else: 
      return 2
  df['label_price'] = df['price'].apply(label)
  return df 



###### IMAGE PREPROCESSING ######
def resize_images(df, column_name:str, width:int, height:int)-> np.array:
  resized_images = np.array([cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR) for image in df[column_name]])
  return resized_images

def convert_to_grayscale(images: np.array)-> np.array:
  gray_images = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images])
  #reshape the images
  gray_images = np.array([image.reshape(image.shape[0], image.shape[1], 1) for image in gray_images])
  return gray_images

def threshold_images(images: np.array)-> np.array:
  image_shape = images[0].shape
  if len(image_shape) == 3: #RGB
    images = convert_to_grayscale(images)
    thresholded_images = np.array([cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)[1] for image in images])
    #Convert back to RGB 
    thresholded_images = np.array([cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) for image in thresholded_images])
  else: #GrayScale
    thresholded_images = np.array([cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)[1] for image in images])
    
  return thresholded_images

def preprocess_images(df:pd.DataFrame, column_name:str, width:int, height:int, resize:bool, gray_scale:bool, threshhold:bool)-> np.array:
  if resize:
    images = resize_images(df, column_name, width, height)
  if gray_scale:
    images = convert_to_grayscale(images)
  if threshhold:
    images = threshold_images(images)
  return images


###### MODEL UTILS #######
def save_model(model, name): 
  model.save(name)
  
def load_model(model_name): 
  model = tf.keras.models.load_model(model_name)
  return model


###### MODEL TRAINING #######
def set_gpu():
  gpus = tf.config.experimental.list_physical_devices('GPU')
  if gpus:
    # Set the GPU to be used
    try:
      tf.config.experimental.set_memory_growth(gpus[0], True)
      tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
      logical_gpus = tf.config.experimental.list_logical_devices('GPU')
      print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
      # Visible devices must be set before GPUs have been initialized
      print(e)
  else:
    print("No GPU available")

def set_cpu():
  print("Setting CPU")
  tf.config.set_visible_devices([], 'GPU')

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
  for i, (image, label, prediction) in enumerate(zip(test_images, actual_prices[0:9], predicted_prices[0:9])):
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


###### MODEL EVALUATION CLASSIFICATION ######
def label_score(predicted_labels, actual_labels): 
  """
  Returns the accuracy, precision, recall and f1-score of the model
  """
  accuracy = accuracy_score(actual_labels, predicted_labels)
  precision = precision_score(actual_labels, predicted_labels, average='weighted')
  recall = recall_score(actual_labels, predicted_labels, average='weighted')
  f1 = f1_score(actual_labels, predicted_labels, average='weighted')
  print(f"Accuracy: {accuracy:.2f}")
  print(f"Precision: {precision:.2f}")
  print(f"Recall: {recall:.2f}")
  cm = confusion_matrix(predicted_labels, actual_labels)
  sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
  plt.xlabel("Predicted")
  plt.ylabel("Actual")
  plt.show()
  return None  






