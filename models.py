import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
from IPython.display import display

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras import regularizers

from keras import losses, Model

from xgboost import XGBRegressor
from vit_keras import vit, utils
from sklearn.preprocessing import StandardScaler
from utils import (
    plot_regression_results,
    plot_regression_stats,
    plot_feature_importance,
    prepare_features,
    eval_model,
)
from img_utils import preprocess_images, create_bow_representation, set_gpu, data_to_df, set_cpu

from tensorflow.keras.applications import (
    MobileNet,
    MobileNetV3Large,
    MobileNetV3Small,
    MobileNetV2,
)
from tensorflow.keras.applications import ResNet50, ResNet101, ResNet152, ResNet50V2
from tensorflow.keras.applications import InceptionV3, InceptionResNetV2
from tensorflow.keras.applications import DenseNet121, DenseNet169, DenseNet201
from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.keras.applications import NASNetLarge, NASNetMobile
from tensorflow.keras.applications import (
    EfficientNetB0,
    EfficientNetB1,
    EfficientNetB2,
    EfficientNetB3,
    EfficientNetB4,
    EfficientNetB5,
    EfficientNetB6,
    EfficientNetB7,
)
from tensorflow.keras.applications import (
    EfficientNetV2B0,
    EfficientNetV2M,
    EfficientNetV2L,
)
from tensorflow.keras.applications import (
    EfficientNetV2S,
    EfficientNetV2M,
    EfficientNetV2L,
)
from tensorflow.keras.applications import NASNetMobile, NASNetLarge


#### Feature Models ####
def RF(x_train, y_train):
    

    # Format the 4-dimensional input to 2-dimensional
    try: 
        x_train = x_train.reshape(x_train.shape[0], -1)
    except:
        pass
    try:
        x_test = x_test.reshape(x_test.shape[0], -1)
    except:
        pass

    #GridSearch
    gridSearch = False
    if gridSearch:
        param_grid = {
            "n_estimators": [100, 200, 400, 800],  # Number of trees in the forest
            "max_depth": [5, 10, 15, 20, 40],  # Maximum depth of individual trees
            "min_samples_split": [2, 4, 8, 16],  # Minimum samples required to split a node
        }
        model = GridSearchCV(RandomForestRegressor(), param_grid, cv=5)
    else:
        model = RandomForestRegressor(n_estimators=100, max_depth=10)
    model.fit(x_train, y_train)
    return model


def SVC(x_train, y_train, x_test, y_test):
    from sklearn.svm import SVC

    model = SVC(kernel="linear")
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print("Results")
    eval_model(y_test, y_pred)
    return model


def XGB(x_train, y_train, x_test, y_test):
    model = XGBRegressor(
        n_estimators=1000,
        max_depth=7,
        eta=0.1,
        subsample=0.7,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        enable_categorical=True,
    )
    fit_history = model.fit(x_train, y_train)
    eval_model(model, x_test, y_test)
    return model, fit_history


def neural_network(x_train, y_train, x_test, y_test):
    # y_train = np.log(y_train)
    # y_test = np.log(y_test)
    scale = False
    if scale:
        scale = StandardScaler()
        x_train = scale.fit_transform(x_train)
        x_test = scale.transform(x_test)

    model = Sequential()
    # Adding layers
    num_features = x_train.shape[1]
    model.add(Dense(9, input_dim=num_features, activation="relu"))
    model.add(Dense(9, activation="relu"))
    model.add(Dense(9, activation="relu"))
    model.add(Dense(9, activation="relu"))
    model.add(Dense(9, activation="relu"))
    model.add(Dense(1, activation="linear"))
    # Compiling and fitting
    model.compile(
        optimizer=Adam(learning_rate=0.02),
        loss="mean_squared_error",
        metrics=["mean_absolute_error"],
    )

    fit_history = model.fit(
        x_train,
        y_train,
        epochs=300,
        batch_size=64,
        validation_data=(x_test, y_test),
        callbacks=EarlyStopping(
            monitor="val_loss", patience=9, restore_best_weights=True
        ),
        verbose=0,
    )
    print("Test score: ", model.evaluate(x_test, y_test))
    y_pred = model.predict(x_test).flatten()
    eval_model(y_test, y_pred)
    return model, fit_history


#### Image Models ####
def CNN_model(
    # pretrained_model, custom_layers, train_images, y_train, validation_images, y_valid
    pretrained_model: object,
    custom_layers: bool,
    train_images: np.array,
    y_train: np.array,
    validation_images: np.array,
    y_valid: np.array,
):
    # Load the Pretrained Model
    target_width = train_images[0].shape[0]
    target_height = train_images[0].shape[1]
    input_shape = (target_width, target_height, 3)
    base_model = pretrained_model(
        weights="imagenet", include_top=False, input_shape=input_shape
    )

    # Freeze the pretrained weights
    for layer in base_model.layers:
        layer.trainable = False

    # Create the model with the pretrained model and a new classification layer
    if custom_layers:
        model = Sequential(
            [
                base_model,
                Flatten(),
                Dense(512, activation="relu", kernel_regularizer=regularizers.l1(0.01)),
                # BatchNormalization(),
                layers.Dropout(0.1),
                Dense(256, activation="relu", kernel_regularizer=regularizers.l1(0.01)),
                layers.Dropout(0.1),
                Dense(128, activation="relu"),
                Dense(64, activation="relu"),
                Dense(1, activation="linear"),
            ]
        )
    else:
        model = Sequential([base_model, Flatten(), Dense(1, activation="linear")])

    print("Compiling Model")
    # Check which type of model we are building
    model.compile(
        optimizer="Adam", loss="mean_absolute_error", metrics=["mean_absolute_error"]
    )

    print("Fitting Model")
    # Fit the model
    fit_history = model.fit(
        train_images,
        y_train,
        epochs=100,
        validation_data=(validation_images, y_valid),
        callbacks=[EarlyStopping(patience=30, restore_best_weights=True)],
    )
    return model, fit_history


def N_CNN_model(
    # pretrained_model, train_images, y_train, validation_images, y_valid, n
    pretrained_model: object,
    train_images: np.array,
    y_train: np.array,
    validation_images: np.array,
    y_valid: np.array,
    n: int,
):
    # Step 1: Make N splits
    splits = list(zip(np.array_split(train_images, n), np.array_split(y_train, n)))
    # Step 2: Train N models. Model(i) is trained on n-1 splits
    models = []
    fit_histories = []
    for i in range(n):
        # Train the model on n-1 splits
        train_images = np.concatenate(
            [split[0] for j, split in enumerate(splits) if j != i]
        )
        y_train = np.concatenate([split[1] for j, split in enumerate(splits) if j != i])
        model, fit_history = CNN_model(
            pretrained_model, True, train_images, y_train, validation_images, y_valid
        )
        models.append(model)
        fit_histories.append(fit_history)
    return models, fit_histories



#### AutoEncoder ####
class Denoise(Model):
    def __init__(self):
        super(Denoise, self).__init__()
        self.encoder = tf.keras.Sequential(
            [
                layers.Input(shape=(448, 448, 3)),
                layers.Conv2D(16, (3, 3), activation="relu", padding="same", strides=2),
                layers.Conv2D(8, (3, 3), activation="relu", padding="same", strides=2),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                layers.Conv2DTranspose(
                    8, kernel_size=3, strides=2, activation="relu", padding="same"
                ),
                layers.Conv2DTranspose(
                    16, kernel_size=3, strides=2, activation="relu", padding="same"
                ),
                layers.Conv2D(
                    1, kernel_size=(3, 3), activation="sigmoid", padding="same"
                ),
            ]
        )

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def calculate_error(self, input_images):
        reconstruction_errors = []
        for image in input_images:
            # Expand the single image to a batch dimension
            image_batch = tf.expand_dims(image, axis=0)
            decoded = self.__call__(image_batch)  # Call the model with the batch
            error = tf.keras.losses.MeanSquaredError()(image_batch, decoded)
            reconstruction_errors.append(tf.reduce_mean(error))
        reconstruction_errors = np.array(reconstruction_errors)
        return reconstruction_errors


def autoEncoder(train_images):
    train_images_scaled = train_images / 255.0
    train_img, test_img = train_test_split(
        train_images_scaled, test_size=0.1, random_state=42
    )

    autoencoder = Denoise()
    autoencoder.compile(optimizer="adam", loss=losses.MeanSquaredError())
    autoencoder.fit(
        train_img,
        train_img,
        epochs=10,
        shuffle=True,
        validation_data=(test_img, test_img),
    )
    return autoencoder


#### Ensemble Models ####
class CNN_RF:
    def __init__(self, image_model):
        self.image_model = image_model


    #Fit that needs images, features. Predict images, features
    def fit(self, train_images, train_features, train_y):
        # Get the image predictions
        train_image_predictions = self.image_model.predict(train_images).flatten()

        # Concatenate the predictions
        train_input = np.column_stack((train_image_predictions, train_features))
        train_input = pd.DataFrame(train_input)
        train_input.columns = ["image_predictions"] + list(train_features.columns)

        # Train the model
        self.model = RandomForestRegressor(n_estimators=100, max_depth=10)
        self.model.fit(train_input, train_y)
        self.feature_importances_ = self.model.fit(train_input, train_y).feature_importances_
        self.feature_importances_ = (dict(zip(train_input.columns, self.feature_importances_)))
        
    def predict(self, test_images, test_features):
        # Get the image predictions
        test_image_predictions = self.image_model.predict(test_images).flatten()

        # Concatenate the predictions
        test_input = np.column_stack((test_image_predictions, test_features))

        # Predict the prices
        return self.model.predict(test_input)

def CNN_RF_model(
    image_model,
    train_images,
    train_features, 
    train_y,
):
    CNN_RF_ = CNN_RF(image_model)
    CNN_RF_.fit(train_images, train_features, train_y)
    return CNN_RF_

class CNN_AE_RF:
    def __init__(self, image_model):
        self.image_model = image_model
        self.autoEncoder_ = None

    def fit(self, train_images, train_features, train_y):
        #Calculate the reconstruction error
        self.autoEncoder_ = autoEncoder(train_images)
        reconstruction_error = self.autoEncoder_.calculate_error(train_images)

        # Get the image predictions
        train_image_predictions = self.image_model.predict(train_images).flatten()
        train_input = np.column_stack((train_image_predictions, reconstruction_error, train_features))
        train_input = pd.DataFrame(train_input)
        train_input.columns = ["image_predictions", "reconstruction_error"] + list(train_features.columns)

        self.model = RandomForestRegressor(n_estimators=100, max_depth=10)
        self.model.fit(train_input, train_y)
        self.feature_importances_ = self.model.fit(train_input, train_y).feature_importances_
        self.feature_importances_ = (dict(zip(train_input.columns, self.feature_importances_)))
        
    def get_error(self, image):
        return self.autoEncoder_.calculate_error(image)

    def predict(self, test_images, test_features):
        test_image_predictions = self.image_model.predict(test_images).flatten()
        reconstruction_error = self.autoEncoder_.calculate_error(test_images)
        test_input = np.column_stack((test_image_predictions,reconstruction_error, test_features))
        return self.model.predict(test_input)

def CNN_AE_RF_model(      
    image_model,
    train_images,
    train_features,
    train_y,
):
    CNN_AE_RF_model = CNN_AE_RF(image_model)
    CNN_AE_RF_model.fit(train_images, train_features, train_y)
    return CNN_AE_RF_model


class N_CNN_RF:
    #Setup a class that train N-models and combines them into a single model. Does not take image model as input
    def __init__(self, n, base_model):
        self.models = []
        self.fit_histories = []
        self.n = n
        self.base_model = base_model
    
    def fit(self, train_images, train_features, train_y, n):
        #Step 1: Train N models
        splits = list(zip(np.array_split(train_images, n), np.array_split(train_y, n)))
        for i in range(n):
            # Train the model on n-1 splits
            train_images_i = np.concatenate(
                [split[0] for j, split in enumerate(splits) if j != i]
            )
            train_y_i = np.concatenate([split[1] for j, split in enumerate(splits) if j != i])
            valid_images_i = np.concatenate(
                [split[0] for j, split in enumerate(splits) if j == i])
            y_valid_i = np.concatenate([split[1] for j, split in enumerate(splits) if j == i])

            image_model, fit_history = CNN_model(
                self.base_model, True, train_images_i, train_y_i, valid_images_i, y_valid_i)
            
            self.models.append(image_model)
            self.fit_histories.append(fit_history)
        #Step 2: Combine the models into a single model.
        img_pred = np.mean([model.predict(train_images) for model in self.models])
        input = np.column_stack((img_pred, train_features))
        self.model = RandomForestRegressor(n_estimators=100, max_depth=10)
        self.model.fit(input, train_y)

    def predict(self, test_images, test_features):
        img_pred = np.mean([model.predict(test_images) for model in self.models])
        test_input = np.column_stack((img_pred, test_features))
        return self.model.predict(test_input)
    
def N_CNN_RF_model(
    n,
    base_model,
    train_images,
    train_features,
    train_y,
):
    N_CNN_RF_ = N_CNN_RF(n, base_model)
    N_CNN_RF_.fit(train_images, train_features, train_y, n)
    return N_CNN_RF_



def CNN_MLP_model(
    pretrained_model,
    train_images,
    train_features,
    train_y,
    validation_images,
    validation_features,
    validation_y,
):
    # Create the model with the pretrained model and a new classification layer
    try:
        train_features = train_features.drop(columns=["image_predictions"])
    except:
        pass
    try:
        validation_features = validation_features.drop(columns=["image_predictions"])
    except:
        pass

    # Vision Part
    target_width = train_images[0].shape[0]
    target_height = train_images[0].shape[1]
    base_model = pretrained_model(
        weights="imagenet",
        include_top=False,
        input_shape=(target_width, target_height, 3),
    )
    for layer in base_model.layers:
        layer.trainable = False

    image_input = Input(shape=(target_width, target_height, 3))
    image_features = base_model(image_input)
    # image_features = Flatten()(image_features)
    # image_features = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.02))(image_features)
    # image_features = layers.Dropout(0.2)(image_features)
    # #image_features = Dense(256, activation='relu', kernel_regularizer=regularizers.l1(0.01))(image_features)
    # image_features = Dense(128, activation='relu', kernel_regularizer=regularizers.l1(0.02))(image_features)
    # image_features = Dense(32, activation='relu')(image_features)
    flattened_image_features = Flatten()(image_features)

    # Features Part
    num_features = len(train_features.iloc[0])
    features_input = Input(shape=(num_features,))
    numeric_features = Dense(
        num_features, activation="relu", kernel_regularizer=regularizers.l2(0.05)
    )(features_input)
    numeric_features = Dense(
        num_features, activation="relu", kernel_regularizer=regularizers.l1(0.04)
    )(numeric_features)
    numeric_features = Dense(
        num_features, activation="relu", kernel_regularizer=regularizers.l1(0.03)
    )(numeric_features)
    numeric_features = Dense(num_features, activation="relu")(numeric_features)
    numeric_features = Dense(num_features, activation="relu")(numeric_features)
    numeric_features = Flatten()(numeric_features)

    # Combined
    combined_features = Concatenate()([numeric_features, flattened_image_features])
    combined_features = Dense(
        48, activation="relu", kernel_regularizer=regularizers.l2(0.08)
    )(combined_features)
    image_features = layers.Dropout(0.1)(image_features)
    combined_features = Dense(
        12, activation="relu", kernel_regularizer=regularizers.l2(0.04)
    )(combined_features)
    combined_features = Dense(
        4, activation="relu", kernel_regularizer=regularizers.l2(0.04)
    )(combined_features)
    output_layer = Dense(1, activation="linear", name="output")(combined_features)

    # Create the model
    combined_model = Model(inputs=[image_input, features_input], outputs=output_layer)
    combined_model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="mean_absolute_error",
        metrics=["mean_absolute_error"],
    )

    # Fit the model
    fit_history = combined_model.fit(
        [train_images, train_features],
        train_y,
        epochs=250,
        validation_data=([validation_images, validation_features], validation_y),
        callbacks=[EarlyStopping(patience=15, restore_best_weights=True)],
    )
    return combined_model, fit_history

def CNN_RF_model_V2(
    CNN_model,
    RF_model,
    train_images,
    train_features,
    train_prices,
    test_images,
    test_features,
    test_prices,
):
    # If image_predictions exists in either train_features or test_features, drop them
    try:
        train_features = train_features.drop(columns=["image_predictions"])
    except:
        pass
    try:
        train_features = train_features.drop(columns=["reconstruction_error"])
    except:
        pass
    try:
        test_features = test_features.drop(columns=["image_predictions"])
    except:
        pass
    try:
        test_features = test_features.drop(columns=["reconstruction_error"])
    except:
        pass

    # Get the image predictions
    CNN_train = CNN_model.predict(train_images).flatten()
    RF_train = RF_model.predict(train_features)

    CNN_test = CNN_model.predict(test_images).flatten()
    RF_test = RF_model.predict(test_features)

    # Concatenate the predictions
    train_input = np.column_stack((CNN_train, RF_train))
    test_input = np.column_stack((CNN_test, RF_test))
    # Train the model

    train_input_without_img_pred = RF_train

    RF_model = RF(train_input, train_prices, test_input, test_prices)
    return RF_model



























################# LEGACY / Not in USE ###################
def CNN_confidence_model(
    CNN_model, train_images, validation_images, test_images, y_train, y_valid, y_test
):
    # Get the predictions
    train_predictions = CNN_model.predict(train_images).flatten()
    validation_predictions = CNN_model.predict(validation_images).flatten()
    test_predictions = CNN_model.predict(test_images).flatten()

    # Calculate the residuals
    train_abs_differnce = np.abs(train_predictions - y_train)
    validation_abs_differnce = np.abs(validation_predictions - y_valid)
    test_abs_differnce = np.abs(test_predictions - y_test)

    # Calculate absolute residuals in percentage
    train_residuals_percent_error = train_abs_differnce / y_train
    validation_residuals_percent_error = validation_abs_differnce / y_valid
    test_residual_percent_error = test_abs_differnce / y_test

    # Set labels 0 = above 20% error, 1 = below 20% error
    train_confidence_labels = np.where(train_residuals_percent_error > 0.2, 0, 1)
    validation_confidence_labels = np.where(
        validation_residuals_percent_error > 0.2, 0, 1
    )
    test_confidence_labels = np.where(test_residual_percent_error > 0.2, 0, 1)

    print(np.round(train_confidence_labels, 2))
    print(validation_confidence_labels)
    print(test_confidence_labels)
    # Train a new model on the iamges and confidence_labels
    confidence_model = Sequential()
    confidence_model.add(Flatten(input_shape=(train_images.shape[1:])))
    confidence_model.add(Dense(256, activation="relu"))
    confidence_model.add(Dropout(0.2))
    confidence_model.add(Dense(128, activation="relu"))
    confidence_model.add(Dense(64, activation="relu"))
    confidence_model.add(Dense(1, activation="sigmoid"))

    confidence_model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    confidence_model.fit(
        train_images,
        train_confidence_labels,
        epochs=10,
        validation_data=(validation_images, validation_confidence_labels),
    )

    # Test the model
    test_score = confidence_model.evaluate(test_images, test_confidence_labels)

    return confidence_model


def CNN_model_size(pretrained_model, train_images, y_train, validation_images, y_valid):
    # Load the Pretrained Model
    target_width = train_images[0].shape[0]
    target_height = train_images[0].shape[1]
    input_shape = (target_width, target_height, 3)
    base_model = pretrained_model(
        weights="imagenet", include_top=False, input_shape=input_shape
    )

    # Freeze the pretrained weights
    for layer in base_model.layers:
        layer.trainable = False

    # Create the model with the pretrained model and a new classification layer
    model = Sequential(
        [
            base_model,
            Flatten(),
            Dense(512, activation="relu", kernel_regularizer=regularizers.l2(0.001)),
            layers.Dropout(0.1),
            Dense(256, activation="relu", kernel_regularizer=regularizers.l1(0.001)),
            layers.Dropout(0.1),
            Dense(128, activation="relu"),
            Dense(64, activation="relu"),
            Dense(1, activation="linear"),
        ]
    )

    # Check which type of model we are building
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="mean_absolute_error",
        metrics=["mean_squared_error", "mean_absolute_error"],
    )

    # Fit the model
    fit_history = model.fit(
        train_images,
        y_train,
        epochs=100,
        validation_data=(validation_images, y_valid),
        callbacks=[EarlyStopping(patience=4, restore_best_weights=True)],
    )
    return model, fit_history


def CNN_RF_Weighted_modelV2(
    image_model,
    train_images,
    train_features,
    train_y,
    test_images,
    test_features,
    test_y,
    error_column,
    threshhold=0.2,
):
    # Step 1: Split the train_images, train_features and train_y, test_y into two groups
    # Group 1: train/test_features['error_column'] <= threshhold
    # Group 2: train/test_features['error_column'] > threshhold

    train_features_group1 = []
    train_features_group2 = []
    train_y_group1 = []
    train_y_group2 = []

    test_features_group1 = []
    test_features_group2 = []
    test_y_group1 = []
    test_y_group2 = []

    # img_train_predictions = image_model.predict(train_images)
    # img_test_predictions = image_model.predict(test_images)

    for i in range(len(train_features)):
        if train_features.iloc[i][error_column] <= threshhold:
            train_features_group1.append((train_features.iloc[i]))
            train_y_group1.append(train_y.iloc[i])
        else:
            train_features_group2.append((train_features.iloc[i]))
            train_y_group2.append(train_y.iloc[i])

    for i in range(len(test_features)):
        if test_features.iloc[i][error_column] <= threshhold:
            test_features_group1.append((test_features.iloc[i]))
            test_y_group1.append(test_y.iloc[i])
        else:
            test_features_group2.append((test_features.iloc[i]))
            test_y_group2.append(test_y.iloc[i])

    # turn them into dataframes
    train_features_group1 = pd.DataFrame(train_features_group1)
    test_features_group1 = pd.DataFrame(test_features_group1)

    train_features_group2 = pd.DataFrame(train_features_group2)
    test_features_group2 = pd.DataFrame(test_features_group2)
    train_features_group2 = train_features_group2.drop(columns=["image_predictions"])
    test_features_group2 = test_features_group2.drop(columns=["image_predictions"])

    # Step 2: Train two RF models on the two groups
    RF_model1 = RF(
        train_features_group1, train_y_group1, test_features_group1, test_y_group1
    )
    RF_model2 = RF(
        train_features_group2, train_y_group2, test_features_group2, test_y_group2
    )

    # Step 3: Combine the two models into a single model

    return None


def CNN_AE_RF_MODEL_V2(
    image_model,
    train_images,
    train_features,
    train_y,
    test_images,
    test_features,
    test_y,
    autoEncoder_,
):

    # autoEncoder_ = autoEncoder(train_images)
    train_features["image_prediction"] = image_model.predict(train_images)
    test_features["image_prediction"] = image_model.predict(test_images)

    # Step 2: Calculate the reconstruction error for each training_example
    train_features["reconstruction_error"] = autoEncoder_.calculate_error(train_images)
    test_features["reconstruction_error"] = autoEncoder_.calculate_error(test_images)

    # Step 4: Set the optimal threshhold as the 1st quartile of the reconstruction error
    threshold = np.quantile(train_features["reconstruction_error"], 0.60)
    print(threshold)
    # threshold = np.mean(train_features['reconstruction_error'])

    X2_train = train_features[train_features["reconstruction_error"] <= threshold]
    X2_test = test_features[test_features["reconstruction_error"] <= threshold]
    Y2_train = train_y[train_features["reconstruction_error"] <= threshold]
    Y2_test = test_y[test_features["reconstruction_error"] <= threshold]

    X1_train = train_features[train_features["reconstruction_error"] > threshold]
    X1_test = test_features[test_features["reconstruction_error"] > threshold]
    Y1_train = train_y[train_features["reconstruction_error"] > threshold]
    Y1_test = test_y[test_features["reconstruction_error"] > threshold]
    X1_train = X1_train.drop(columns=["image_prediction"])
    X1_test = X1_test.drop(columns=["image_prediction"])

    # Step 6: Train a RF model on X1, Y1
    print("Training RF model on X1, Y1")
    display(X1_train.head(1))
    display(X1_test.head(1))
    RF_model2 = RF(X1_train, Y1_train, X1_test, Y1_test)

    # Step 7: Train a CNN model on X2, Y2
    print("Training RF model on X2, Y2")
    display(X2_train.head(1))
    display(X2_test.head(1))
    RF_model2 = RF(X2_train, Y2_train, X2_test, Y2_test)

    # Remove image_predictions and reconstruction_error if exist
    try:
        train_features = train_features.drop(
            columns=["image_prediction", "reconstruction_error"]
        )
    except:
        pass
    try:
        test_features = test_features.drop(
            columns=["image_prediction", "reconstruction_error"]
        )
    except:
        pass
    # Test on the test_features, with the same principle
    return None


def CNN_RF_Size_model_V2(
    image_model,
    size_model,
    train_images,
    train_features,
    train_y,
    test_images,
    test_features,
    test_y,
):
    # Step 1: Use the image model to make predictions
    train_image_predictions = image_model.predict(train_images)
    test_image_predictions = image_model.predict(test_images)

    # Add predictions as a new column in features
    train_features["image_predictions"] = train_image_predictions
    test_features["image_predictions"] = test_image_predictions

    # Step 2: Calculate the size error percentage
    train_image_size_predictions = size_model.predict(train_images)
    test_image_size_predictions = size_model.predict(test_images)
    train_features["size_pred"] = train_image_size_predictions
    test_features["size_pred"] = test_image_size_predictions
    train_features["size_error_pct"] = (
        train_features["size_pred"] - train_features["size"]
    ) / train_features["size"]
    test_features["size_error_pct"] = (
        test_features["size_pred"] - test_features["size"]
    ) / test_features["size"]

    # Step 2: Calculate weights for the image_prediction based on column 'size_error_pct'
    # train_features['weights_based'] = 1 / (1 + train_features['size_error_pct'])
    # test_features['weights_based'] = 1 / (1 + test_features['size_error_pct'])
    # train_features['weights'] = np.where(abs(train_features['size_error_pct']) < 0.005, train_features['weights_based'], 0)
    # test_features['weights'] = np.where(abs(test_features['size_error_pct']) < 0.005, test_features['weights_based'], 0)

    # # Step 3: Weight the image prediction
    # train_features['weighted_image_predictions'] = train_features['weights'] * train_features['image_predictions']
    # test_features['weighted_image_predictions'] = test_features['weights'] * test_features['image_predictions']

    display(train_features)
    display(test_features)

    # #drop image_predictions and weights
    # train_features_ = train_features.drop(columns=['image_predictions', 'weights', 'weights_based', 'size_error_pct', 'size_pred'])
    # test_features_ = test_features.drop(columns=['image_predictions', 'weights', 'weights_based', 'size_error_pct', 'size_pred'])

    # Step 4: Train a RF on the combined image prediction and features
    RF_model = RF(train_features, train_y, test_features, test_y)
    # Drop size_pred and size_errror_pct
    train_features = train_features.drop(columns=["size_pred", "size_error_pct"])
    test_features = test_features.drop(columns=["size_pred", "size_error_pct"])
    return RF_model


#### Image Models (Unsupervised) ####
def BoW_K_means_model(train_images, k):
    train_bow = create_bow_representation(train_images, k)
    train_bow = [x.flatten() for x in train_bow]
    train_bow = np.array(train_bow)

    # Run K-means to group the data
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=100, random_state=42)
    kmeans.fit(train_bow)

    # Remake how predict works
    class kmeans_custom:
        def __init__(self, kmeans):
            self.kmeans = kmeans

        def predict(self, x):
            x = preprocess_images(x, "image_floorplan", 500, 500, True, False, True)
            x = create_bow_representation(x, k)
            x = [x.flatten() for x in x]
            x = np.array(x)
            return self.kmeans.predict(x)

    kmeans = kmeans_custom(kmeans)
    return kmeans


def SIFT(
    train_images,
    test_images,
    train_y,
    test_y,
    # train_features, test_features,
):
    # Initialize SURF object
    surf = cv2.SIFT_create()

    # Compute SURF features for train images
    train_descriptors = []
    for img in train_images:
        _, des = surf.detectAndCompute(img, None)
        if des is not None:
            train_descriptors.append(des)

    # Compute SURF features for test images
    test_descriptors = []
    for img in test_images:
        _, des = surf.detectAndCompute(img, None)
        if des is not None:
            test_descriptors.append(des)

    # Flatten the descriptors
    train_descriptors_flat = [item for sublist in train_descriptors for item in sublist]
    test_descriptors_flat = [item for sublist in test_descriptors for item in sublist]

    # Convert to numpy arrays
    train_descriptors_np = np.array(train_descriptors_flat)
    test_descriptors_np = np.array(test_descriptors_flat)
    print("Train Descriptors Shape: ", train_descriptors_np.shape)
    print("Test Descriptors Shape: ", test_descriptors_np.shape)

    # Train a Random Forest Regressor
    rf = RF(train_descriptors_np, train_y, test_descriptors_np, test_y)
    return None


def find_optimal_threshhold(prices, reconstruction_errors):
    # Find the optimal threshold
    thresholds = np.linspace(0, 0.1, 100)
    best_threshold = 0
    best_mae = 1000000
    for threshold in thresholds:
        predictions = np.where(reconstruction_errors > threshold, 1, 0)
        mae = np.mean(np.abs(prices - predictions))
        if mae < best_mae:
            best_mae = mae
            best_threshold = threshold
    return best_threshold
