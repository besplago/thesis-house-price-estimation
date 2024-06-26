import os
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
from sklearn.preprocessing import StandardScaler

from skimage.metrics import structural_similarity as ssim

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras import regularizers

import keras
from keras import losses, Model

from xgboost import XGBRegressor
from vit_keras import vit, utils

#from img_utils import preprocess_images, create_bow_representation, set_gpu, data_to_df, set_cpu
from utils import *
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


RF_SEED: int | None = None
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 25


#### Feature Models ####
def Lasso_(x_train, y_train):
    from sklearn.linear_model import Lasso

    model = Lasso(alpha=0.1)
    model.fit(x_train, y_train)
    return model


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

    # GridSearch
    gridSearch = False
    if gridSearch:
        param_grid = {
            "n_estimators": [100, 200, 300, 400],  # Number of trees in the forest
            "max_depth": [5, 10, 15, 20, 40],  # Maximum depth of individual trees
            "min_samples_split": [
                2,
                4,
                8,
                16,
            ],  # Minimum samples required to split a node
        }
        model = GridSearchCV(
            RandomForestRegressor(random_state=RF_SEED),
            param_grid,
            cv=5,
            n_jobs=-1,
            verbose=2,
        )
        # get the model with best params
        model.fit(x_train, y_train)
        print("Best Params from GridSearch: ", model.best_params_)
        model = model.best_estimator_

    else:
        model = RandomForestRegressor(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            random_state=RF_SEED,
            n_jobs=-1,
        )
        model.fit(x_train, y_train)
    return model


def SVC(x_train, y_train, x_test, y_test):
    from sklearn.svm import SVC

    model = SVC(kernel="sigmoid", C=1, gamma="auto")
    model.fit(x_train, y_train)
    #y_pred = model.predict(x_test)
    #print("Results")
    #eval_model(y_test, y_pred)
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
    #eval_model(model, x_test, y_test)
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
    print("num_features", num_features)
    model.add(Dense(num_features, input_dim=num_features, activation="relu"))
    model.add(Dense(num_features, activation="relu"))
    model.add(Dense(num_features, activation="relu"))
    model.add(Dense(num_features, activation="relu"))
    model.add(Dense(num_features, activation="relu"))
    model.add(Dense(num_features, activation="relu"))
    model.add(Dense(num_features, activation="relu"))
    model.add(Dense(num_features, activation="relu"))
    model.add(Dense(1, activation="linear"))
    # Compiling and fitting
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="mean_absolute_error",
        metrics=["mean_absolute_error"],
    )

    fit_history = model.fit(
        x_train,
        y_train,
        epochs=300,
        batch_size=64,
        validation_data=(x_test, y_test),
        callbacks=EarlyStopping(
            monitor="val_loss", patience=30, restore_best_weights=True
        ),
        verbose=1,
    )
    #print("Test score: ", model.evaluate(x_test, y_test))
    #y_pred = model.predict(x_test).flatten()
    #eval_model(y_test, y_pred)
    return model, fit_history


#### Image Models ####
def CNN_model(
    # pretrained_model, custom_layers, train_images, y_train, validation_images, y_valid
    pretrained_model: object,
    train_images: np.array,
    y_train: np.array,
    validation_images: np.array,
    y_valid: np.array,
    custom_layers: list = [],
):
    # Load the Pretrained Model
    # target_width = train_images[0].shape[0]
    # target_height = train_images[0].shape[1]
    # input_shape = (target_width, target_height, 3)
    input_shape: tuple = train_images[0].shape
    base_model = pretrained_model(
        weights="imagenet", include_top=False, input_shape=input_shape
    )

    # Freeze the pretrained weights
    for layer in base_model.layers:
        layer.trainable = False

    # Create the model with the pretrained model and a new classification layer
    if custom_layers:
        model = Sequential([base_model] + custom_layers)
    else:
        model = Sequential([base_model, Flatten(), Dense(1, activation="linear")])

    # Check which type of model we are building
    model.compile(optimizer=Adam(learning_rate=1), loss="mean_absolute_error")

    # Fit the model
    fit_history = model.fit(
        train_images,
        y_train,
        epochs=100,
        validation_data=(validation_images, y_valid),
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True)
        ],
    )
    return model, fit_history


def CNN_model1(
    # pretrained_model, custom_layers, train_images, y_train, validation_images, y_valid
    pretrained_model: object,
    train_images: np.array,
    y_train: np.array,
    validation_images: np.array,
    y_valid: np.array,
    custom_layers: list = [],
):
    # Load the Pretrained Model
    # target_width = train_images[0].shape[0]
    # target_height = train_images[0].shape[1]
    # input_shape = (target_width, target_height, 3)
    input_shape: tuple = train_images[0].shape
    base_model = pretrained_model(
        weights="imagenet", include_top=False, input_shape=input_shape
    )

    # Freeze the pretrained weights
    for layer in base_model.layers:
        layer.trainable = False

    # Create the model with the pretrained model and a new classification layer
    if custom_layers:
        model = Sequential([base_model] + custom_layers)
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
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True)
        ],
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


#### Ensemble Models ####
class CNN_RF:
    def __init__(self, image_model):
        self.image_model = image_model

    # Fit that needs images, features. Predict images, features
    def fit(self, train_images, train_features, train_y):
        # Get the image predictions
        train_image_predictions = self.image_model.predict(train_images).flatten()

        # Concatenate the predictions
        train_input = np.column_stack((train_image_predictions, train_features))
        train_input = pd.DataFrame(train_input)
        train_input.columns = ["image_predictions"] + list(train_features.columns)

        # Train the model
        # self.model = RandomForestRegressor(n_estimators=RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH, random_state=RF_SEED)
        self.model = RF(train_input, train_y)
        self.model.fit(train_input, train_y)
        self.feature_importances_ = self.model.fit(
            train_input, train_y
        ).feature_importances_
        self.feature_importances_ = dict(
            zip(train_input.columns, self.feature_importances_)
        )

    def predict(self, test_images, test_features):
        # Get the image predictions
        test_image_predictions = self.image_model.predict(test_images).flatten()

        # Concatenate the predictions
        test_input = np.column_stack((test_image_predictions, test_features))

        # Predict the prices
        return self.model.predict(test_input)


def CNN_RF_model(
    image_model,
    train_images: np.array,
    train_features: pd.DataFrame,
    train_y: np.array,
):
    image_model = (
        keras.models.load_model(image_model)
        if isinstance(image_model, str)
        else image_model
    )
    CNN_RF_ = CNN_RF(image_model)
    CNN_RF_.fit(train_images, train_features, train_y)
    return CNN_RF_


"""
An implementation of a convolutional autoencoder (CAE) using Keras.

Jason M. Manley, 2018
jmanley@rockefeller.edu
"""


class ConvAutoEncoder:

    def __init__(
        self,
        input_shape,
        output_dim,
        filters=[32, 64, 128, 256],
        kernel=(3, 3),
        stride=(1, 1),
        strideundo=2,
        pool=(2, 2),
        optimizer="adamax",
        lossfn="mse",
    ):
        # For now, assuming input_shape is mxnxc, and m,n are multiples of 2.

        self.input_shape = input_shape
        self.output_dim = output_dim

        # define encoder architecture
        self.encoder = keras.models.Sequential()
        self.encoder.add(keras.layers.InputLayer(input_shape))
        for i in range(len(filters)):
            self.encoder.add(
                keras.layers.Conv2D(
                    filters=filters[i],
                    kernel_size=kernel,
                    strides=stride,
                    activation="elu",
                    padding="same",
                )
            )
            self.encoder.add(keras.layers.MaxPooling2D(pool_size=pool))
        self.encoder.add(keras.layers.Flatten())
        self.encoder.add(keras.layers.Dense(output_dim))

        # define decoder architecture
        self.decoder = keras.models.Sequential()
        self.decoder.add(keras.layers.InputLayer((output_dim,)))
        self.decoder.add(
            keras.layers.Dense(
                filters[len(filters) - 1]
                * int(input_shape[0] / (2 ** (len(filters))))
                * int(input_shape[1] / (2 ** (len(filters))))
            )
        )
        self.decoder.add(
            keras.layers.Reshape(
                (
                    int(input_shape[0] / (2 ** (len(filters)))),
                    int(input_shape[1] / (2 ** (len(filters)))),
                    filters[len(filters) - 1],
                )
            )
        )
        for i in range(1, len(filters)):
            self.decoder.add(
                keras.layers.Conv2DTranspose(
                    filters=filters[len(filters) - i],
                    kernel_size=kernel,
                    strides=strideundo,
                    activation="elu",
                    padding="same",
                )
            )
        self.decoder.add(
            keras.layers.Conv2DTranspose(
                filters=input_shape[2],
                kernel_size=kernel,
                strides=strideundo,
                activation=None,
                padding="same",
            )
        )

        # compile model
        input = keras.layers.Input(input_shape)
        code = self.encoder(input)
        reconstructed = self.decoder(code)

        self.ae = keras.models.Model(inputs=input, outputs=reconstructed)
        self.ae.compile(optimizer=optimizer, loss=lossfn)

    def fit(self, x, epochs=30, callbacks=[keras.callbacks.BaseLogger()], **kwargs):

        # self.ae.fit(x=x, y=x, epochs=epochs, callbacks=callbacks, **kwargs)
        self.ae.fit(x=x, y=x, epochs=epochs, callbacks=None, **kwargs)

    def save_weights(self, path=None, prefix=""):
        if path is None:
            path = os.getcwd()
        # create path if not existing
        if not os.path.exists(path):
            os.makedirs(path)

        self.encoder.save_weights(os.path.join(path, prefix + "encoder_weights.h5"))
        self.decoder.save_weights(os.path.join(path, prefix + "decoder_weights.h5"))

    def load_weights(self, path=None, prefix=""):
        if path is None:
            path = os.getcwd()
        self.encoder.load_weights(os.path.join(path, prefix + "encoder_weights.h5"))
        self.decoder.load_weights(os.path.join(path, prefix + "decoder_weights.h5"))

    def encode(self, input):
        return self.encoder.predict(input)

    def decode(self, codes):
        return self.decoder.predict(codes)

    def predict(self, input):
        return self.encode(input)

    def save_model(self):
        self.ae.save("cae_model.h5")

    def calculate_error(self, input):
        # encode, decode, and calculate error for images return array of reconstruction errors
        encoded = self.encode(input)
        decoded = self.decode(encoded)
        error = np.mean(np.square(input - decoded), axis=(1, 2, 3))
        # reconstruction_error = tf.reduce_mean(tf.square(input-decoded))
        return error

    def calculate_ssim(self, input):
        # encode, decode, and calculate error for images return array of reconstruction errors
        encoded = self.encode(input)
        decoded = self.decode(encoded)
        norm_input = input / 225.0
        norm_decoded = decoded / 225.0

        def calc_ssim(img1, img2):
            from skimage.metrics import structural_similarity as ssim

            return ssim(img1, img2, channel_axis=2, data_range=1)

        return [calc_ssim(img1, img2) for img1, img2 in zip(norm_input, norm_decoded)]


def autoEncoder(train_images, latent_dim):
    autoencoder = ConvAutoEncoder(
        input_shape=train_images.shape[1:], output_dim=latent_dim
    )
    autoencoder.fit(train_images)
    return autoencoder


class CNN_AE_RF:
    def __init__(self, image_model, AE_):
        self.image_model = image_model
        self.autoEncoder_ = None if AE_ is None else AE_

    def fit(self, train_images, train_features, train_y):
        # Calculate the reconstruction error
        if self.autoEncoder_ is None:
            self.autoEncoder_ = autoEncoder(train_images, latent_dim=64)

        reconstruction_error = self.calculate_ssim(train_images)
        # reconstruction_error = self.autoEncoder_.calculate_ssim(train_images)

        # Get the image predictions
        train_image_predictions = self.image_model.predict(train_images).flatten()

        # Combine the predictions
        train_input = np.column_stack(
            (train_image_predictions, reconstruction_error, train_features)
        )
        train_input = pd.DataFrame(train_input)
        train_input.columns = ["image_predictions", "reconstruction_error"] + list(
            train_features.columns
        )

        # Run RF
        self.model = RandomForestRegressor(
            n_estimators=RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH, random_state=RF_SEED
        )
        self.model.fit(train_input, train_y)
        self.feature_importances_ = self.model.fit(
            train_input, train_y
        ).feature_importances_
        self.feature_importances_ = dict(
            zip(train_input.columns, self.feature_importances_)
        )

    # def calculate_error(self, image):
    #     encoded_img = encoded_img = self.autoEncoder_.encoder(image)
    #     decoded_img = self.autoEncoder_.decoder(encoded_img)
    #     reconstruction_error = tf.reduce_mean(tf.sqaure(input-decoded_img))

    #     #return self.autoEncoder_.calculate_error(image)

    # def get_reconstruction(self, image):
    #     encoded_img = self.autoEncoder_.encoder(image)
    #     decoded_img = self.autoEncoder_.decoder(encoded_img)
    #     diff = np.mean(np.square(image - decoded_img), axis=(1,2,3))
    #     return decoded_img, diff

    def calculate_ssim(self, input):
        # encode, decode, and calculate error for images return array of reconstruction errors
        encoded = self.autoEncoder_.encode(input)
        decoded = self.autoEncoder_.decode(encoded)
        norm_input = input / 225.0
        norm_decoded = decoded / 225.0

        def calc_ssim(img1, img2):
            return ssim(img1, img2, channel_axis=2, data_range=1)

        return [calc_ssim(img1, img2) for img1, img2 in zip(norm_input, norm_decoded)]

    def predict(self, test_images, test_features):
        # reconstruction_error = self.autoEncoder_.calcuate_ssim(test_images)
        # reconstruction_error = self.autoEncoder_.calculate_error(test_images)
        reconstruction_error = self.calculate_ssim(test_images)
        test_image_predictions = self.image_model.predict(test_images).flatten()
        test_input = np.column_stack(
            (test_image_predictions, reconstruction_error, test_features)
        )
        return self.model.predict(test_input)


def CNN_AE_RF_model(
    image_model,
    AE_: object,
    train_images: np.array,
    train_features: pd.DataFrame,
    train_y: np.array,
):
    image_model = (
        keras.models.load_model(image_model)
        if isinstance(image_model, str)
        else image_model
    )
    CNN_AE_RF_model = CNN_AE_RF(image_model, AE_)
    CNN_AE_RF_model.fit(train_images, train_features, train_y)
    return CNN_AE_RF_model


################# LEGACY / Not in USE ###################
class N_CNN_RF:
    # Setup a class that train N-models and combines them into a single model. Does not take image model as input
    def __init__(self, n, base_model):
        self.models = []
        self.fit_histories = []
        self.n = n
        self.base_model = base_model

    def fit(self, train_images, train_features, train_y, n):
        # Step 1: Train N models
        splits = list(zip(np.array_split(train_images, n), np.array_split(train_y, n)))
        for i in range(n):
            # Train the model on n-1 splits
            train_images_i = np.concatenate(
                [split[0] for j, split in enumerate(splits) if j != i]
            )
            train_y_i = np.concatenate(
                [split[1] for j, split in enumerate(splits) if j != i]
            )
            valid_images_i = np.concatenate(
                [split[0] for j, split in enumerate(splits) if j == i]
            )
            y_valid_i = np.concatenate(
                [split[1] for j, split in enumerate(splits) if j == i]
            )

            image_model, fit_history = CNN_model(
                self.base_model,
                True,
                train_images_i,
                train_y_i,
                valid_images_i,
                y_valid_i,
            )

            self.models.append(image_model)
            self.fit_histories.append(fit_history)

        img_preds = [model.predict(train_images) for model in self.models]
        img_pred = np.mean(img_preds, axis=0)
        img_pred = img_pred.flatten()
        train_input = np.column_stack((img_pred, train_features))
        train_input = pd.DataFrame(train_input)
        train_input.columns = ["image_predictions"] + list(train_features.columns)
        self.model = RandomForestRegressor(
            n_estimators=RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH, random_state=RF_SEED
        )
        self.model.fit(train_input, train_y)
        self.feature_importances_ = self.model.fit(
            train_input, train_y
        ).feature_importances_
        self.feature_importances_ = dict(
            zip(train_input.columns, self.feature_importances_)
        )

    def predict(self, test_images, test_features):
        img_preds = [model.predict(test_images) for model in self.models]
        img_preds = np.mean(img_preds, axis=0)
        img_preds = img_preds.flatten()
        test_input = np.column_stack((img_preds, test_features))
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


class AE_OLD(Model):
    def __init__(self):
        super(AE_OLD, self).__init__()

        self.encoder = tf.keras.Sequential(
            [
                layers.Input(shape=(224, 224, 3)),
                layers.Conv2D(32, (3, 3), activation="relu", padding="same", strides=2),
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
                layers.Conv2DTranspose(
                    32, (3, 3), activation="relu", padding="same", strides=2
                ),
                layers.Conv2D(
                    1, kernel_size=(3, 3), activation="sigmoid", padding="same"
                ),
            ]
        )

    def call(self, x):
        # grayscale x
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def calculate_error(self, input_images):
        # graysacle images
        reconstruction_errors = []
        for image in input_images:
            # Expand the single image to a batch dimension
            image_batch = tf.expand_dims(image, axis=0)
            decoded = self.__call__(image_batch)  # Call the model with the batch
            error = tf.keras.losses.MeanSquaredError()(image_batch, decoded)
            reconstruction_errors.append(tf.reduce_mean(error))
        reconstruction_errors = np.array(reconstruction_errors)
        return reconstruction_errors

    def reconstruct_img(self, img):
        img = tf.expand_dims(img, axis=0)
        return self.__call__(img)


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
    image_model_path: str,
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

    # image_model = keras.models.load_model(image_model_path)
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
    image_model_path: str,
    train_images,
    train_features,
    train_y,
    test_images,
    test_features,
    test_y,
    autoEncoder_,
):
    image_model = keras.models.load_model(image_model_path)

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
    image_model_path,
    size_model,
    train_images,
    train_features,
    train_y,
    test_images,
    test_features,
    test_y,
):
    image_model = keras.models.load_model(image_model_path)

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
