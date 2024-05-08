###### MODEL EVALUATION CLASSIFICATION ######
def label_score(predicted_labels, actual_labels):
    """
    Returns the accuracy, precision, recall and f1-score of the model
    """
    accuracy = accuracy_score(actual_labels, predicted_labels)
    precision = precision_score(actual_labels, predicted_labels, average="weighted")
    recall = recall_score(actual_labels, predicted_labels, average="weighted")
    f1 = f1_score(actual_labels, predicted_labels, average="weighted")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    cm = confusion_matrix(predicted_labels, actual_labels)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
    return None


def label_score_softmax(softmax_labels, actual_labels):
    """
    Evaluate the softmax predictions, based on how far from actual they are
    Actual labels: Onehot-encoded
    """
    # Get the predicted labels
    predicted_labels = np.argmax(softmax_labels, axis=1)
    # Get the actual labels
    actual_labels = np.argmax(actual_labels, axis=1)
    # Get the accuracy
    accuracy = accuracy_score(actual_labels, predicted_labels)
    # Get the precision
    precision = precision_score(actual_labels, predicted_labels, average="weighted")
    # Get the recall
    recall = recall_score(actual_labels, predicted_labels, average="weighted")
    # Get the f1-score
    f1 = f1_score(actual_labels, predicted_labels, average="weighted")

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"f1: {f1:.2f}")

    cm = confusion_matrix(predicted_labels, actual_labels)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
    return None


def label_score(predicted_labels, actual_labels):
    """
    Returns the accuracy, precision, recall and f1-score of the model
    """
    accuracy = accuracy_score(actual_labels, predicted_labels)
    precision = precision_score(actual_labels, predicted_labels, average="weighted")
    recall = recall_score(actual_labels, predicted_labels, average="weighted")
    f1 = f1_score(actual_labels, predicted_labels, average="weighted")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    cm = confusion_matrix(predicted_labels, actual_labels)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
    return None


# SIFT-Features
def get_sift_features(image):
    # Convert to image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(image, None)
    return kp, des


def create_bow_representation(images, k=25):
    """
    Creates a Bag-of-Visual-Words representation for a set of images.
    """


def display_sift_features(image, kp):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print(image.shape)
    img = cv2.drawKeypoints(image, kp, None)
    plt.imshow(img)
    plt.show()


def create_bow_representation(images, k=100):
    """
      Creates a Bag-of-Visual-Words representation for a set of images.

    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(all_descriptors)
      Args:
          images: A list of numpy arrays representing grayscale images.
          k: Number of visual words (clusters) to use.

      Returns:
          A list of histograms representing the frequency of each visual word for each image.
    """

    all_descriptors = []
    sift = cv2.SIFT_create()
    for image in images:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, descriptors = sift.detectAndCompute(image, None)
        all_descriptors.extend(descriptors)

    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(all_descriptors)

    bow_histograms = []
    for image in images:
        _, descriptors = sift.detectAndCompute(image, None)
        hist = np.zeros(k)
        for descriptor in descriptors:
            cluster_id = kmeans.predict([descriptor])[0]
            hist[cluster_id] += 1
        hist /= len(descriptors)  # Normalize histogram
        bow_histograms.append(hist)

    return bow_histograms


# Room Classification
def classify_room(pipe, image):
    image_PIL = Image.fromarray(image)
    room_predictions = pipe.predict([image_PIL])
    # print("Room Predcitions", room_predictions)
    best_pred = room_predictions[0][0]
    # print("Best Prediction", best_pred)
    score = best_pred["score"]
    label = best_pred["label"]
    # print("Score:",score, "Label", label)
    return label, score


def find_rooms(images, pipe, room, threshold):
    # Loop through images, and find the room. Return the Room with the highest score
    # or empty None if no room is found

    advanced = False
    if advanced:
        room_images = []
        for image in images:
            label, score = classify_room(pipe, image)
            if label == room and score > threshold:
                room_images.append((image, score))
        if len(room_images) > 0:
            room_images.sort(key=lambda x: x[1], reverse=True)
            return room_images[0][0]
        else:
            return np.zeros((100, 100, 3), dtype=np.uint8)

    else:
        # find room with score above threshold, return the first one
        for image in images:
            label, score = classify_room(pipe, image)
            print(label, room)
            print(label == room)
            print(score, threshold)
            if label == room and score > threshold:
                print("YEEESSSIIIR")
                return image
        return np.zeros((100, 100, 3), dtype=np.uint8)


def find_rooms_create_image(images, pipe, threshold):
    # Loop through images and find "Kitchen", "Living Room", "Bedroom", "Bathroom".
    # Create a 2x2 image with the four rooms.
    # If some rooms are not found, fill the spot with a black image.
    # ONly accept the image if "score" is above 0.6
    # Stop when we have found all four rooms. Or if we have looped through all images.
    # Make it a dict with the room as key and the image as value
    rooms = ["Kitchen", "Living Room", "Bedroom", "Bathroom"]
    found_rooms = {}
    room_images = {}
    room_scores = {}
    for image in images:
        label, score = classify_room(pipe, image)
        if score > threshold and label in rooms:
            if label not in room_scores:
                room_scores[label] = score
                room_images[label] = image
                print("First image of", label, "   Score", score)
                plt.imshow(image)
                plt.show()
            elif score > room_scores[label]:
                room_scores[label] = score
                room_images[label] = image
                print("Updating image of", label, "   Score", score)
                plt.imshow(image)
                plt.show()
            found_rooms[label] = True

    for room in rooms:
        if room not in room_images:
            found_rooms[room] = False
            room_images[room] = np.zeros((100, 100, 3), dtype=np.uint8)
            found_all_rooms = False
    return (
        room_images["Kitchen"],
        room_images["Living Room"],
        room_images["Bedroom"],
        room_images["Bathroom"],
    )


# Room Classification
def classify_room(pipe, image):
    room = pipe.predict([image])
    return room


##### LABEL ENCODING ######
def prices_to_n_labels(
    all_prices: np.array, prices: np.array, n_labels: int
) -> np.array:
    """
    Convert the prices to n_labels

    Args:
      all_prices(`np.array`): All the prices
      prices(`np.array`): The prices we want to convert
      n_labels(`int`): The number of labels we want to convert the prices to

    Returns:
      `np.array`: The prices converted to n_labels
    """

    # Calculate the quantiles
    quantiles = [np.quantile(all_prices, i / n_labels) for i in range(1, n_labels)]
    print(quantiles)

    labels = [
        (
            0
            if price < quantiles[0]
            else (
                n_labels - 1
                if price > quantiles[-1]
                else np.argmax([price < quantile for quantile in quantiles])
            )
        )
        for price in prices
    ]
    encoder = OneHotEncoder(sparse_output=False)
    one_hot_labels = encoder.fit_transform(np.array(labels).reshape(-1, 1))
    return one_hot_labels


def price_categories(all_prices, prices) -> np.array:
    # Calculate quantiles
    low_quantile = np.quantile(all_prices, 0.33)
    # print(low_quantile)
    high_quantile = np.quantile(all_prices, 0.66)
    # print(high_quantile)
    labels = [
        0 if price < low_quantile else 1 if price < high_quantile else 2
        for price in prices
    ]
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


def normal_distribution_label(data: pd.DataFrame, num_labels: int) -> pd.DataFrame:
    """
    Add labels-column to the data-points, based on prices
    Creates labels based on a normal distribution around the data.
    That is, we have more labels the closer we are to the mean price, and less the further away we are.
    Return a data-Frame with the labels and the label codes.
    """
    # We want to predict the price of the house
    min = data["price"].min()
    first_quan = data["price"].quantile(0.25)
    mean = data["price"].mean()
    third_quan = data["price"].quantile(0.75)
    max = data["price"].max()
    # Create a normal distribution of the labels
    f1 = np.linspace(0, min, round(num_labels * 0.023))
    f2 = np.linspace(min, first_quan, round(num_labels * 0.14))
    f3 = np.linspace(first_quan, mean, round(num_labels * 0.34))
    f4 = np.linspace(mean, third_quan, round(num_labels * 0.34))
    f5 = np.linspace(third_quan, max, round(num_labels * 0.14))
    f6 = np.linspace(max, max * 2, round(num_labels * 0.023))
    potential_labels = np.concatenate((f1, f2, f3, f4, f5, f6))

    # Create the label codes
    label_codes = [(i, label) for i, label in enumerate(potential_labels)]

    # Create the labels
    price_labels = []
    price_bracket = []
    for price in data["price"]:
        diff = abs(potential_labels - price)
        index = np.argmin(diff)
        price_labels.append(index)
        left = potential_labels[index - 1] if index > 0 else potential_labels[index]
        right = (
            potential_labels[index + 1]
            if index < len(potential_labels) - 1
            else potential_labels[index]
        )
        price_bracket.append((left, right))

    data["label"] = price_labels
    data["price_bracket"] = price_bracket
    return data, label_codes


def label_low_med_high(df: pd.DataFrame, onehot: bool) -> pd.DataFrame:
    """
    Add labels-column to the data-points, based on prices. THREE labels: low, medium, high
    """
    price_ranges = {
        "low": (0, df["price"].quantile(0.33)),
        "med": (df["price"].quantile(0.33), df["price"].quantile(0.66)),
        "high": (df["price"].quantile(0.66), df["price"].max()),
    }

    def label(price):
        if price >= price_ranges["low"][0] and price <= price_ranges["low"][1]:
            return 0
        elif price >= price_ranges["med"][0] and price <= price_ranges["med"][1]:
            return 1
        else:
            return 2

    df["label_price"] = df["price"].apply(label)
    return df



def scale_by_size(canvas_size, house_size, max_house_size, image):
    """
    Create a canvaz, and fit the image corresponding to the house size on the canvas and max_house_size
    """
    # Create a canvas
    canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8)
    # Scale the image
    scale = house_size / max_house_size  # <- Magic variable
    print(scale)
    new_width = int(image.shape[1] * scale)
    new_height = int(image.shape[0] * scale)
    # Resize the image
    resized_image = cv2.resize(
        image, (new_width, new_height), interpolation=cv2.INTER_NEAREST
    )
    # Place the image in the center of the canvas
    x_offset = int((canvas_size - new_width) / 2)
    y_offset = int((canvas_size - new_height) / 2)
    canvas[
        y_offset : y_offset + resized_image.shape[0],
        x_offset : x_offset + resized_image.shape[1],
    ] = resized_image
    # invert the colors
    canvas = cv2.bitwise_not(canvas)
    return canvas


def scale_by_size(canvas_size, house_sizes, images, max_house_size):
    """
    Create a canvaz, and fit the images corresponding to the house size on the canvas and max_house_size
    """
    canvases = []
    for house_size, image in zip(house_sizes, images):
        # Create a canvas
        canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8)
        # Scale the image
        scale = house_size / max_house_size
        new_width = int(image.shape[1] * scale)
        new_height = int(image.shape[0] * scale)
        # Resize the image
        resized_image = cv2.resize(
            image, (new_width, new_height), interpolation=cv2.INTER_NEAREST
        )
        # Place the image in the center of the canvas
        x_offset = int((canvas_size - new_width) / 2)
        y_offset = int((canvas_size - new_height) / 2)
        canvas[
            y_offset : y_offset + resized_image.shape[0],
            x_offset : x_offset + resized_image.shape[1],
        ] = resized_image
        # invert the colors
        canvas = cv2.bitwise_not(canvas)
        canvases.append(canvas)
    return canvases


def preprocces_data_old(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the data.
    """
    # df = df.drop(columns=["address"])
    # Feature Columns
    df["basement_size"] = df["basement_size"].fillna(0)
    df["year_rebuilt"] = (
        df["year_rebuilt"]
        .where(~df["year_rebuilt"].isna(), df["year_built"])
        .astype(int)
    )
    # df['type'] = df['type'].astype('category').cat.codes
    df["energy_label"] = df["energy_label"].astype("category").cat.codes
    # data.dropna(inplace=True)

    # Image Columns
    # df['image_floorplan'] = df['image_floorplan'].apply(convert_to_grayscale)
    # Optimal: use ImageGenerator to augment the images#

    # Adding Labels
    # df = (label_low_med_high(df, onehot=True))

    # Add a column that holds the image resolution
    df["image_resolution"] = df["image_floorplan"].apply(lambda x: x.shape)
    return df



# def preprocess_images(
#     df: pd.DataFrame,
#     column_name: str,
#     width: int,
#     height: int,
#     resize: bool,
#     gray_scale: bool,
#     threshhold: bool,
# ) -> np.array:
#     if resize:
#         images = resize_images(df, column_name, width, height)
#     if gray_scale:
#         images = convert_to_grayscale(images)
#     if threshhold:
#         images = threshold_images(images)
#     return images


# def resize_images(df, column_name: str, width: int, height: int) -> np.array:
#     resized_images = np.array(
#         [
#             cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)
#             for image in df[column_name]
#         ]
#     )
#     return resized_images