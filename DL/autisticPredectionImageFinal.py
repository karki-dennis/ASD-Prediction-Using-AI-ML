import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import PIL
from tensorflow import keras

# Configuration - override via environment variables for non-Colab environments
DATA_DIR = os.environ.get('ASD_DL_DATA_DIR', '/content/drive/MyDrive/audata/train')
MODEL_SAVE_PATH = os.environ.get('ASD_DL_MODEL_PATH', '/content/drive/MyDrive/resnet_model3.keras')
IMAGE_HEIGHT = 256
BATCH_SIZE = 50

data_dir = DATA_DIR

data_d = os.path.join(DATA_DIR, 'Autistic')
lst = os.listdir(data_d)
number_files = len(lst)
print(number_files)

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom

image_height = IMAGE_HEIGHT
batch_size = BATCH_SIZE

# Define image augmentation layers
data_augmentation = tf.keras.Sequential([
    RandomFlip("horizontal_and_vertical"),  # Randomly flip the images horizontally and vertically
    RandomRotation(0.2),  # Randomly rotate images by up to 20%
    RandomZoom(0.2),  # Randomly zoom into the images by up to 20%
])

# Load Data with Augmentation
train_ds_init = image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=3,
    image_size=(image_height, image_height),
    batch_size=batch_size
)
train_ds = train_ds_init.map(lambda x, y: (data_augmentation(x, training=True), y))  # Apply augmentation to training data

class_names = train_ds_init.class_names
print(class_names)

val_ds = image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=3,
    image_size=(image_height, image_height),
    batch_size=batch_size)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Flatten
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from keras.applications import ResNet50
from tensorflow.keras.callbacks import EarlyStopping

# Model Architecture
resnet_model = Sequential()

pretrained_model = ResNet50(include_top=False,
                            input_shape=(256, 256, 3),
                            pooling='avg',
                            weights='imagenet')

for layer in pretrained_model.layers:
    layer.trainable = False

resnet_model.add(pretrained_model)
resnet_model.add(Flatten())
resnet_model.add(Dense(512, activation='relu',kernel_regularizer=l2(0.001)))
resnet_model.add(BatchNormalization())  # Batch Normalization Layer
resnet_model.add(Dropout(0.5))  # Adding Dropout for Regularization
resnet_model.add(Dense(2, activation='softmax'))


# Compile Model
resnet_model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])

# Early Stopping Callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train Model with Early Stopping
epochs = 8
history = resnet_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[early_stopping]  # Include Early Stopping
)

resnet_model.save(MODEL_SAVE_PATH)

import cv2
from google.colab import files

def upload_and_predict_opencv(model, target_size=(IMAGE_HEIGHT, IMAGE_HEIGHT)):
    """Upload an image and predict ASD classification.

    Args:
        model: Trained Keras model for prediction.
        target_size: Tuple of (height, width) for image resizing.

    Returns:
        Predicted class name string, or None if prediction failed.
    """
    uploaded = files.upload()
    if not uploaded:
        print("No file uploaded.")
        return None

    file_name = list(uploaded.keys())[0]
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    if not file_name.lower().endswith(valid_extensions):
        print(f"Error: Unsupported file type. Please upload one of: {valid_extensions}")
        return None

    img = cv2.imread(file_name)
    if img is None:
        print(f"Error: Could not read image file '{file_name}'.")
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)

    img_array = np.expand_dims(img, axis=0) / 255.0

    plt.imshow(img)
    plt.axis('off')
    plt.show()

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=-1)
    predicted_class_name = class_names[predicted_class_index[0]]

    print(f"Predicted class: {predicted_class_name}")
    return predicted_class_name


predicted_class = upload_and_predict_opencv(resnet_model)

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score

val_labels = []
val_probs = []
val_preds = []

# Iterate over the validation dataset
for images, labels in val_ds:
    preds = resnet_model.predict(images)
    val_probs.extend(preds[:, 1])  # Assuming that class 1 is the positive class
    val_labels.extend(labels.numpy())
    val_preds.extend(np.argmax(preds, axis=-1))  # Predicted classes

# Calculate precision, recall, and F1 score
precision = precision_score(val_labels, val_preds)
recall = recall_score(val_labels, val_preds)
f1 = f1_score(val_labels, val_preds)

print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')

# Calculate the ROC curve
fpr, tpr, thresholds = roc_curve(val_labels, val_probs)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

