# Import libraries
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers
from tensorflow.keras import Model

# Define the paths
base_dir = '/content/drive/MyDrive/dataset'

# Print the number of images for each type
for category in os.listdir(base_dir):
    category_path = os.path.join(base_dir, category)
    if os.path.isdir(category_path):
        num_images = len(os.listdir(category_path))
        print(f"{category}: {num_images} images")

# Split the dataset for training
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(128, 128),
    batch_size=16,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(128, 128),
    batch_size=16,
    class_mode='categorical',
    subset='validation'
)

from google.colab import drive
drive.mount('/content/drive')

# Load the pre-trained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

# Freeze the convolutional layers
for layer in base_model.layers:
    layer.trainable = False

# Create your own model
x = layers.Flatten()(base_model.output)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(5, activation='softmax')(x)

model = Model(base_model.input, x)

# Train the model
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)

import matplotlib.pyplot as plt

# Plot the model graph
from tensorflow.keras.utils import plot_model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

# Visualize training metrics
plt.figure(figsize=(12, 4))

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.show()

# Save the trained model
model.save('food_classification_model.h5')

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Evaluate the model on the test set
test_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(128, 128),
    batch_size=16,
    class_mode='categorical',
    subset='validation'
)

# Load the trained model
loaded_model = tf.keras.models.load_model('food_classification_model.h5')

# Evaluate the model on the test set
evaluation = loaded_model.evaluate(test_generator)

# Print the evaluation metrics
print(f"Test Accuracy: {evaluation[1] * 100:.2f}%")
print(f"Test Loss: {evaluation[0]:.4f}")

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Predict on the test set
y_pred = loaded_model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes

# Classification Report
print("Classification Report:")
print(classification_report(y_true, y_pred_classes))

# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred_classes)
print("Confusion Matrix:")
print(conf_matrix)

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from IPython.display import Markdown, display

print(train_generator.class_indices)

# Load the trained model
loaded_model = tf.keras.models.load_model('food_classification_model.h5')

# Class indices mapping
class_indices = {'Apple': 0, 'Banana': 1, 'Carrot': 2, 'Lemon': 3, 'Mango': 4}


def predict_food(img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Make a prediction
    prediction = loaded_model.predict(img_array)

    # Get the predicted class
    predicted_class = np.argmax(prediction)

    # Get the predicted class name
    predicted_class_name = list(class_indices.keys())[list(class_indices.values()).index(predicted_class)]

    # Display the image
    plt.imshow(img)
    plt.axis('off')
    plt.show()

    # Print the predicted class name in bold
    display(Markdown(f"**Predicted Class Name: {predicted_class_name}**"))

image_path = '/content/drive/MyDrive/dataset/apple/Image_1.jpg'
predict_food(image_path)

image_path = '/content/drive/MyDrive/dataset/banana/Image_10.jpg'
predict_food(image_path)

image_path = '/content/drive/MyDrive/dataset/carrot/Image_6.jpg'
predict_food(image_path)

image_path = '/content/drive/MyDrive/dataset/lemon/Image_1.png'
predict_food(image_path)

image_path = '/content/drive/MyDrive/dataset/mango/Image_10.jpg'
predict_food(image_path)
