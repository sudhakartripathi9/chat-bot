pip install tensorflow opencv-python scikit-image matplotlib



import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Function to load images
def load_images_from_folder(folder):
    images = []
    labels = []
    for label in os.listdir(folder):
        label_folder = os.path.join(folder, label)
        for filename in os.listdir(label_folder):
            img = cv2.imread(os.path.join(label_folder, filename))
            if img is not None:
                img = cv2.resize(img, (300, 300))  # Resize to 300x300
                images.append(img)
                labels.append(label)
    return np.array(images), np.array(labels)

# Load images
X, y = load_images_from_folder('path_to_your_dataset')

# Split the data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalize the images
X_train = X_train / 255.0
X_val = X_val / 255.0




import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Create a data generator for augmentation
datagen = ImageDataGenerator(rotation_range=20,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             shear_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True,
                             fill_mode='nearest')

# Load MobileNetV2 model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(300, 300, 3))
base_model.trainable = False  # Freeze the base model

# Build the model
model = Sequential([
    base_model,
    Flatten(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(len(np.unique(y)), activation='softmax')  # Number of classes
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(datagen.flow(X_train, y_train, batch_size=32), 
          validation_data=(X_val, y_val), 
          epochs=50)



# Save the trained model
model.save('rice_disease_detector.h5')




from flask import Flask, request, render_template
import numpy as np
from tensorflow.keras.models import load_model
import cv2

app = Flask(__name__)
model = load_model('rice_disease_detector.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (300, 300)) / 255.0
    img = np.expand_dims(img, axis=0)
    
    prediction = model.predict(img)
    class_index = np.argmax(prediction, axis=1)
    
    return str(class_index)  # Return the predicted class

if __name__ == '__main__':
    app.run(debug=True)



<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rice Disease Detector</title>
</head>
<body>
    <h1>Upload Rice Leaf Image</h1>
    <form action="/predict" method="post" enctype="multipart/form-data">
        <input type="file" name="image" accept="image/*" required>
        <button type="submit">Predict</button>
    </form>
</body>
</html>



