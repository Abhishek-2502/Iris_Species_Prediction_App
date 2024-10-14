import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# Load dataset
df = pd.read_csv('D:/Docs/Abhi/SIT/CSE/Android_Dev/Kotlin Practice and Project/PracticeApps/Iris_Species_Prediction_App/ML/Iris.csv')
# Features and labels
X = df.iloc[:, 1:5].values  # Exclude 'Id'
y = df.iloc[:, 5].values  # 'Species' column

# Label encoding
le = LabelEncoder()
y = le.fit_transform(y)
y = to_categorical(y)

# Build the model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=[4]))  # 4 input features
model.add(Dense(64, activation='relu'))  # Added activation function here
model.add(Dense(3, activation='softmax'))  # 3 output classes

# Compile the model
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['acc'])

# Train the model
model.fit(X, y, epochs=200)

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tfmodel = converter.convert()

# Save the TFLite model
with open('iris.tflite', 'wb') as f:
    f.write(tfmodel)

print("Model saved as 'iris.tflite'")
