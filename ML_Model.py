import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Load the Excel file
df = pd.read_excel("C:\\Users\\harsh\\OneDrive\\Documents\\Smart Chair\\Test.xlsx")

# Split data and labels
X = df.iloc[:, 0:4].values  # 4 sensor inputs == X is a matrix of size (no.of samples * 4)
y = df.iloc[:, 4].values    # posture label

# Encode labels to integers (i.e. posture 1 = 1)
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Reshape input for Conv1D: (samples, timesteps, features) as it needs a 3D input
# Each sensor reading is treated as one feature.
#[ [[2.3],[4.5],[0.002],[123]], [[2.3],[4.5],[0.002],[123]] ]
X_reshaped = X.reshape(X.shape[0], X.shape[1], 1) # X.shape = [no.of samples, 4] i.e. = X.reshape(samples, 4, 1)

# Train-test split
# 20% of the data is kept for testing
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_categorical, test_size=0.2)

# Build CNN Model
model = Sequential()

# 👇 You can adjust these CNN layers
# ReLU(x) = max(0, x) x if x >=0 and 0 when x is negative.
model.add(Conv1D(filters=16, kernel_size=2, activation='relu', input_shape=(4, 1)))
model.add(Conv1D(filters=32, kernel_size=2, activation='relu'))
model.add(Conv1D(filters=64, kernel_size=2, activation='relu'))
# model.add(Conv1D(filters=64, kernel_size=2, activation='relu'))
model.add(Flatten())

# Dense Layers
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))  # Keeps 30% of the neurons when training to avoid overfitting of data.
model.add(Dense(y_categorical.shape[1], activation='softmax'))  # output layer

# Compile the model
# Adaptive moment estimation - ADAM Pays atention for how much the error changes with wieghts,
# how fast the weights are moving
#Categorical crossentrophy compares the hot encoded value with the predicted value to get the error
# [1 0 0] with predicted value [0.8 0.1 0.1]
# TensorFlow is asked to track and report the accuracy of the training
# Since One hot encoded data, categorical crosssentrophy is used. [1, 0, 0, 0]
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
history = model.fit(X_train, y_train, epochs=50, validation_split=0.2, verbose=1)

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\n✅ Model Accuracy: {accuracy * 100:.2f}%")

# Make predictions
pred_probs = model.predict(X_test)
pred_labels = np.argmax(pred_probs, axis=1)
confidence = np.max(pred_probs, axis=1) * 100  # highest probability as confidence

# Show example predictions
for i in range(10):
    label_name = le.inverse_transform([pred_labels[i]])[0]
    print(f"Prediction: {label_name}, Confidence: {confidence[i]:.2f}%")

model.save("C:\\Users\\harsh\\PycharmProjects\\Smart_Chair\\posture_model.h5")  # HDF5 format (recommended for LabVIEW integration)