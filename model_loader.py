import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all logs, 3 = fatal only


from tensorflow.keras.models import load_model

model = load_model("C:\\Users\\harsh\\PycharmProjects\\Smart_Chair\\posture_model.h5", compile=False)

# Then you can use it:
# input_data must be shape (1, 4, 1)
import numpy as np

# Global variables to accumulate data
data = [0, 0, 0, 0]         # Running sum of 4 values
mean_data = [0, 0, 0, 0]    # Mean after 6 data sets
count = 0                   # How many inputs received

def gather_data(sensor_data):
    global data, mean_data, count

    for i in range(4):
        data[i] += sensor_data[i]

    count += 1

    if count == 6:
        # Compute the mean
        mean_data = [val / 6 for val in data]
        #print("6 inputs received. Mean:", mean_data)

        result = mean_data.copy()

        # Reset for next batch
        data = [0, 0, 0, 0]
        mean_data = [0, 0, 0, 0]
        count = 0
        return result
    else:
        return []

def get_posture(mean_data):
    new_input = np.array([mean_data]).reshape(1, 4, 1)
    prediction = model.predict(new_input)
    predicted_class = int(np.argmax(prediction))
    confidence = float(np.max(prediction) * 100)
    #print(f"Predicted class: {predicted_class}, Confidence: {confidence:.2f}%")
    return [predicted_class, confidence]

print(get_posture([2.23, 0.062, 3.693, 279.93]))

