from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import numpy as np
import librosa
from tensorflow.image import resize
import torch
import torchaudio
import plotly.graph_objects as go
import gdown
import os
import tempfile
from threading import Thread

app = Flask(__name__)

# Download and load the model once, if not already downloaded
MODEL_PATH = "Trained_model.h5"

def download_model():
    if not os.path.exists(MODEL_PATH):
        url = "https://drive.google.com/uc?export=download&id=1vc4b2RpeXmnZMn2SOF0snIjos9paVEVH"
        gdown.download(url, MODEL_PATH, quiet=False)

def load_model():
    download_model()
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

# Load model globally
model = load_model()

# Preprocess source file
def load_and_preprocess_file(file_path, target_shape=(210, 210)):
    try:
        audio_data, sample_rate = librosa.load(file_path, sr=None)
        chunk_duration = 4  # Duration of each chunk in seconds
        overlap_duration = 2  # Duration of overlap in seconds
        chunk_samples = chunk_duration * sample_rate
        overlap_samples = overlap_duration * sample_rate
        num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1

        # Process and predict chunks one by one to save memory
        for i in range(num_chunks):
            start = i * (chunk_samples - overlap_samples)
            end = start + chunk_samples
            chunk = audio_data[start:end]

            # Convert chunk to Mel Spectrogram
            mel_spectrogram = torchaudio.transforms.MelSpectrogram()(torch.tensor(chunk).unsqueeze(0)).numpy()

            # Resize matrix using TensorFlow's resize function
            mel_spectrogram = resize(tf.convert_to_tensor(np.expand_dims(mel_spectrogram, axis=-1), dtype=tf.float32), target_shape)

            # Yield preprocessed chunk for prediction (to save memory)
            yield tf.reshape(mel_spectrogram, (1, target_shape[0], target_shape[1], 1))
    except Exception as e:
        raise Exception(f"Error in file preprocessing: {str(e)}")

# Predict values from the model
def model_prediction(x_test):
    try:
        y_pred = model.predict(x_test)
        predicted_cats = np.argmax(y_pred, axis=1)
        unique_elements, counts = np.unique(predicted_cats, return_counts=True)
        max_count = np.max(counts)
        max_elements = unique_elements[counts == max_count]
        return unique_elements, counts, max_elements[0]
    except Exception as e:
        raise Exception(f"Error in model prediction: {str(e)}")

# Flask Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return render_template('index.html', error_message="No file uploaded")

    file = request.files['audio']
    if file.filename == '':
        return render_template('index.html', error_message="No file selected")

    try:
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
            file.save(tmp_file.name)
            filepath = tmp_file.name

        # Initialize variables for predictions
        all_labels = []
        all_values = []

        # Preprocess the audio file and predict genre chunk by chunk
        for x_test in load_and_preprocess_file(filepath):
            labels, values, c_index = model_prediction(x_test)
            all_labels.extend(labels)
            all_values.extend(values)

        # Genre classes
        classes = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
        genre = classes[c_index]  # Predicted genre

        # Prepare data for the pie chart
        genre_labels = [classes[i] for i in all_labels]

        # Remove the temporary file
        os.remove(filepath)

        return render_template(
            'index.html',
            genre_prediction=genre,
            pie_values=all_values,
            pie_labels=genre_labels
        )
    except Exception as e:
        return render_template('index.html', error_message=f"An error occurred: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
