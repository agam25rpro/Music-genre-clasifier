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

app = Flask(__name__)

# Download and load the model
def download_model():
    url = "https://drive.google.com/uc?export=download&id=1vc4b2RpeXmnZMn2SOF0snIjos9paVEVH"
    output = "Trained_model.h5"
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)

def load_model():
    download_model()
    model = tf.keras.models.load_model("Trained_model.h5")
    return model

# Load the model once and reuse it
model = load_model()

# Preprocess source file
def load_and_preprocess_file(file_path, target_shape=(210, 210)):
    data = []
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    chunk_duration = 4
    overlap_duration = 2
    chunk_samples = chunk_duration * sample_rate
    overlap_samples = overlap_duration * sample_rate
    num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1

    # Process and predict chunks one by one
    for i in range(num_chunks):
        start = i * (chunk_samples - overlap_samples)
        end = start + chunk_samples
        chunk = audio_data[start:end]

        # Convert chunk to Mel Spectrogram
        mel_spectrogram = torchaudio.transforms.MelSpectrogram()(torch.tensor(chunk).unsqueeze(0)).numpy()

        # Resize matrix based on provided target shape
        mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)

        # Instead of appending, process chunk and predict directly
        x_test = mel_spectrogram.reshape(1, target_shape[0], target_shape[1], 1)
        yield x_test  # Yielding the preprocessed data for prediction to save memory

# Predict values
def model_prediction(x_test):
    y_pred = model.predict(x_test)
    predicted_cats = np.argmax(y_pred, axis=1)
    unique_elements, counts = np.unique(predicted_cats, return_counts=True)
    max_count = np.max(counts)
    max_elements = unique_elements[counts == max_count]
    return unique_elements, counts, max_elements[0]

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
        return render_template('index.html', error_message=str(e))

if __name__ == '__main__':
    app.run(debug=True)
