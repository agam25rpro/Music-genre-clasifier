import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
from tensorflow.image import resize
import torch
import torchaudio
import plotly.graph_objects as go
import gdown
import tempfile
from fastapi import FastAPI, UploadFile, File
import uvicorn
import io

# Initialize FastAPI app
app = FastAPI()

def download_model():
    url = "https://drive.google.com/uc?export=download&id=1vc4b2RpeXmnZMn2SOF0snIjos9paVEVH"
    output = "Trained_model.h5"
    gdown.download(url, output, quiet=False)

# Load the model after downloading
def load_model():
    download_model()
    model = tf.keras.models.load_model("Trained_model.h5")
    return model

# Preprocess source file
def load_and_preprocess_file(file_path, target_shape=(210, 210)):
    data = []
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    chunk_duration = 4
    overlap_duration = 2
    chunk_samples = chunk_duration * sample_rate
    overlap_samples = overlap_duration * sample_rate
    num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1
    
    for i in range(num_chunks):
        start = i * (chunk_samples - overlap_samples)
        end = start + chunk_samples
        chunk = audio_data[start:end]
        
        # Convert chunk to Mel Spectrogram
        mel_spectrogram = torchaudio.transforms.MelSpectrogram()(torch.tensor(chunk).unsqueeze(0)).numpy()
        
        # Resize matrix based on provided target shape (150, 150)
        mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
        data.append(mel_spectrogram)
    
    # Convert to numpy array and ensure correct shape: (num_chunks, height, width, channels)
    return np.array(data).reshape(-1, target_shape[0], target_shape[1], 1)

# Predict values
def model_prediction(x_test):
    model = load_model()
    y_pred = model.predict(x_test)
    predicted_cats = np.argmax(y_pred, axis=1)
    unique_elements, counts = np.unique(predicted_cats, return_counts=True)
    max_count = np.max(counts)
    max_elements = unique_elements[counts == max_count]
    return unique_elements, counts, max_elements[0]

# Show pie chart
def show_pie(values, labels, test_mp3):
    classes = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    
    # Get the genre names corresponding to the labels
    genre_labels = [classes[i] for i in labels]
    
    # Create a plotly pie chart
    fig = go.Figure(
        go.Pie(
            labels=genre_labels,
            values=values,
            hole=0.3,  # Creates a donut chart
            textinfo='label+percent',  # Show label and percentage on the chart
            insidetextorientation='radial',  # Display text in a radial fashion
            pull=[0.2 if i == np.argmax(values) else 0 for i in range(len(values))],  # Highlight the largest slice
            textfont=dict(
                family="Arial, sans-serif",  # Font family
                size=14,  # Font size
                color="white",  # Font color
                weight="bold"  # Make text bold
            )
        )
    )
    
    # Update the title
    fig.update_layout(
        title_text=f"Music Genre Classification: {test_mp3.name}",
        title_x=0.5,  # Center the title
        height=600,  # Increase the height
        width=600,   # Increase the width
        legend=dict(
            font=dict(
                family="Arial, sans-serif",  # Font family for legend
                size=16,  # Font size for legend text
                color="white"  # Font color for legend text
            ),
            title="Genres",  # Optional: You can add a title to the legend
            title_font=dict(
                size=18,  # Font size for the legend title
                color="white"
            )
        )
    )
    
    # Show the plot in Streamlit
    st.plotly_chart(fig)

# FastAPI route to predict genre from uploaded audio file
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
        tmp_file.write(file.file.read())
        filepath = tmp_file.name
        X_test = load_and_preprocess_file(filepath)
        labels, values, c_index = model_prediction(X_test)
        return {"genre": labels, "values": values.tolist(), "genre_name": str(labels[0])}

# Streamlit frontend
def streamlit_ui():
    # Sidebar UI
    st.sidebar.title("Dashboard")
    app_mode = st.sidebar.selectbox("Select page", ["About app", "How it works?", "Predict music genre"])

    if app_mode == "About app":
        st.markdown("**About app**")
        # Add About app information here

    elif app_mode == "How it works?":
        st.markdown("**How it works?**")
        # Add How it works information here

    elif app_mode == 'Predict music genre':
        st.header("**_Predict Music Genre_**")
        st.markdown('##### Upload the audio file (mp3 format)')
        test_mp3 = st.file_uploader('', type=['mp3'])

        if test_mp3 is not None:
            st.audio(test_mp3)

        # Predict
        if st.button("Know Genre") and test_mp3 is not None:
            with st.spinner("Please wait ..."):
                result = model_prediction(test_mp3)
                st.snow()
                show_pie(result['values'], result['labels'], test_mp3)

# Run both FastAPI and Streamlit together
def main():
    import threading
    # Run Streamlit UI in a separate thread
    threading.Thread(target=streamlit_ui).start()
    # Run FastAPI in the main thread
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
