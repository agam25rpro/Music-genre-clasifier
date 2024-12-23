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
# Preprocess source file
def load_and_preprocess_file(file_path, target_shape=(210,210)):
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


# Sidebar UI
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select page", ["About app", "How it works?", "Predict music genre"])

# Main page
if app_mode == "About app":
    st.markdown(
        """
        <style>
        .stapp {
            background-color: #0E1117;
            color: white;
        }
        h2 {
            color: #00BFFF; /* Deep Sky Blue for main title */
            font-size: 36px;
            font-weight: bold;
        }
        h3 {
            color: #32CD32; /* Light Blue for subtitles */
            font-size: 28px;
        }
        p {
            color: #f0f8ff; /* Light Gray for text */
            font-size: 18px;
        }
        .stmarkdown {
            color: #f0f8ff;
        }
        .stimage {
            border-radius: 15px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('''## Welcome to the,''')
    st.markdown('''## Music Genre Classifier 🎶🎧''')
    image_path = "music_genre_home.png"
    st.image(image_path, width=350)

    st.markdown("""
    ## Welcome to the Music Genre Classifier, an AI-powered app designed to help you explore and categorize music with ease! ✨
                
    Leveraging deep learning (DL) techniques, this app automatically analyzes your music tracks and classifies them into various genres with impressive accuracy.

    Whether you’re a music enthusiast, a DJ, or just someone who loves discovering new tunes, this app brings AI to your fingertips, transforming how you interact with music. Simply upload a song, and within seconds, our advanced model will determine the genre, providing you with a seamless and intuitive music discovery experience.

    ### **Key Features: 💪**

    AI-Powered Music Classification: Built using deep learning, our app accurately classifies music into multiple genres.
                
    Fast & Easy: Upload a track, and let the app work its magic in seconds.
                
    Explore New Music: Find genres you might not have explored before and discover new favorites.
                
    User-Friendly Interface: An easy-to-use design makes classifying your music a breeze.
                
    #### Let our deep learning model do the heavy lifting while you enjoy the world of music in a whole new way! 🎧

    (_P.S. -> It is an AI model so it may give wrong predictions too_)
    """)

elif app_mode == "How it works?":
    st.markdown("""
    # How to know the music genre?
    **1. Upload music: Start off with uploading the music file**\n
    **2. Analysis: Our system will process the music file with advanced algorithms to classify it into a number of genres**\n
    **3. Results: After the analysis phase, you will get a pie chart depicting the percentage of genres the music belongs to (A music is not purely a single genre)**

    #### _P.S. -> You can also listen to the music in the app itself_
    """)

elif app_mode == 'Predict music genre':
    st.header("**_Predict Music Genre_**")
    st.markdown('##### Upload the audio file (mp3 format)')
    test_mp3 = st.file_uploader('', type=['mp3'])

    if test_mp3 is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
            tmp_file.write(test_mp3.getbuffer())
            filepath = tmp_file.name
            st.success(f"File {test_mp3.name} uploaded successfully!")

    # Play audio
    if st.button("Play Audio") and test_mp3 is not None:
        st.audio(test_mp3)

    # Predict
    if st.button("Know Genre") and test_mp3 is not None:
        with st.spinner("Please wait ..."):
            X_test = load_and_preprocess_file(filepath)
            labels, values, c_index = model_prediction(X_test)
            st.snow()
            st.markdown("The music genre is : ")
            show_pie(values, labels, test_mp3)

