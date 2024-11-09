# Music Genre Classifier 🎶🎧

Welcome to the Music Genre Classifier! This AI-powered application is designed to classify music into genres using deep learning. Whether you're a music lover, DJ, or just curious, this tool offers a unique way to explore and analyze your music collection.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Dependencies](#dependencies)

## Features
- *AI-Powered Classification*: Classifies music into genres using deep learning.
- *Fast & Easy*: Just upload an audio file, and the app will predict the genre within seconds.
- *Visualization*: Displays a pie chart showing the genre composition of the audio.
- *User-Friendly Interface*: Built with Streamlit for an interactive and intuitive experience.

## Installation

To run this application locally, please follow the instructions below:

1. *Clone the Repository*:
   bash
   git clone https://github.com/ACHYUTAM2004/Music-genre-clasifier.git
   cd Music_Genre_Classifier
2. **Install Dependencies**:
   Ensure that you have Python installed. Then, install the required libraries
   bash
   pip install -r requirements.txt
3. *Download the Pre-trained Model*: Place the pre-trained model Trained_model_final.h5 in the project directory.
    (Due to space limitation, model is hosted externally)  
     Download pretrained model from #https://drive.google.com/file/d/1zH-zgsuZQU9jverJg5us5w_2YvCJOIO0/view?usp=sharing
4. *Run the Application:*
    bash
    streamlit run app.py
5. **Upload Audio File and enjoy**

## Usage
1. **Launch the App**: Run streamlit run app.py.
2. **Choose an Option**:
- About App: Overview of the app’s purpose.
- How it Works?: Detailed guide on the app's functionality.
- Predict Music Genre: Allows you to upload an audio file for genre prediction.
3. **Upload Audio File**: Select an mp3 file and upload it to the app.
4. **Play Audio**: Listen to the uploaded audio within the app.
5. **Predict Genre**: Click on "Know Genre" to classify the audio file. The app will display a pie chart showing genre predictions.

## How It Works
**The Music Genre Classifier leverages deep learning and digital signal processing techniques to analyze audio and predict genres.**

1. **Audio Processing:**
The uploaded audio file is divided into overlapping chunks to capture various sections of the song.
2. **Feature Extraction:**
Each audio chunk is converted to a Mel Spectrogram using torchaudio.
The spectrogram is resized to a (210, 210) format to match the input dimensions expected by the model.
3. **Model Prediction**:
The pre-trained TensorFlow model classifies each chunk into a genre.
The final genre is determined based on the frequency of genre predictions across all chunks.
4. **Visualization**:
A pie chart displays the genre distribution, with the primary genre highlighted.

## Dependencies
The project requires the following dependencies:

Streamlit - For building the web interface.
TensorFlow - For loading and using the pre-trained model.
Torch and Torchaudio - For audio processing.
Librosa - For audio file manipulation and loading.
NumPy - For numerical computations.
Plotly - For visualization of the pie chart.

**For installing all dependencies use**
bash
pip install -r requirements.txt
