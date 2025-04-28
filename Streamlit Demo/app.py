import streamlit as st
import json
import time
import matplotlib.pyplot as plt
import numpy as np
import os
import openai
from dotenv import load_dotenv
import text2emotion as te
from nrclex import NRCLex
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline
import nltk
from analyzer import analyze_user_input
from responder import generate_response
import base64

# Set up NLTK resources
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

# Set up OpenAI API key (pre-configured)
# openai.api_key = 
# os.environ["OPENAI_API_KEY"] = openai.api_key

# Set up Transformer pipeline
classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

# Set page configuration
st.set_page_config(
    page_title="Speech Emotion AI Therapist",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #4e54c8;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subheader {
        font-size: 1.5rem;
        font-weight: 500;
        color: #6e7ac8;
        margin-bottom: 1rem;
    }
    .response-box {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        border-left: 5px solid #4e54c8;
        margin: 10px 0;
    }
    .emotion-box {
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        text-align: center;
        font-weight: 500;
    }
    .insight-box {
        background-color: #f0f2f6;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        border-left: 5px solid rgba(78, 84, 200, 0.7);
    }
    .stProgress > div > div > div > div {
        background-color: #4e54c8;
    }
    .audio-recorder {
        margin-bottom: 20px;
    }
    .speech-text {
        border-left: 3px solid #32CD32;
        padding-left: 10px;
        margin: 10px 0;
        font-style: italic;
    }
    .pulsating-dot {
        width: 20px;
        height: 20px;
        background-color: #f44336;
        border-radius: 50%;
        margin-right: 10px;
        display: inline-block;
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0% {
            transform: scale(0.95);
            box-shadow: 0 0 0 0 rgba(244, 67, 54, 0.7);
        }
        
        70% {
            transform: scale(1);
            box-shadow: 0 0 0 10px rgba(244, 67, 54, 0);
        }
        
        100% {
            transform: scale(0.95);
            box-shadow: 0 0 0 0 rgba(244, 67, 54, 0);
        }
    }
</style>
""", unsafe_allow_html=True)

# Function to analyze emotions using multiple methods
def analyze_emotions_comprehensive(text):
    """Analyze emotions using multiple methods and return combined results"""
    results = {}
    
    # Text2Emotion (5 basic emotions)
    try:
        t2e_result = te.get_emotion(text)
        results["text2emotion"] = t2e_result
    except Exception as e:
        results["text2emotion_error"] = str(e)
    
    # NRCLex (more nuanced emotions)
    try:
        nrc = NRCLex(text)
        results["nrclex_raw"] = nrc.raw_emotion_scores
        results["nrclex_top"] = nrc.top_emotions
    except Exception as e:
        results["nrclex_error"] = str(e)
    
    # VADER (sentiment analysis)
    try:
        sid = SentimentIntensityAnalyzer()
        results["vader"] = sid.polarity_scores(text)
    except Exception as e:
        results["vader_error"] = str(e)
    
    # Transformer-based model (more accurate for modern text)
    try:
        transformer_result = classifier(text)[0]
        transformer_dict = {item['label']: round(item['score'], 4) for item in transformer_result}
        results["transformer"] = transformer_dict
    except Exception as e:
        results["transformer_error"] = str(e)
    
    return results

# Function to handle speech transcription using OpenAI Whisper with better error handling
def transcribe_audio(audio_file):
    try:
        # Verify the file is valid
        file_size = os.fstat(audio_file.fileno()).st_size
        if file_size == 0:
            st.error("Audio file is empty. Please try recording again.")
            return None
        
        # Use a different approach for audio conversion
        try:
            # Create OpenAI client
            client = openai.OpenAI(api_key=openai.api_key)
            
            # Transcribe the audio
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
            
            return transcript.text
        except Exception as e:
            st.error(f"OpenAI API error: {str(e)}")
            
            # Alternative approach: Let's create a file uploader as a backup
            st.markdown("### ⚠️ Voice recording failed")
            st.markdown("""
            Please try uploading an audio file instead, or type your message below.
            """)
            
            audio_upload = st.file_uploader("Upload audio file", type=["mp3", "wav", "webm", "m4a"])
            if audio_upload is not None:
                try:
                    transcript = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_upload
                    )
                    return transcript.text
                except Exception as e2:
                    st.error(f"Error with uploaded audio: {str(e2)}")
            
            return None
    except Exception as e:
        st.error(f"Error handling audio: {str(e)}")
        return None

# Javascript for audio recording
# Function to get audio data URL directly from the browser
def get_audio_recorder_js():
    return """
    <script>
    const sleep = time => new Promise(resolve => setTimeout(resolve, time));
    const audioChunks = [];
    let mediaRecorder;
    let audioBlob;
    let isRecording = false;

    const startRecording = async () => {
        audioChunks.length = 0;
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream, {mimeType: 'audio/webm'});
            
            mediaRecorder.addEventListener("dataavailable", event => {
                audioChunks.push(event.data);
            });
            
            mediaRecorder.start();
            isRecording = true;
            
            document.getElementById('record-status').textContent = 'Recording... (Click again to stop)';
            document.getElementById('record-button').classList.add('recording');
            document.getElementById('recording-indicator').style.display = 'inline-block';
            
        } catch (err) {
            console.error("Error accessing microphone:", err);
            document.getElementById('record-status').textContent = 'Error: ' + err.message;
        }
    };

    const stopRecording = () => {
        if (!mediaRecorder || mediaRecorder.state === 'inactive') return;
        
        mediaRecorder.addEventListener("stop", () => {
            try {
                audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                
                // Create an object URL for the audio blob
                const audioUrl = URL.createObjectURL(audioBlob);
                
                // Create an audio element to let user hear the recording
                const audioPlayer = document.createElement('audio');
                audioPlayer.controls = true;
                audioPlayer.src = audioUrl;
                
                // Add the audio player to the page
                const audioContainer = document.getElementById('audio-playback');
                audioContainer.innerHTML = '';
                audioContainer.appendChild(audioPlayer);
                
                // Use Fetch API to upload the audio file
                const formData = new FormData();
                formData.append('audio', audioBlob, 'recording.webm');
                
                // Show upload status
                document.getElementById('record-status').textContent = 'Uploading audio...';
                
                // Here you would typically upload to a server endpoint
                // Since we're in Streamlit, we'll convert to base64 and use hidden input
                const reader = new FileReader();
                reader.readAsDataURL(audioBlob);
                reader.onloadend = () => {
                    const base64data = reader.result.split(',')[1];
                    document.getElementById('audio-data').value = base64data;
                    document.getElementById('submit-audio').click();
                    document.getElementById('record-status').textContent = 'Processing audio...';
                };
            } catch (err) {
                console.error("Error processing recording:", err);
                document.getElementById('record-status').textContent = 'Error processing recording: ' + err.message;
            }
        });
        
        mediaRecorder.stop();
        
        // Stop tracks in the stream
        if (mediaRecorder.stream) {
            mediaRecorder.stream.getTracks().forEach(track => track.stop());
        }
        
        isRecording = false;
        document.getElementById('record-button').classList.remove('recording');
        document.getElementById('recording-indicator').style.display = 'none';
    };

    const toggleRecording = () => {
        if (isRecording) {
            stopRecording();
        } else {
            startRecording();
        }
    };
    </script>

    <style>
    #record-button {
        background-color: #4e54c8;
        border: none;
        color: white;
        padding: 12px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 4px;
        transition: background-color 0.3s;
    }
    #record-button.recording {
        background-color: #f44336;
    }
    #record-button:hover {
        opacity: 0.8;
    }
    #record-status {
        margin-top: 10px;
        color: #666;
    }
    #recording-indicator {
        display: none;
    }
    #audio-playback {
        margin-top: 10px;
        width: 100%;
    }
    #audio-playback audio {
        width: 100%;
    }
    </style>

    <div>
        <button id="record-button" onclick="toggleRecording()">🎙️ Click to Record</button>
        <span id="recording-indicator" class="pulsating-dot"></span>
        <div id="record-status">Click the button to start recording</div>
        <div id="audio-playback"></div>
    </div>
    """

# Function to extract key insights for display
def extract_key_insights(psychoanalysis, threshold=0.4):
    insights = []
    
    for category, entries in psychoanalysis.items():
        if isinstance(entries, list):
            for insight in entries:
                if not isinstance(insight, dict):
                    continue
                    
                description = insight.get("description", "")
                confidence = insight.get("short_term", 0)
                
                if confidence >= threshold and description:
                    insights.append({
                        "category": category,
                        "description": description,
                        "confidence": confidence
                    })
    
    # Sort by confidence
    return sorted(insights, key=lambda x: x["confidence"], reverse=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'last_analysis' not in st.session_state:
    st.session_state.last_analysis = None
if 'emotion_analysis_method' not in st.session_state:
    st.session_state.emotion_analysis_method = "Comprehensive (All Methods)"

# Header
st.markdown("<h1 class='main-header'>Speech Emotion AI Therapist</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Speak or type your thoughts and our AI will analyze emotions and provide supportive responses</p>", unsafe_allow_html=True)

# Sidebar with settings
with st.sidebar:
    st.markdown("<h2 class='subheader'>Settings</h2>", unsafe_allow_html=True)
    
    # Analysis method selection
    st.markdown("### Emotion Analysis Method")
    emotion_analysis_method = st.radio(
        "Select analysis method:",
        ["Comprehensive (All Methods)", "Text2Emotion", "NRCLex", "VADER Sentiment", "Transformer"],
        index=0,
        help="Choose which emotion analysis method to use. Comprehensive uses all methods."
    )
    st.session_state.emotion_analysis_method = emotion_analysis_method
    
    response_model = st.selectbox(
        "Response Generation Model",
        ["gpt-4", "gpt-3.5-turbo"],
        index=0
    )
    
    show_analysis = st.checkbox("Show Emotion Analysis Details", value=True)
    show_psychoanalysis = st.checkbox("Show Psychoanalytic Insights", value=False)
    
    st.markdown("---")
    st.markdown("<h2 class='subheader'>About</h2>", unsafe_allow_html=True)
    st.markdown(
        """
        This application uses:
        - **Speech Recognition**: OpenAI Whisper for accurate transcription
        - **Emotion Analysis**: Multiple models for comprehensive detection
        - **Psychological Analysis**: AI-powered insights into patterns and traits
        - **Response Generation**: Therapeutic responses using OpenAI GPT models
        """
    )

# Main content layout
col1, col2 = st.columns([2, 3])

with col1:
    st.markdown("<h2 class='subheader'>Speak or Type</h2>", unsafe_allow_html=True)
    
    # Audio recording component
    st.markdown("<div class='audio-recorder'>", unsafe_allow_html=True)
    st.components.v1.html(get_audio_recorder_js(), height=150)
    
    # Hidden elements for audio submission - using container to hide completely
    with st.container():
        st.markdown("<div style='display: none;'>", unsafe_allow_html=True)
        audio_data = st.text_area("", key="audio_data", help="")
        submit_audio = st.button("Submit Audio", key="submit_audio")
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Text input as alternative to speech
    st.markdown("### Or type your message:")
    user_input = st.text_area("Enter your thoughts and feelings...", height=120)
    submit_text = st.button("Submit", use_container_width=True)
    
    # Process audio submission
    speech_text = None
    if submit_audio and audio_data:
        with st.spinner("Transcribing audio..."):
            try:
                # Try a more reliable approach - write to file directly without base64 decode
                import tempfile
                
                # Create temporary audio file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as temp_audio_file:
                    temp_path = temp_audio_file.name
                
                # Alternative approach - create temporary file from raw bytes
                with open(temp_path, 'wb+') as f:
                    try:
                        # Try standard base64 decode first
                        f.write(base64.b64decode(audio_data))
                    except Exception as e1:
                        # If that fails, try a more permissive approach
                        try:
                            # Remove potential problematic characters
                            cleaned_data = ''.join(c for c in audio_data if c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=')
                            # Add padding if needed
                            while len(cleaned_data) % 4 != 0:
                                cleaned_data += '='
                            f.write(base64.b64decode(cleaned_data))
                        except Exception as e2:
                            # Last resort - try uploading a file directly
                            st.error("Could not process the audio recording. Please try another approach.")
                            uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a", "webm"])
                            if uploaded_file is not None:
                                with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as temp_file:
                                    temp_path = temp_file.name
                                    temp_file.write(uploaded_file.read())
                
                # Open the file for transcription
                with open(temp_path, 'rb') as audio_file:
                    speech_text = transcribe_audio(audio_file)
                
                # Clean up the temp file
                try:
                    os.unlink(temp_path)
                except:
                    pass
                
                if speech_text:
                    st.markdown("<div class='speech-text'>", unsafe_allow_html=True)
                    st.markdown(f"**Transcribed:** {speech_text}")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Process the transcribed text
                    user_input = speech_text
                    submit_text = True
                else:
                    st.info("No speech detected or transcription failed. Please try speaking more clearly or typing your message.")
            except Exception as e:
                st.error(f"Audio processing error: {str(e)}")
                st.info("Please type your message instead.")
                
                # Add a file uploader as a backup option
                audio_upload = st.file_uploader("Or upload an audio file", type=["mp3", "wav", "m4a", "webm"])
                if audio_upload is not None:
                    try:
                        client = openai.OpenAI(api_key=openai.api_key)
                        
                        speech_text = client.audio.transcriptions.create(
                            model="whisper-1",
                            file=audio_upload
                        ).text
                        
                        if speech_text:
                            st.markdown("<div class='speech-text'>", unsafe_allow_html=True)
                            st.markdown(f"**Transcribed:** {speech_text}")
                            st.markdown("</div>", unsafe_allow_html=True)
                            
                            # Process the transcribed text
                            user_input = speech_text
                            submit_text = True
                    except Exception as e2:
                        st.error(f"Error with uploaded audio: {str(e2)}")
    
    # Process text input (either typed or transcribed)
    if submit_text and user_input:
        with st.spinner("Processing your message..."):
            # Create a progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Step 1: Emotion Analysis
                status_text.text("Analyzing emotions...")
                progress_bar.progress(25)
                
                # Perform comprehensive emotion analysis
                emotions_result = analyze_emotions_comprehensive(user_input)
                
                # Choose which result to use based on user selection
                if st.session_state.emotion_analysis_method == "Text2Emotion":
                    primary_emotions = emotions_result.get("text2emotion", {})
                elif st.session_state.emotion_analysis_method == "NRCLex":
                    primary_emotions = emotions_result.get("nrclex_raw", {})
                elif st.session_state.emotion_analysis_method == "VADER Sentiment":
                    primary_emotions = emotions_result.get("vader", {})
                elif st.session_state.emotion_analysis_method == "Transformer":
                    primary_emotions = emotions_result.get("transformer", {})
                else:  # Comprehensive - use transformer as primary
                    primary_emotions = emotions_result.get("transformer", {})
                
                # Step 2: Psychoanalysis
                status_text.text("Performing psychological analysis...")
                progress_bar.progress(50)
                
                # Use the existing analyzer for psychoanalysis
                analysis_result = analyze_user_input(user_input)
                
                # Combine our emotion analysis with the analyzer's output
                analysis_result["emotions"] = primary_emotions
                analysis_result["all_emotions"] = emotions_result
                
                # Step 3: Response Generation
                status_text.text("Generating response...")
                progress_bar.progress(75)
                
                ai_response = generate_response(
                    user_text=user_input,
                    psychoanalysis=analysis_result["psychoanalysis"],
                    emotions=primary_emotions,
                    about_user=analysis_result["about_user"],
                    model=response_model
                )
                
                # Step 4: Completion
                progress_bar.progress(100)
                status_text.empty()
                
                # Add to chat history
                st.session_state.chat_history.append({
                    "user": user_input,
                    "ai": ai_response,
                    "timestamp": time.strftime("%H:%M:%S"),
                    "analysis": analysis_result,
                    "speech": speech_text is not None
                })
                
                st.session_state.last_analysis = analysis_result
                
            except Exception as e:
                st.error(f"Error processing your message: {str(e)}")
                st.error("Please check that all required libraries are installed")
    
    # Display chat history
    st.markdown("<h3 class='subheader'>Conversation History</h3>", unsafe_allow_html=True)
    
    for i, entry in enumerate(reversed(st.session_state.chat_history)):
        if entry.get("speech"):
            st.markdown(f"**You ({entry['timestamp']}):** 🎙️ *via speech*")
        else:
            st.markdown(f"**You ({entry['timestamp']}):**")
        st.markdown(f"{entry['user']}")
        
        st.markdown("<div class='response-box'>", unsafe_allow_html=True)
        st.markdown(f"**AI Therapist:**\n{entry['ai']}")
        st.markdown("</div>", unsafe_allow_html=True)
        
        if i < len(st.session_state.chat_history) - 1:
            st.markdown("---")

with col2:
    if st.session_state.last_analysis:
        analysis = st.session_state.last_analysis
        
        st.markdown("<h2 class='subheader'>Emotion Analysis</h2>", unsafe_allow_html=True)
        
        # Get the selected emotion analysis method
        method = st.session_state.emotion_analysis_method
        
        # Create tabs for different emotion analysis methods
        if "all_emotions" in analysis:
            emotion_tabs = st.tabs(["Main Results", "Text2Emotion", "NRCLex", "VADER", "Transformer"])
            
            with emotion_tabs[0]:
                # Primary emotions (based on selected method)
                primary_emotions = analysis.get("emotions", {})
                
                # Title based on selected method
                if method == "Comprehensive (All Methods)":
                    st.markdown("### Transformer-Based Analysis (Primary)")
                else:
                    st.markdown(f"### {method} Analysis")
                
                # Sort emotions by score
                if isinstance(primary_emotions, dict):
                    sorted_emotions = dict(sorted(primary_emotions.items(), key=lambda x: x[1], reverse=True))
                    
                    # Create horizontal bar chart for primary emotions
                    if sorted_emotions:
                        fig, ax = plt.subplots(figsize=(10, 5))
                        
                        # Define colors for emotions - expanded color palette
                        emotion_colors = {
                            # Basic emotions
                            'joy': '#FFD700',      # Gold
                            'surprise': '#FF8C00', # Dark Orange
                            'neutral': '#A9A9A9',  # Dark Gray
                            'sadness': '#4682B4',  # Steel Blue
                            'fear': '#800080',     # Purple
                            'disgust': '#006400',  # Dark Green
                            'anger': '#B22222',    # Firebrick
                            
                            # VADER sentiment
                            'pos': '#32CD32',      # Lime Green
                            'neg': '#DC143C',      # Crimson
                            'neu': '#A9A9A9',      # Dark Gray
                            'compound': '#4169E1', # Royal Blue
                            
                            # Additional emotions
                            'trust': '#40E0D0',    # Turquoise
                            'anticipation': '#FF69B4', # Hot Pink
                            'positive': '#32CD32',     # Lime Green
                            'negative': '#DC143C',     # Crimson
                            'Happy': '#FFD700',      # Gold
                            'Sad': '#4682B4',        # Steel Blue
                            'Angry': '#B22222',      # Firebrick
                            'Surprise': '#FF8C00',   # Dark Orange
                            'Fear': '#800080',       # Purple
                        }
                        
                        # Get default color for emotions not in the dictionary
                        default_color = '#1f77b4'  # Default matplotlib blue
                        
                        # Limit to top 10 emotions for visibility
                        top_emotions = list(sorted_emotions.items())[:10]
                        
                        # Create bars with appropriate colors
                        bars = ax.barh(
                            [k for k, v in top_emotions],
                            [v for k, v in top_emotions],
                            color=[emotion_colors.get(k, default_color) for k, v in top_emotions]
                        )
                        
                        # Add percentage labels
                        for bar in bars:
                            width = bar.get_width()
                            label_x_pos = width + 0.01
                            ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, 
                                    f'{width:.2f}' if width < 0.01 else f'{width:.1%}',
                                    va='center', fontsize=10)
                        
                        # Set appropriate X-axis limit based on data type
                        has_percentage = any(v > 0 and v < 1 for v in sorted_emotions.values())
                        if has_percentage:
                            ax.set_xlim(0, 1.0)
                        
                        ax.set_title(f'Detected Emotions ({method})', fontsize=14)
                        ax.set_xlabel('Score / Confidence')
                        fig.tight_layout()
                        
                        st.pyplot(fig)
                    else:
                        st.warning("No emotion data available for this method.")
                
                # Show top emotions as colored boxes
                if sorted_emotions:
                    st.markdown("<h3>Dominant Emotions</h3>", unsafe_allow_html=True)
                    
                    # Create a row of the top 3 emotions with colored boxes
                    emotion_cols = st.columns(min(3, len(sorted_emotions)))
                    
                    for i, (emotion, score) in enumerate(list(sorted_emotions.items())[:min(3, len(sorted_emotions))]):
                        color = emotion_colors.get(emotion, default_color)
                        # Format score based on its magnitude
                        if score < 0.01:
                            score_display = f"{score:.3f}"
                        elif score < 1:
                            score_display = f"{score:.1%}"
                        else:
                            score_display = f"{score:.2f}"
                            
                        emotion_cols[i].markdown(
                            f"""<div class='emotion-box' style='background-color: {color}; color: {"white" if color != "#FFD700" else "black"};'>
                            {emotion.upper()}<br>{score_display}
                            </div>""",
                            unsafe_allow_html=True
                        )
            
            # Individual tabs for each emotion analysis method
            if "text2emotion" in analysis["all_emotions"]:
                with emotion_tabs[1]:
                    st.markdown("### Text2Emotion Analysis")
                    t2e_result = analysis["all_emotions"]["text2emotion"]
                    
                    if t2e_result:
                        # Create pie chart for Text2Emotion
                        fig, ax = plt.subplots(figsize=(8, 8))
                        labels = list(t2e_result.keys())
                        sizes = list(t2e_result.values())
                        
                        # Only include emotions with non-zero values
                        non_zero_indices = [i for i, size in enumerate(sizes) if size > 0]
                        if non_zero_indices:
                            labels = [labels[i] for i in non_zero_indices]
                            sizes = [sizes[i] for i in non_zero_indices]
                            
                            colors = [emotion_colors.get(label, default_color) for label in labels]
                            
                            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
                                   startangle=90, shadow=True)
                            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
                            st.pyplot(fig)
                            
                            # Display raw scores
                            st.markdown("#### Raw Scores:")
                            for emotion, score in t2e_result.items():
                                st.markdown(f"- **{emotion}**: {score:.4f}")
                        else:
                            st.info("No significant emotions detected by Text2Emotion.")
                    else:
                        st.warning("Text2Emotion analysis not available.")
            
            if "nrclex_raw" in analysis["all_emotions"]:
                with emotion_tabs[2]:
                    st.markdown("### NRCLex Analysis")
                    nrc_raw = analysis["all_emotions"]["nrclex_raw"]
                    nrc_top = analysis["all_emotions"].get("nrclex_top", [])
                    
                    if nrc_raw:
                        # Create radar chart for NRCLex
                        categories = list(nrc_raw.keys())
                        values = list(nrc_raw.values())
                        
                        # Convert to percentages of the total
                        total = sum(values)
                        if total > 0:
                            values_norm = [v/total for v in values]
                            
                            # Create a radar chart
                            fig = plt.figure(figsize=(8, 8))
                            ax = fig.add_subplot(111, polar=True)
                            
                            # Compute angle for each category
                            angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
                            # Make the plot circular
                            values_norm.append(values_norm[0])
                            angles.append(angles[0])
                            categories.append(categories[0])
                            
                            # Plot data
                            ax.plot(angles, values_norm, 'o-', linewidth=2)
                            # Fill area
                            ax.fill(angles, values_norm, alpha=0.25)
                            # Set category labels
                            ax.set_thetagrids(np.degrees(angles[:-1]), categories[:-1])
                            
                            ax.set_title("NRCLex Emotion Profile")
                            ax.grid(True)
                            
                            st.pyplot(fig)
                        
                        # Display top emotions
                        if nrc_top:
                            st.markdown("#### Top Emotions:")
                            for emotion, score in nrc_top:
                                st.markdown(f"- **{emotion}**: {score:.4f}")
                        
                        # Display all raw scores
                        st.markdown("#### All Emotion Scores:")
                        for emotion, score in nrc_raw.items():
                            st.markdown(f"- **{emotion}**: {score}")
                    else:
                        st.warning("NRCLex analysis not available.")
            
            if "vader" in analysis["all_emotions"]:
                with emotion_tabs[3]:
                    st.markdown("### VADER Sentiment Analysis")
                    vader_result = analysis["all_emotions"]["vader"]
                    
                    if vader_result:
                        # Create sentiment gauge
                        compound = vader_result.get("compound", 0)
                        
                        # Create a custom horizontal gauge
                        fig, ax = plt.subplots(figsize=(10, 2))
                        
                        # Set up gauge range (-1 to 1)
                        ax.set_xlim(-1, 1)
                        ax.set_ylim(0, 1)
                        
                        # Create colored regions
                        ax.axvspan(-1, -0.05, alpha=0.3, color='red')
                        ax.axvspan(-0.05, 0.05, alpha=0.3, color='gray')
                        ax.axvspan(0.05, 1, alpha=0.3, color='green')
                        
                        # Plot compound score marker
                        ax.scatter(compound, 0.5, s=300, color='blue', zorder=5)
                        
                        # Add labels
                        ax.text(-0.7, 0.5, "Negative", ha='center', va='center', fontsize=12)
                        ax.text(0, 0.5, "Neutral", ha='center', va='center', fontsize=12)
                        ax.text(0.7, 0.5, "Positive", ha='center', va='center', fontsize=12)
                        
                        # Compound score as text
                        ax.text(compound, 0.8, f"Compound: {compound:.2f}", 
                                ha='center', va='center', fontsize=14, fontweight='bold')
                        
                        # Remove y-axis
                        ax.get_yaxis().set_visible(False)
                        
                        # Custom x-axis
                        ax.set_xticks([-1, -0.5, 0, 0.5, 1])
                        
                        st.pyplot(fig)
                        
                        # Display raw scores
                        st.markdown("#### Raw Sentiment Scores:")
                        st.markdown(f"- **Positive**: {vader_result.get('pos', 0):.4f}")
                        st.markdown(f"- **Neutral**: {vader_result.get('neu', 0):.4f}")
                        st.markdown(f"- **Negative**: {vader_result.get('neg', 0):.4f}")
                        st.markdown(f"- **Compound**: {vader_result.get('compound', 0):.4f}")
                    else:
                        st.warning("VADER sentiment analysis not available.")
            
            if "transformer" in analysis["all_emotions"]:
                with emotion_tabs[4]:
                    st.markdown("### Transformer-Based Analysis")
                    transformer_result = analysis["all_emotions"]["transformer"]
                    
                    if transformer_result:
                        # Sort emotions by score
                        sorted_emotions = dict(sorted(transformer_result.items(), key=lambda x: x[1], reverse=True))
                        
                        # Create horizontal bar chart
                        fig, ax = plt.subplots(figsize=(10, 8))
                        
                        # Get colors
                        emotion_colors = {
                            'joy': '#FFD700',      # Gold
                            'surprise': '#FF8C00', # Dark Orange
                            'neutral': '#A9A9A9',  # Dark Gray
                            'sadness': '#4682B4',  # Steel Blue
                            'fear': '#800080',     # Purple
                            'disgust': '#006400',  # Dark Green
                            'anger': '#B22222',    # Firebrick
                        }
                        default_color = '#1f77b4'  # Default matplotlib blue
                        colors = [emotion_colors.get(k, default_color) for k in sorted_emotions.keys()]
                        
                        # Create bars
                        bars = ax.barh(list(sorted_emotions.keys()), list(sorted_emotions.values()), color=colors)
                        
                        # Add percentage labels
                        for bar in bars:
                            width = bar.get_width()
                            label_x_pos = width + 0.01
                            ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.1%}',
                                    va='center', fontsize=10)
                        
                        ax.set_xlim(0, 1.0)
                        ax.set_title('Transformer Emotion Analysis', fontsize=14)
                        ax.set_xlabel('Confidence Score')
                        fig.tight_layout()
                        
                        st.pyplot(fig)
                        
                        # Display raw scores
                        st.markdown("#### Full Emotion Scores:")
                        for emotion, score in sorted_emotions.items():
                            st.markdown(f"- **{emotion}**: {score:.4f}")
                    else:
                        st.warning("Transformer analysis not available.")
        else:
            # Legacy display for older format
            emotions = analysis.get("emotions", {})
            
            if emotions:
                # Sort emotions by score
                sorted_emotions = dict(sorted(emotions.items(), key=lambda x: x[1], reverse=True))
                
                # Create horizontal bar chart for emotions
                fig, ax = plt.subplots(figsize=(10, 5))
                
                # Define colors for emotions
                emotion_colors = {
                    'joy': '#FFD700',        # Gold
                    'surprise': '#FF8C00',   # Dark Orange
                    'neutral': '#A9A9A9',    # Dark Gray
                    'sadness': '#4682B4',    # Steel Blue
                    'fear': '#800080',       # Purple
                    'disgust': '#006400',    # Dark Green
                    'anger': '#B22222'       # Firebrick
                }
                
                # Get default color for emotions not in the dictionary
                default_color = '#1f77b4'  # Default matplotlib blue
                
                # Create bars with appropriate colors
                bars = ax.barh(
                    list(sorted_emotions.keys()),
                    list(sorted_emotions.values()),
                    color=[emotion_colors.get(emotion, default_color) for emotion in sorted_emotions.keys()]
                )
                
                # Add percentage labels
                for bar in bars:
                    width = bar.get_width()
                    label_x_pos = width + 0.01
                    ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.1%}',
                            va='center', fontsize=10)
                
                ax.set_xlim(0, 1.0)
                ax.set_title('Detected Emotions', fontsize=14)
                ax.set_xlabel('Confidence Score')
                fig.tight_layout()
                
                st.pyplot(fig)
                
                # Top emotions as colored boxes
                st.markdown("<h3>Dominant Emotions</h3>", unsafe_allow_html=True)
                
                # Create a row of the top 3 emotions with colored boxes
                emotion_cols = st.columns(min(3, len(sorted_emotions)))
                
                for i, (emotion, score) in enumerate(list(sorted_emotions.items())[:min(3, len(sorted_emotions))]):
                    color = emotion_colors.get(emotion, default_color)
                    emotion_cols[i].markdown(
                        f"""<div class='emotion-box' style='background-color: {color}; color: {"white" if color != "#FFD700" else "black"};'>
                        {emotion.upper()}<br>{score:.1%}
                        </div>""",
                        unsafe_allow_html=True
                    )
        
        # Display psychoanalytic insights if enabled
        if show_psychoanalysis and 'psychoanalysis' in analysis:
            st.markdown("<h2 class='subheader'>Psychological Insights</h2>", unsafe_allow_html=True)
            
            # Extract key insights
            key_insights = extract_key_insights(analysis["psychoanalysis"])
            
            if key_insights:
                # Create tabs for different views
                insight_tabs = st.tabs(["Key Insights", "By Category", "Raw Data"])
                
                with insight_tabs[0]:
                    # Display top insights across categories
                    st.markdown("### Most Significant Patterns")
                    
                    for i, insight in enumerate(key_insights[:5]):  # Show top 5
                        color_intensity = min(100, int(insight["confidence"] * 100))
                        st.markdown(
                            f"""<div class='insight-box' style='border-left: 5px solid rgba(78, 84, 200, {insight["confidence"]})'>
                            <div><strong>{insight["description"]}</strong></div>
                            <div style='color: #666; font-size: 0.9em;'>Category: {insight["category"]} • Confidence: {insight["confidence"]:.2f}</div>
                            </div>""",
                            unsafe_allow_html=True
                        )
                
                with insight_tabs[1]:
                    # Group insights by category
                    st.markdown("### Insights by Category")
                    
                    # Create expandable sections for each category
                    categories = set(insight["category"] for insight in key_insights)
                    
                    for category in categories:
                        with st.expander(category):
                            category_insights = [i for i in key_insights if i["category"] == category]
                            
                            for insight in category_insights:
                                st.markdown(
                                    f"""<div class='insight-box'>
                                    <div><strong>{insight["description"]}</strong></div>
                                    <div style='color: #666; font-size: 0.9em;'>Confidence: {insight["confidence"]:.2f}</div>
                                    </div>""",
                                    unsafe_allow_html=True
                                )
                
                with insight_tabs[2]:
                    # Show raw JSON data
                    st.markdown("### Raw Analysis Data")
                    st.json(analysis["psychoanalysis"])
            else:
                st.info("No significant psychological insights detected.")
        
        # Display about user if detailed analysis is enabled
        if show_analysis and 'about_user' in analysis:
            st.markdown("<h2 class='subheader'>About the User</h2>", unsafe_allow_html=True)
            
            about_user = analysis["about_user"]
            
            # Create tabs for different types of observations
            if about_user:
                about_tabs = st.tabs(["Observations", "Raw Data"])
                
                with about_tabs[0]:
                    # Display certain observations
                    if "certain" in about_user and about_user["certain"]:
                        st.markdown("### Confident Observations")
                        for observation in about_user["certain"]:
                            st.markdown(
                                f"""<div class='insight-box' style='border-left: 5px solid #32CD32'>
                                <div><strong>{observation}</strong></div>
                                </div>""",
                                unsafe_allow_html=True
                            )
                    
                    # Display uncertain observations
                    if "unsure" in about_user and about_user["unsure"]:
                        st.markdown("### Potential Observations")
                        for observation, scores in about_user["unsure"].items():
                            confidence = scores.get("short_term", 0)
                            st.markdown(
                                f"""<div class='insight-box' style='border-left: 5px solid rgba(255, 140, 0, {confidence})'>
                                <div><strong>{observation}</strong></div>
                                <div style='color: #666; font-size: 0.9em;'>Confidence: {confidence:.2f}</div>
                                </div>""",
                                unsafe_allow_html=True
                            )
                
                with about_tabs[1]:
                    # Show raw data
                    st.json(about_user)
            else:
                st.info("No user observations available.")
    else:
        st.markdown("<h2 class='subheader'>Emotion Analysis</h2>", unsafe_allow_html=True)
        st.markdown(
            """
            ### How It Works
            
            1. **Speak or Type**: Click the record button to speak your thoughts or type them in the text area
            2. **Real-time Analysis**: Your input is analyzed using multiple emotion detection methods
            3. **Visual Results**: See detailed visualizations of your emotional state
            4. **AI Response**: Receive a supportive response based on your emotions
            
            ### Try These Sample Prompts:
            
            - "I'm feeling really excited about my new job opportunity!"
            - "I'm worried about my upcoming presentation at work."
            - "I had a disagreement with my friend and now I feel bad."
            - "Today was such a beautiful day, I felt so peaceful during my walk."
            """
        )
        
        # Sample visualization placeholder
        st.markdown("""
        <div style="text-align: center; padding: 20px; background-color: #f5f5f5; border-radius: 10px;">
            <p style="font-size: 1.2em;">🎙️ Record or type your message to see emotional analysis</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center;'>
    <p>This application uses OpenAI Whisper for speech recognition and multiple emotion analysis models.</p>
    <p>Remember that AI suggestions are not a substitute for professional mental health support.</p>
    </div>
    """, 
    unsafe_allow_html=True
)

# Add a help expander
with st.expander("Help & Information"):
    st.markdown("""
    ### About This Application
    
    This Speech Emotion AI Therapist combines voice recognition with multiple emotion analysis methods to detect emotions in your speech or text and provides personalized responses.
    
    ### Speech Recognition
    
    The application uses OpenAI's Whisper model to accurately transcribe your spoken words into text. To use this feature:
    
    1. Click the "Click to Record" button
    2. Speak clearly into your microphone
    3. Click the button again to stop recording
    4. Wait a moment for transcription and analysis
    
    ### Emotion Analysis Methods
    
    - **Text2Emotion**: Analyzes basic emotions (Joy, Fear, Anger, Sadness, Surprise)
    - **NRCLex**: Uses a lexicon-based approach to identify nuanced emotions
    - **VADER**: Focuses on sentiment (Positive, Negative, Neutral)
    - **Transformer**: Uses an advanced neural network for more accurate emotion detection
    
    ### Privacy Note
    
    Your conversations and audio data are processed for the current session only and are not stored permanently.
    """)

# Add a troubleshooting expander
with st.expander("Troubleshooting"):
    st.markdown("""
    ### Common Issues
    
    **Microphone Access**
    - Make sure your browser has permission to access your microphone
    - Try using Chrome or Edge for best compatibility with the audio recording feature
    - If you see "Error accessing microphone" message, check your browser settings
    
    **Speech Recognition Issues**
    - Speak clearly and at a moderate pace
    - Minimize background noise for better results
    - If transcription fails, try typing your message instead
    
    **Installation Requirements**
    To run this application locally, you need to install:
    ```
    pip install streamlit openai matplotlib nltk transformers text2emotion nrclex python-dotenv
    ```
    """)

# Additional information at the bottom
st.markdown("""
<div style='text-align: center; margin-top: 30px; font-size: 0.8em; color: #666;'>
Speech Emotion AI Therapist • Powered by OpenAI Whisper and Multiple Analysis Models
</div>
""", unsafe_allow_html=True)