import streamlit as st
from dotenv import load_dotenv
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import google.generativeai as genai
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("API key not found. Please set the GOOGLE_API_KEY environment variable.")
    st.stop()

genai.configure(api_key=api_key)

# Load Hugging Face translation model
@st.cache_resource
def load_translation_model():
    model_name = "Helsinki-NLP/opus-mt-hi-en"  # Hindi to English
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading translation model: {e}")
        return None, None

translator_tokenizer, translator_model = load_translation_model()

# Prompt for summarization
prompt = """You are a YouTube video summarizer. 
You will be taking the transcript text and summarizing the entire video and providing the important summary in point 
wise within 1000 words. Please provide the summary of the text given here: 
Points to Remember:-
- Give the summary in points.
- Highlights the important points.
"""

# Getting the transcript data from YouTube videos
def extract_transcript_details(youtube_video_url, language='en'):
    try:
        video_id = youtube_video_url.split("v=")[1].split("&")[0]  # Improved extraction
        transcript_data = YouTubeTranscriptApi.get_transcript(video_id, languages=[language])
        
        # Concatenate transcript text
        transcript = " ".join([i["text"] for i in transcript_data])
        return transcript
        
    except NoTranscriptFound:
        st.error(f"No transcript found for the video in the requested language: {language}.")
        return None
    except TranscriptsDisabled:
        st.error("Transcripts are disabled for this video.")
        return None
    except Exception as e:
        st.error(f"Error fetching transcript: {e}")
        return None

# Translate transcript text to English using Hugging Face
def translate_to_english_hf(text):
    try:
        inputs = translator_tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True)
        outputs = translator_model.generate(inputs, max_length=512, num_beams=4, early_stopping=True)
        translated_text = translator_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translated_text
    except Exception as e:
        st.error(f"Error translating text: {e}")
        return None

# Getting the summary based on prompt from Google Gemini
def generate_gemini_content(transcript_text, prompt):
    try:
        model = genai.GenerativeModel("gemini-2.0-flash-001")
        response = model.generate_content(prompt + transcript_text)
        return response.text
    except Exception as e:
        st.error(f"Error generating summary: {e}")
        return None

# Streamlit UI
st.title("YouTube Linked please: ")
youtube_link = st.text_input("Video Link:")

if youtube_link:
    video_id = youtube_link.split("v=")[1].split("&")[0] if "v=" in youtube_link else None
    if video_id:
        st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg")  # Fixed image rendering
    else:
        st.error("Invalid YouTube link. Please enter a valid link.")

if st.button("Get summaries Notes"):
    with st.spinner("Fetching transcript, translating, and generating summary..."):
        # Attempt to fetch transcript in English first
        transcript_text = extract_transcript_details(youtube_link, language='en')
        
        # If no transcript found in English, try Hindi
        if not transcript_text:
            transcript_text = extract_transcript_details(youtube_link, language='hi')
        
        if transcript_text:
            # Translate transcript to English if not already in English
            translated_text = translate_to_english_hf(transcript_text) if not transcript_text.isascii() else transcript_text
            if translated_text:
                summary = generate_gemini_content(translated_text, prompt)
                if summary:
                    st.markdown("## Detailed Notes:")
                    st.write(summary)
