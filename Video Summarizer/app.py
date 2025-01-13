# Import necessary libraries
import streamlit as st  # For building the web application
from phi.agent import Agent  # To initialize and use the AI agent
from phi.model.google import Gemini  # For using Gemini AI model
from phi.tools.duckduckgo import DuckDuckGo  # For supplementary web research
from google.generativeai import upload_file, get_file  # For Google Generative AI file handling
import google.generativeai as genai  # Google Generative AI library

import time  # To handle delays and wait for processes
from pathlib import Path  # For file path management
import tempfile  # To handle temporary file creation

from dotenv import load_dotenv  # For loading environment variables
load_dotenv()  # Load the .env file to access API keys and other settings

import os  # For environment variable management

# Fetch API Key from environment variables
API_KEY = os.getenv("GOOGLE_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)  # Configure the Generative AI client

# Streamlit page configuration
st.set_page_config(
    page_title="Multimodal AI Agent- Video Summarizer",  # Title of the web app
    page_icon="🎥",  # Page icon
    layout="wide"  # Use wide layout for better visualization
)

# Display app title and header
st.title("Phidata Video AI Summarizer Agent 🎥🎤🖬")
st.header("Powered by Gemini 2.0 Flash Exp")  # Highlight the AI technology used

# Cache the initialization of the AI agent for better performance
@st.cache_resource
def initialize_agent():
    return Agent(
        name="Video AI Summarizer",  # Agent's name
        model=Gemini(id="gemini-2.0-flash-exp"),  # Use the Gemini AI model
        tools=[DuckDuckGo()],  # Include DuckDuckGo tool for web research
        markdown=True,  # Support markdown in the agent's responses
    )

# Initialize the AI agent
multimodal_Agent = initialize_agent()

# File uploader for users to upload a video file
video_file = st.file_uploader(
    "Upload a video file", type=['mp4', 'mov', 'avi'], help="Upload a video for AI analysis"
)

# Check if a video file has been uploaded
if video_file:
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
        temp_video.write(video_file.read())  # Write the uploaded video content
        video_path = temp_video.name  # Store the temporary file path

    # Display the uploaded video on the app
    st.video(video_path, format="video/mp4", start_time=0)

    # Input area for user queries
    user_query = st.text_area(
        "What insights are you seeking from the video?",
        placeholder="Ask anything about the video content. The AI agent will analyze and gather additional context if needed.",
        help="Provide specific questions or insights you want from the video."
    )

    # Analyze video when the button is clicked
    if st.button("🔍 Analyze Video", key="analyze_video_button"):
        # Check if user has provided a query
        if not user_query:
            st.warning("Please enter a question or insight to analyze the video.")
        else:
            try:
                # Show a loading spinner during the processing
                with st.spinner("Processing video and gathering insights..."):
                    # Upload the video to Google Generative AI service
                    processed_video = upload_file(video_path)
                    
                    # Wait until the video processing is complete
                    while processed_video.state.name == "PROCESSING":
                        time.sleep(1)  # Wait for 1 second
                        processed_video = get_file(processed_video.name)  # Get the latest status of the video

                    # Create a prompt for the AI agent to analyze the video
                    analysis_prompt = (
                        f"""
                        Analyze the uploaded video for content and context.
                        Respond to the following query using video insights and supplementary web research:
                        {user_query}

                        Provide a detailed, user-friendly, and actionable response.
                        """
                    )

                    # Use the AI agent to process the query and video
                    response = multimodal_Agent.run(analysis_prompt, videos=[processed_video])

                # Display the AI's response
                st.subheader("Analysis Result")
                st.markdown(response.content)

            except Exception as error:
                # Handle errors during analysis
                st.error(f"An error occurred during analysis: {error}")
            finally:
                # Delete the temporary video file after processing
                Path(video_path).unlink(missing_ok=True)
else:
    # Inform the user to upload a video to begin analysis
    st.info("Upload a video file to begin analysis.")

# Customize the height of the text area for better user experience
st.markdown(
    """
    <style>
    .stTextArea textarea {
        height: 100px;
    }
    </style>
    """,
    unsafe_allow_html=True
)