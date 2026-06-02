import streamlit as st
import os
import tempfile
from state import VideoState
from edges import app

def validate_video_file(video_path: str) -> None:
    """Validate input video file"""
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"File not found: {video_path}")
    if not video_path.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        raise ValueError("Unsupported video format")

def run_pipeline(video_path: str):
    validate_video_file(video_path)
    initial_state = VideoState(input_video=video_path)
    st.info("🚀 Starting full pipeline... Please wait.")
    
    final_output = app.invoke(initial_state)
    validated = VideoState(**final_output)
    
    st.success("✅ Processing completed!")
    st.json(validated.model_dump(), expanded=False)

    if validated.output_files:
        st.markdown("### 📂 Output Files")
        for t, path in validated.output_files.items():
            st.markdown(f"- **{t}**: [Download]({path})")

# Streamlit UI
st.set_page_config(page_title="Video Pipeline", page_icon="🎥")

st.title("🎬 Video Processing App")
st.markdown("Upload a video file to run the full processing pipeline.")

uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_video_path = temp_file.name
        st.video(temp_video_path)

        if st.button("Run Pipeline"):
            run_pipeline(temp_video_path)
