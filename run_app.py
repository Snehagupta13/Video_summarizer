import streamlit as st
import os
import tempfile
from state import VideoState
from nodes import process_video, process_captions, process_easyocr, summarize_captions
from tools.rag import create_vectorstore_from_text, build_qa_chain

def validate_video_file(video_path: str) -> None:
    """Validate input video file"""
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"File not found: {video_path}")
    if not video_path.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        raise ValueError("Unsupported video format")

def run_pipeline(video_path: str) -> VideoState:
    """Run the summarization pipeline (skips the terminal-based chatbot node,
    which would otherwise block the Streamlit process on an input() loop)."""
    validate_video_file(video_path)
    state = VideoState(input_video=video_path)

    with st.spinner("Extracting frames..."):
        state = state.model_copy(update=process_video(state))
    with st.spinner("Captioning scenes..."):
        state = state.model_copy(update=process_captions(state))
    with st.spinner("Reading on-screen text (OCR)..."):
        state = state.model_copy(update=process_easyocr(state))
    with st.spinner("Summarizing with LLM..."):
        state = summarize_captions(state)

    return VideoState(**state.model_dump())

# Streamlit UI
st.set_page_config(page_title="Video Pipeline", page_icon="🎥")

st.title("🎬 Video Processing App")
st.markdown("Upload a video file to run the full processing pipeline.")

for key, default in [
    ("video_state", None),
    ("qa_chain", None),
    ("chat_history", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default

uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_video_path = temp_file.name
        st.video(temp_video_path)

        if st.button("Run Pipeline"):
            try:
                st.info("🚀 Starting full pipeline... Please wait.")
                validated = run_pipeline(temp_video_path)
                st.success("✅ Processing completed!")

                st.session_state.video_state = validated
                st.session_state.chat_history = []

                if validated.scene_summary and len(validated.scene_summary.strip()) >= 20:
                    with st.spinner("Building Q&A index for this video..."):
                        vs = create_vectorstore_from_text(validated.scene_summary)
                        st.session_state.qa_chain = build_qa_chain(vs)
                else:
                    st.session_state.qa_chain = None
                    st.warning("Scene summary too short — Q&A chat unavailable for this video.")
            except Exception as e:
                st.error(f"Pipeline failed: {e}")

if st.session_state.video_state:
    validated = st.session_state.video_state
    st.json(validated.model_dump(), expanded=False)

    if validated.output_files:
        st.markdown("### 📂 Output Files")
        for t, path in validated.output_files.items():
            st.markdown(f"- **{t}**: [Download]({path})")

if st.session_state.qa_chain:
    st.markdown("### 💬 Ask about the video")

    for question, answer in st.session_state.chat_history:
        st.chat_message("user").write(question)
        st.chat_message("assistant").write(answer)

    question = st.chat_input("Ask a question about the video...")
    if question:
        st.chat_message("user").write(question)
        with st.spinner("Thinking..."):
            try:
                result = st.session_state.qa_chain.invoke({"query": question})
                answer = result["result"]
            except Exception as e:
                answer = f"⚠️ Error answering question: {e}"
        st.chat_message("assistant").write(answer)
        st.session_state.chat_history.append((question, answer))
