from tools.blip_tools import convert_video, extract_frames, generate_captions
from tools.summarize_tools import generate_caption_summary  
from tools.easy_ocr import extract_text_from_frames
from tools.rag import create_vectorstore_from_text, save_vectorstore, build_qa_chain, run_chatbot_from_file
from state import VideoState
from typing import Dict, Any
import os
from datetime import datetime
import glob

def process_video(state: VideoState) -> dict:
    """Process video and extract frames (returns only modified fields)"""
    try:
        converted_vid = convert_video(state.input_video)
        return {
            "converted_video": converted_vid,
            "frames": extract_frames(converted_vid)
        }
    except Exception as e:
        print(f"Video processing error: {e}")
        return {"converted_video": None, "frames": []}

def process_captions(state: VideoState) -> dict:
    try:
        if not state.frames:
            raise ValueError("No frames available")
        
        captions = generate_captions(state.frames)
        if not captions:
            raise ValueError("No captions generated")
        
        return {"captions": captions}

    except Exception as e:
        print(f"[❌] Captioning error: {e}")
        return {"captions": []}

def process_easyocr(state: VideoState) -> dict:
    """Extract OCR text from frame folder."""
    try:
        text, _ = extract_text_from_frames()
        return {"extracted_text": text}
    except Exception as e:
        print(f"OCR error: {e}")
        return {"extracted_text": "Extraction failed"}

def summarize_captions(state: VideoState) -> VideoState:
    try:
        if not state.input_video:
            raise ValueError("Missing input video path")

        if not state.captions and not state.extracted_text:
            raise ValueError("No captions or extracted text available")

        result = generate_caption_summary(state, state.input_video)

        if not result.get("output_file"):
            state.final_summary = "Partial summary - some components failed"
        else:
            state.final_summary = "Formatted captions and raw OCR text saved."

        state.output_files = result.get("output_files", {})
        state.captions = state.captions or result.get("raw_captions", [])
        state.extracted_text = state.extracted_text or result.get("extracted_text", "")
        state.scene_summary = result.get("formatted_captions", "No summary available")
        state.text_summary = result.get("extracted_text", "No text extracted")

        return state
    except Exception as e:
        print(f"[❌] Summarize captions node error: {e}")
        state.output_files = {}
        state.scene_summary = f"Summary failed: {str(e)}"
        state.text_summary = "Text extraction failed"
        state.final_summary = "Pipeline completed with errors"
        return state

def get_latest_summary(output_dir):
    """Get most recent summary file from outputs directory"""
    files = glob.glob(os.path.join(output_dir, "*llm_summary_full*.txt"))
    if not files:
        raise FileNotFoundError(f"No summary files found in {output_dir}")
    return max(files, key=os.path.getmtime)

def run_chatbot_node(state: VideoState) -> VideoState:
    """Initialize and automatically launch chatbot"""
    try:
        # Validate preconditions
        if not state.scene_summary or len(state.scene_summary.strip()) < 20:
            raise ValueError("Scene summary is empty or too short")
        
        # Prepare paths
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(base_dir, "outputs")
        store_path = os.path.join(base_dir, "faiss_store")
        
        # Ensure directories exist
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(store_path, exist_ok=True)

        # Verify GROQ API key
        if not os.getenv("GROQ_API_KEY"):
            raise EnvironmentError("GROQ_API_KEY not set")

        # Create and verify vector store
        vs = create_vectorstore_from_text(state.scene_summary)
        if vs is None:
            raise RuntimeError("Vector store creation failed")
        if len(vs.docstore._dict) == 0:
            raise ValueError("Empty vector store created")

        # Save artifacts
        save_vectorstore(vs, path=store_path)
        
        # Test QA chain
        qa_chain = build_qa_chain(vs)
        if qa_chain is None:
            raise RuntimeError("QA chain creation failed")

        # Update state
        state.final_summary = f"{state.final_summary}\nChatbot initialized successfully"
        state.output_files.update({
            "chatbot_index": store_path,
            "chatbot_ready_flag": os.path.join(output_dir, "chatbot.ready")
        })

        # Automatically launch chatbot with latest summary
        latest_summary = get_latest_summary(output_dir)
        print("\n" + "="*50)
        print("🚀 Launching chatbot with latest summary...")
        print("="*50 + "\n")
        run_chatbot_from_file(latest_summary)
        
        return state

    except Exception as e:
        error_msg = f"\nChatbot failed: {str(e)}"
        print(f"[ERROR] {error_msg}")
        state.final_summary = f"{state.final_summary or ''}{error_msg}"
        
        # Log error
        error_log_path = os.path.join(output_dir, "chatbot_error.log")
        with open(error_log_path, 'a') as f:
            f.write(f"{datetime.now()}: {str(e)}\n")
        
        return state