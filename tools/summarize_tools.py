import os
import sys
import cv2
import glob
from datetime import datetime
from typing import Dict, Any
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from state import VideoState
from .blip_tools import generate_captions
from .easy_ocr import extract_text_from_frames
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

load_dotenv()

def save_to_file(content: str, video_path: str, file_type: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"{base_name}_{file_type}_{timestamp}.txt")
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"[✅] Saved to: {filename}")
    return filename


def format_captions_with_llm(captions: str) -> str:
    """Format raw video captions into a coherent summary using LLM."""
    try:
        # Validate API key
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")

        # Initialize LLM with more conservative parameters for summarization
        llm = ChatGroq(
            temperature=0.4,  # Lower temperature for more factual output
            api_key=api_key,
            model_name="llama3-8b-8192",
            max_tokens=1024  # Explicit token limit
        )

        # Improved prompt template
        prompt_template = """You are a professional video analysis assistant. Your task is to:
1. Analyze these video frame captions
2. Identify the most important information
3. Create a coherent narrative summary

Guidelines:
- Focus on people, vehicles, actions, and text that appears important
- Ignore timestamps, frame numbers, and technical metadata
- Combine similar captions to avoid repetition
- Maintain chronological order of events
- Be concise but descriptive

Raw Captions:
{captions}

Processed Summary:"""
        
        # Truncate captions to fit model context window
        truncated_captions = captions[:4000]  # More conservative truncation
        
        # Create and invoke the chain
        response = llm.invoke(prompt_template.format(captions=truncated_captions))
        
        # Clean and validate the response
        summary = response.content.strip()
        if not summary or len(summary) < 20:  # Minimum length check
            raise ValueError("LLM returned an empty or too-short summary")
            
        return summary

    except Exception as e:
        print(f"[❌] LLM Formatting Error: {str(e)}")
        return "Summary unavailable (processing error)"


def generate_caption_summary(state: VideoState, video_path: str) -> Dict[str, Any]:
    try:
        if not state.captions:
            if not state.frames:
                raise ValueError("No frames to caption")
            state.captions = generate_captions(state.frames)

        if not state.extracted_text:
            state.extracted_text, _ = extract_text_from_frames()

        raw_caption_text = "\n".join([f"{os.path.basename(f)}: {c}" for f, c in state.captions])
        formatted_captions = format_captions_with_llm(raw_caption_text)
        extracted_text = (state.extracted_text or "").strip()

        final_text = (
            "=== RAW BLIP CAPTIONS ===\n"
            f"{raw_caption_text}\n\n"
            "=== LLM-FORMATTED CAPTIONS ===\n"
            f"{formatted_captions}\n\n"
            "=== RAW OCR TEXT (EasyOCR) ===\n"
            f"{extracted_text}"
        )

        filename = save_to_file(final_text, video_path, "llm_summary_full")

        return {
            "raw_captions": raw_caption_text,
            "formatted_captions": formatted_captions,
            "extracted_text": extracted_text,
            "output_file": filename,
            "output_files": {"captions_and_text": filename} if filename else None
        }

    except Exception as e:
        print(f"[❌] Failed to generate formatted summary: {e}")
        return {
            "raw_captions": "",
            "formatted_captions": "",
            "extracted_text": "",
            "output_file": None,
            "output_files": None
        }
