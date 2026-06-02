import os
import easyocr
import re
from datetime import datetime

# Save to root-level /outputs
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

def extract_text_from_frames(folder_path='frames') -> tuple[str, int]:
    try:
        reader = easyocr.Reader(['en'])  # English only for cleaner results
        
        # Enhanced filtering rules
        NOISE_PATTERNS = [
            r'\$?\d+\.?\d*',  # Monetary values
            r'www\.?\S+',     # Websites
            r'copyright',     # Copyright notices
            r'inform your manager',  # Repeated phrase
            r'[^a-zA-Z0-9\s]{3,}',  # Excessive special chars
            r'^\W*$',         # Non-word characters only
            r'\b\w{1,2}\b',   # Single/double letters
        ]
        
        image_files = [f for f in os.listdir(folder_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        unique_texts = set()
        
        for img_file in image_files:
            results = reader.readtext(
                os.path.join(folder_path, img_file),
                decoder='beamsearch',  # Better for structured text
                width_ths=0.5,        # Merge nearby boxes
                height_ths=0.5,
                min_size=20,          # Ignore small text
                text_threshold=0.7    # Higher confidence
            )
            
            for (_, text, confidence) in results:
                if confidence < 0.7:
                    continue
                    
                text = text.strip()
                if not any(re.search(pattern, text, re.IGNORECASE) 
                   for pattern in NOISE_PATTERNS):
                    unique_texts.add(text)
        
        return "\n".join(unique_texts), len(image_files)
        
    except Exception as e:
        print(f"[OCR ERROR] {str(e)}")
        return "", 0

