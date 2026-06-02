
# --------------------------- extractor.py ---------------------------
import pytesseract
import re
from PIL import Image

def extract_key_values(image_path):
    text = pytesseract.image_to_string(Image.open(image_path))
    invoice_number = re.search(r"Invoice\s*No\.?\s*[:\-]?\s*(\w+)", text, re.IGNORECASE)
    invoice_date = re.search(r"Date\s*[:\-]?\s*([\d\-/]+)", text, re.IGNORECASE)

    return {
        "invoice_number": invoice_number.group(1) if invoice_number else None,
        "invoice_date": invoice_date.group(1) if invoice_date else None,
        "raw_text": text
    }