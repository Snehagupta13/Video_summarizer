import os
from pdf2image import convert_from_path
from PIL import Image

def pdf_to_images(pdf_path, output_folder):
    images = convert_from_path(pdf_path)
    os.makedirs(output_folder, exist_ok=True)
    paths = []
    for i, img in enumerate(images):
        path = os.path.join(output_folder, f"page_{i+1}.png")
        img.save(path)
        paths.append(path)
    return paths