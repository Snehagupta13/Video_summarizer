# donut_infer_and_save.py

import os
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel

# ------------------ Load Donut Model and Processor ------------------
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
model.eval()

# ------------------ Dataset Paths ------------------
base_folder = "dataset/batch_2/batch_2"
subfolders = ["batch2_1", "batch2_2", "batch2_3"]

# ------------------ Inference Loop ------------------
results = []

for subfolder in subfolders:
    folder_path = os.path.join(base_folder, subfolder)
    for img_file in sorted(os.listdir(folder_path)):
        if img_file.lower().endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(folder_path, img_file)
            image = Image.open(img_path).convert("RGB")

            # Prepare input using DonutProcessor
            inputs = processor(images=image, return_tensors="pt")
            prompt = "<s_docvqa><question>What is the invoice number, invoice date, and line items?</question><answer>"
            decoder_input_ids = processor.tokenizer(
                prompt,
                add_special_tokens=False,
                return_tensors="pt"
            ).input_ids

            # Generate output
            outputs = model.generate(
                pixel_values=inputs.pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=512
            )
            decoded = processor.batch_decode(outputs, skip_special_tokens=True)[0]

            results.append({
                "file": img_file,
                "subfolder": subfolder,
                "output": decoded
            })

            print(f"\n📄 {img_file} in {subfolder} → \n{decoded}")
            print("-" * 80)

# ------------------ Save Model Locally ------------------
model_dir = "saved_model/donut_invoice"
model.save_pretrained(model_dir)
processor.save_pretrained(model_dir)
print(f"\n✅ Model and processor saved to '{model_dir}'")