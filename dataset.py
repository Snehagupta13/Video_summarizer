# --------------------------- dataset.py ---------------------------
import os
from torch.utils.data import Dataset
from PIL import Image

class InvoiceDataset(Dataset):
    def __init__(self, image_dir, labels, transform=None):
        self.image_dir = image_dir
        self.labels = labels
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(img_path).convert("RGB")
        label = self.labels.get(self.images[idx], {})
        if self.transform:
            image = self.transform(image)
        return image, label