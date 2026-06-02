

# --------------------------- model/trainer.py ---------------------------
import torch
from torch.utils.data import DataLoader

def train_model(model, dataset, optimizer, criterion, epochs=10):
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in dataloader:
            # Dummy forward pass
            outputs = model(torch.randn(len(images), 256))
            targets = torch.randint(0, 2, (len(images), 2)).float()
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")