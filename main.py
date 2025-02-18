import torch
from modules.FacialExpressionRecognitionModel import FacialExpressionRecognitionModel
from modules.FER2013Dataset import FER2013Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn, optim
import torchvision.datasets as datasets
import joblib

num_classes = 7  # FER2013 typically has 7 emotions
batch_size = 32
learning_rate = 0.0001
num_epochs = 10
dataset_root = "./data"
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
    transforms.Resize((64, 64)),  # Resize images to 64x64
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
])
train_dataset = datasets.ImageFolder(root=f"{dataset_root}/train", transform=transform)
test_dataset = datasets.ImageFolder(root=f"{dataset_root}/test", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        if batch_idx % 50 == 0:
            print(f"Epoch {epoch} [{batch_idx * len(images)}/{len(train_loader.dataset)}] Loss: {loss.item():.4f}")
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch} Training Loss: {epoch_loss:.4f}")

def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
    
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")


def main():
    model = FacialExpressionRecognitionModel(
        num_classes=num_classes,
        emb_dim=128,
        patch_size=4,
        transformer_depth=4,
        n_heads=8,
        mlp_dim=256
    ) 
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(1, num_epochs + 1):
        train(model, device, train_loader, optimizer, criterion, epoch)
        evaluate(model, test_loader)

    print("Finished Training")
    model_filename = "VTFF.pkl"
    joblib.dump(model, model_filename)
    print(f"💾 Model saved as {model_filename}")

if __name__ == "__main__":
    main()