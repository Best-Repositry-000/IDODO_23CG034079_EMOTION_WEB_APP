import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


transform = transforms.Compose([
    transforms.Grayscale(),             # convert to grayscale
    transforms.Resize((48, 48)),        # resize all images
    transforms.ToTensor(),              # convert to tensor
    transforms.Normalize((0.5,), (0.5,)) # normalize pixels to [-1, 1]
])


train_dataset = datasets.ImageFolder(root="archive/train", transform=transform)
test_dataset = datasets.ImageFolder(root="archive/test", transform=transform)


train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


print(f"Training images: {len(train_dataset)}")
print(f"Test images: {len(test_dataset)}")
print(f"Classes: {train_dataset.classes}")