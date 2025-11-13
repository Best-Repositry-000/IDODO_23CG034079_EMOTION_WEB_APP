import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image


class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc1 = nn.Linear(64 * 11 * 11, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.fc1(x))
        x = self.fc2(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmotionCNN(num_classes=7)
model.load_state_dict(torch.load("emotion_model.pth", map_location=device))
model.eval()


transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# === Predict Emotion from Image ===
def predict_emotion(image_path):
    img = Image.open(image_path)
    img = transform(img).unsqueeze(0)  # Add batch dimension
    outputs = model(img)
    _, predicted = torch.max(outputs, 1)
    emotions = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]    # adjust to your dataset
    print(f"Predicted Emotion: {emotions[predicted.item()]}")


test_image = "PrivateTest_1735299.jpg"  # replace with your own image path
predict_emotion(test_image)
