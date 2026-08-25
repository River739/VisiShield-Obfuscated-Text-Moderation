# Clear old dataset so there is no contamination
os.makedirs("dataset/safe", exist_ok=True)
os.makedirs("dataset/banned", exist_ok=True)

# Robust word databases
ROBUST_SAFE = ["banana", "computer", "mountain", "keyboard", "sunlight",
               "elephant", "guitar", "industry", "captain", "notebook",
               "nature", "laptop", "awesome", "project", "moderator",
               "luck", "duck", "itch", "ship", "hello"]

ROBUST_BANNED = ["bitch", "b!tch", "b1tch", "sh1t", "shit", "$hit",
                 "fuck", "f*ck", "fuxk", "f_u_c_k", "b!+ch", "sh*t"]

def generate_robust_image(text, font_path, width=192, height=64):
    img = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(img)

    # Apply random font size
    font_size = random.randint(18, 26)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        font = ImageFont.load_default()

    # Introduce spatial jittering (helps model ignore exact pixel coordinates)
    x_offset = random.randint(5, 30)
    y_offset = random.randint(10, 22)

    draw.text((x_offset, y_offset), text, fill=0, font=font)
    img_np = np.array(img)

    # Introduce blur variance (helps model adapt to low/high resolutions)
    blur_val = random.choice([5, 7, 9])
    blurred_img = cv2.GaussianBlur(img_np, (blur_val, blur_val), 0)

    return blurred_img

# Collect downloaded ttf files
font_paths = [os.path.join("fonts", f) for f in os.listdir("fonts") if f.endswith('.ttf')]

# Generate 1,000 samples per class
samples_per_class = 1000
print(f"Generating {samples_per_class * 2} synthetic images...")

for i in range(samples_per_class):
    # Select a random font for variation
    selected_font = random.choice(font_paths) if font_paths else "default"

    # Safe word generation
    word_s = random.choice(ROBUST_SAFE)
    img_s = generate_robust_image(word_s, selected_font)
    cv2.imwrite(f"dataset/safe/safe_{i}.png", img_s)

    # Banned word generation
    word_b = random.choice(ROBUST_BANNED)
    img_b = generate_robust_image(word_b, selected_font)
    cv2.imwrite(f"dataset/banned/banned_{i}.png", img_b)

print("Robust dataset successfully saved to Google Colab local storage!")

import os
import glob
import random
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

class BlurredTextDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        # Fixed: Explicitly using glob and os here now
        self.safe_images = glob.glob(os.path.join(root_dir, "safe", "*.png"))
        self.banned_images = glob.glob(os.path.join(root_dir, "banned", "*.png"))

        # Safe = 0.0, Banned = 1.0
        self.all_images = [(path, 0.0) for path in self.safe_images] + [(path, 1.0) for path in self.banned_images]
        random.shuffle(self.all_images)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)) # Normalize pixel intensity
        ])

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        img_path, label = self.all_images[idx]
        image = Image.open(img_path).convert("L")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)

# Instantiate and Split (80% Train, 20% Test)
full_dataset = BlurredTextDataset("dataset")
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Split completed: {len(train_dataset)} train samples, {len(test_dataset)} test samples.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on accelerator: {device}")

model = SilhouetteNet().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10
print("Starting training on robust dataset...")
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        predictions = (torch.sigmoid(outputs) > 0.5).float()
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = (correct / total) * 100
    print(f"Epoch [{epoch+1}/{epochs}] | Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}%")

print("Robust model successfully trained!")