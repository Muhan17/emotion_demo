import streamlit as st
st.set_page_config(page_title="Emotion Recognition", layout="centered")  # ✅ Обязательно первым

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageOps
import gdown
import os

# === Настройки ===
MODEL_PATH = "best_emotion_model_gray48.pth"
FILE_ID = "1h6OZWxlWDr_IDzlb4LucQzWV56kSqgza"  # 👈 Твой Google Drive файл ID
DOWNLOAD_URL = f"https://drive.google.com/uc?id={FILE_ID}"

class_names = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']
positive_emotions = ['happy', 'neutral', 'surprise']

# === Загрузка модели с Google Drive ===
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("⬇️ Загружаем модель с Google Drive..."):
            gdown.download(DOWNLOAD_URL, MODEL_PATH, quiet=False)

    model = models.resnet18()
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, 6)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model

model = load_model()

# === Преобразование изображения ===
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# === Интерфейс ===
st.title("🎭 Emotion Recognition (Grayscale ResNet18)")

uploaded_file = st.file_uploader("📤 Загрузите изображение (лицо)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Загруженное изображение", use_column_width=True)

    gray = ImageOps.grayscale(image)
    input_tensor = transform(gray).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        _, pred = torch.max(output, 1)
        predicted_emotion = class_names[pred.item()]

    st.markdown(f"### 🧠 Эмоция: **{predicted_emotion.upper()}**")
    if predicted_emotion in positive_emotions:
        st.success("✅ Клиент удовлетворён")
    else:
        st.error("❌ Клиент не удовлетворён")
