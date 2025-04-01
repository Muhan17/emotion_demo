import streamlit as st
st.set_page_config(page_title="Real-time Emotion Recognition", layout="centered")

import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms, models
from PIL import Image
from collections import Counter
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import gdown
import os
import matplotlib.pyplot as plt

# ==== Константы ====
MODEL_PATH = "best_emotion_model_gray48_1.pth"
FILE_ID = "1h6OZWxlWDr_IDzlb4LucQzWV56kSqgza"
DOWNLOAD_URL = f"https://drive.google.com/uc?id={FILE_ID}"
class_names = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']
positive_emotions = ['happy', 'neutral', 'surprise']
emotion_log = []

# ==== Загрузка модели ====
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

# ==== Преобразование ====
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ==== Обработчик видео ====
class EmotionProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            face_pil = Image.fromarray(face_img).convert("L")
            input_tensor = transform(face_pil).unsqueeze(0)

            with torch.no_grad():
                output = model(input_tensor)
                _, pred = torch.max(output, 1)
                predicted_emotion = class_names[pred.item()]
                emotion_log.append(predicted_emotion)

            # Отрисовка
            color = (0, 255, 0) if predicted_emotion in positive_emotions else (0, 0, 255)
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            cv2.putText(img, predicted_emotion, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ==== Интерфейс ====
st.title("🎥 Emotion Recognition with Camera")
st.markdown("Включите камеру и наблюдайте за распознаванием эмоций в реальном времени.")

# Камера
ctx = webrtc_streamer(key="emotion", video_processor_factory=EmotionProcessor)

# Статистика
if st.button("📊 Показать статистику"):
    if not emotion_log:
        st.warning("Нет распознанных эмоций — включите камеру и подождите.")
    else:
        stats = Counter(emotion_log)
        st.markdown("### 📈 Эмоции за сессию:")
        total = sum(stats.values())
        for emo, count in stats.items():
            percent = (count / total) * 100
            st.write(f"**{emo}** — {count} раз ({percent:.1f}%)")

        # График
        fig, ax = plt.subplots()
        ax.bar(stats.keys(), stats.values(), color='skyblue')
        ax.set_title("Распределение эмоций")
        st.pyplot(fig)

        # Вердикт
        pos_total = sum(stats[e] for e in positive_emotions)
        ratio = pos_total / total
        st.markdown("### 🧾 Итог:")
        if ratio >= 0.5:
            st.success("✅ Клиент удовлетворён")
        else:
            st.error("❌ Клиент не удовлетворён")
