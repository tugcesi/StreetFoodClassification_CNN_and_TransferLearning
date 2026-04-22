import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model

# ── Sayfa Ayarları ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🍜 Street Food Classifier",
    page_icon="🍔",
    layout="centered"
)

# ── Sabitler ────────────────────────────────────────────────────────────────
IMG_SIZE = (240, 240)

CLASS_NAMES = {
    0: "🧆 Falafel",
    1: "🍔 Burger",
    2: "🫓 Pani Puri",
    3: "🥨 Pretzel",
    4: "🌯 Shawarma",
    5: "🌭 Hot Dog",
    6: "🌮 Tacos",
    7: "🥞 Crepes",
    8: "🍜 Pad Thai"
}

MODEL_PATH = "street_food_cnn.h5"

# ── Model Yükleme ────────────────────────────────────────────────────────────
@st.cache_resource
def load_cnn_model():
    model = load_model(MODEL_PATH)
    return model

# ── Görsel Ön İşleme ─────────────────────────────────────────────────────────
def preprocess_image(image: Image.Image) -> np.ndarray:
    img = np.array(image.convert("RGB"))
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)   # (1, 240, 240, 3)
    return img

# ── Tahmin ───────────────────────────────────────────────────────────────────
def predict(model, img_array: np.ndarray):
    preds = model.predict(img_array, verbose=0)[0]   # (9,)
    top_idx = int(np.argmax(preds))
    return top_idx, preds

# ── Arayüz ───────────────────────────────────────────────────────────────────
st.title("🍜 Street Food Image Classifier")
st.markdown("Bir sokak yiyeceği görseli yükle, model hangi yiyecek olduğunu tahmin etsin!")
st.divider()

uploaded_file = st.file_uploader(
    "📁 Görsel Yükle (JPG / PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("📷 Yüklenen Görsel")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("🔍 Tahmin Sonuçları")

        with st.spinner("Model yükleniyor ve tahmin yapılıyor..."):
            model     = load_cnn_model()
            img_array = preprocess_image(image)
            top_idx, preds = predict(model, img_array)

        top_label      = CLASS_NAMES[top_idx]
        top_confidence = preds[top_idx] * 100

        st.success(f"**Tahmin: {top_label}**")
        st.metric(label="Güven Skoru", value=f"%{top_confidence:.1f}")

        st.divider()
        st.markdown("**📊 Tüm Sınıf Olasılıkları**")

        # Olasılıkları büyükten küçüğe sırala
        sorted_indices = np.argsort(preds)[::-1]
        for idx in sorted_indices:
            label = CLASS_NAMES[idx]
            prob  = preds[idx] * 100
            st.progress(
                int(prob),
                text=f"{label}: %{prob:.1f}"
            )

st.divider()
st.caption("Model: Custom CNN (street_food_cnn.h5) | 9 Sınıf | Input: 240×240")
