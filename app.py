import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from PIL import Image

st.set_page_config(page_title="Fruit Classification Dashboard", page_icon="🍎", layout="centered")

st.title("🍎 Fruit & Plant Classification System")
st.write("ارفع صورة أي فاكهة أو خضار والتطبيق هيتوقع نوعها!")

@st.cache_resource
def load_model():
    model = MobileNetV2(weights='imagenet')
    return model

with st.spinner("جاري تحميل النموذج..."):
    model = load_model()

st.success("تم تحميل النموذج بنجاح! 🚀")
st.markdown("---")

uploaded_file = st.file_uploader("اختر صورة فاكهة للتنبؤ بها", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="الصورة المرفوعة", use_container_width=True)
    
    with col2:
        img_array = np.array(image.convert('RGB'))
        img_resized = cv2.resize(img_array, (224, 224))
        img_preprocessed = preprocess_input(img_resized)
        img_expanded = np.expand_dims(img_preprocessed, axis=0)
        
        predictions = model.predict(img_expanded)
        results = decode_predictions(predictions, top=3)[0]
        
        st.markdown("### نتائج التنبؤ:")
        for i, (imagenet_id, label, score) in enumerate(results):
            st.write(f"**{i+1}. {label.replace('_', ' ').title()}**: {score*100:.2f}%")
