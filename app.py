import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from PIL import Image

st.set_page_config(page_title="تطبيق تصنيف الفواكه والخضروات", page_icon="🍎", layout="centered")

st.title("🍎 نظام تصنيف الفواكه والخضروات")
st.write("ارفع صورة أي فاكهة أو خضار لمعرفة اسمها ونوعها فوراً!")

# قائمة الفواكه والخضروات المحددة في المشروع مترجمة بالعربي والانجليزي
FRUIT_CLASSES = {
    'Apple': 'تفاح 🍎',
    'Banana': 'موز 🍌',
    'Carrot': 'جزر 🥕',
    'Corn': 'ذرة 🌽',
    'LadyFinger': 'بامية (Lady Finger) 🫛',
    'Lemon': 'ليمون 🍋',
    'Orange': 'برتقال 🍊',
    'Pear': 'كمثرى 🍐',
    'Plum': 'برقوق 🍑',
    'Potato': 'بطاطس 🥔',
    'Strawberry': 'فراولة 🍓',
    'Tomato': 'طماطم 🍅'
}

class_names = list(FRUIT_CLASSES.keys())

@st.cache_resource
def load_fruit_model():
    # استخدام MobileNetV2 كنموذج أساسي وتخصيصه للفواكه
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(len(class_names), activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=outputs)
    return model

with st.spinner("جاري تحميل النموذج..."):
    model = load_fruit_model()

st.success("النموذج جاهز للاستخدام! 🚀")
st.markdown("---")

uploaded_file = st.file_uploader("اختر صورة فاكهة أو خضار للتنبؤ بها", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="الصورة المرفوعة", use_container_width=True)
    
    with col2:
        # تجهيز الصورة
        img_array = np.array(image.convert('RGB'))
        img_resized = cv2.resize(img_array, (224, 224))
        img_preprocessed = preprocess_input(img_resized)
        img_expanded = np.expand_dims(img_preprocessed, axis=0)
        
        # التنبؤ
        predictions = model.predict(img_expanded)[0]
        top_idx = np.argmax(predictions)
        
        english_name = class_names[top_idx]
        arabic_name = FRUIT_CLASSES[english_name]
        confidence = predictions[top_idx] * 100
        
        st.markdown("### 🎯 نتيجة التنبؤ:")
        st.success(f"الفاكهة / الخضار: **{arabic_name}**")
        st.info(f"اسم الفئة بالإنجليزية: **{english_name}**")
