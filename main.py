import streamlit as st
import tensorflow as tf
import numpy as np

st.set_page_config(
    page_title="Fruits & Vegetables Recognition",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for background and font
st.markdown("""
    <style>
    .main {
        background-color: #f4f9f4;
        padding: 10px;
    }
    h1, h2, h3 {
        color: #005B41;
    }
    .stButton>button {
        background-color: #28a745;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# TensorFlow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_model.h5")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(64, 64))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

# Sidebar
st.sidebar.title("🍇 Dashboard Menu")
app_mode = st.sidebar.radio("📂 Navigate to", ["🏠 Home", "📘 About Project", "🔍 Prediction"])

# Main Page
if app_mode == "🏠 Home":
    st.title("🥦 Fruits & Vegetables Recognition System")
    st.markdown("### Powered by TensorFlow & Streamlit")
    st.image("rempah.jpg", use_container_width=True, caption="Example - Fruits and Vegetables")

elif app_mode == "📘 About Project":
    st.title("📘 About This Project")
    st.subheader("📊 Dataset Overview")
    st.markdown("This dataset contains images of various **fruits** and **vegetables**.")
    
    st.markdown("#### 🍎 Fruits:")
    st.code("cau, apple, pear, grapes, orange, kiwi, watermelon, pomegranate, pineapple, mango")

    st.markdown("#### 🥕 Vegetables:")
    st.code("cucumber, carrot, capsicum, onion, potato, lemon, tomato, raddish, beetroot, cabbage, lettuce, spinach, soy bean, cauliflower, bell pepper, chilli pepper, turnip, corn, sweetcorn, sweet potato, paprika, jalepeño, ginger, garlic, peas, eggplant")

    st.subheader("📁 Dataset Structure")
    st.markdown("""
    - `train` (100 images each category)  
    - `test` (10 images each)  
    - `validation` (10 images each)
    """)

elif app_mode == "🔍 Prediction":
    st.title("🔍 Predict Fruits or Vegetables")

    test_image = st.file_uploader("📤 Upload an Image", type=["jpg", "jpeg", "png"])
    
    if test_image is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 👁️ Uploaded Image")
            st.image(test_image, use_container_width=True)

        if st.button("🔮 Predict"):
            st.spinner("Predicting...")
            result_index = model_prediction(test_image)
            # Read Labels
            with open("labels.txt") as f:
                labels = [line.strip() for line in f.readlines()]
            prediction_label = labels[result_index]

            with col2:
                st.markdown("### ✅ Prediction Result")
                st.success(f"🎉 Ini adalah **{prediction_label}**!")
                st.balloons()
