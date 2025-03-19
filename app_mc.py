import streamlit as st
import joblib
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

rf_model = joblib.load("random_forest_mnist_optimized.pkl")

st.title("✍️ Handwritten Digit Classification App")
st.write("Upload an image, capture one using the webcam, or draw a digit on the canvas!")

option = st.radio("Choose Input Method:", ["Upload Image", "Use Webcam", "Draw Digit"])

image = None

if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)

elif option == "Use Webcam":
    camera_image = st.camera_input("Take a picture")
    if camera_image is not None:
        image = Image.open(camera_image)

elif option == "Draw Digit":
    st.write("Draw a digit below:")
    canvas_result = st_canvas(
        fill_color="white",  
        stroke_width=15,
        stroke_color="black",  
        background_color="white", 
        update_streamlit=True,
        width=280,
        height=280,  
        drawing_mode="freedraw",
        key="canvas",
    )

    if canvas_result.image_data is not None:
        image = Image.fromarray(canvas_result.image_data.astype(np.uint8))

if image is not None:
    st.write("### Processing Image...")
    image = image.convert('L')
    image = image.resize((28, 28))
    image = np.array(image)
    
    if np.any(image < 255):  
        img = np.invert(image)  

        img_2d = img.flatten().reshape(1, -1)  
        st.image(img, caption="Processed Image", use_column_width=True)

        
        pred = rf_model.predict(img_2d)
        st.write(f"Predicted digit: {pred[0]}")
    else:
        st.write("Draw a digit to get a prediction!")
