import streamlit as st
import pandas as pd
from utils import models
from keras.models import Sequential
from keras.callbacks import LambdaCallback
from streamlit_drawable_canvas import st_canvas
import numpy as np
import matplotlib.pyplot as plt
import cv2

st.title("MNIST (Handwritten Digits) Classification")
st.header("Loading the dataset", "h2")
uploaded_file = st.file_uploader("Upload the training dataset", type="csv")
# Create a callback to print training progress
print_callback = LambdaCallback(
    on_epoch_end=lambda epoch, logs: st.write(f"End of epoch {epoch}: loss = {logs['loss']}, val_loss = {logs['val_loss']}")
)

def split_data(dataframe, label_col, train_size):
    from sklearn.model_selection import train_test_split
    X = dataframe.drop(label_col, axis=1).values.reshape(-1, 28, 28, 1)
    y = dataframe[label_col].values
    st.write(f"The input's number of sample: {X.shape[0]:,}, the image's dimension: {X.shape[1:]}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size)
    st.write(f"The training set number of sample: {X_train.shape[0]:,}")
    st.write(f"The dev set number of sample: {X_test.shape[0]:,}")
    return X_train, X_test, y_train, y_test

def predict_component(model):
    st.header("Prediction", "h2")
    st.write("Draw a digit on the canvas below:")
    canvas_result = drawable_canva()
    if canvas_result.image_data is not None and st.button("Predict"):
        st.write("Model prediction:")
        # Convert from RGBA to 1 channel
        input = image_processing(canvas_result.image_data)
        print(input.shape)
        input = input.reshape(1, 28, 28, 1)
        print(input.shape)
        prediction = model.predict(input).argmax()
        st.write(f"Predicted digit: {prediction}")
    else:
        st.write("Draw a digit on the canvas above to see the model prediction.")

def drawable_canva():
    # Size of the canvas 28x28
    return st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=10,
        stroke_color="black",
        background_color="#fff",
    )

def image_processing(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resize to 28x28
    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    # Invert the image
    inverted = cv2.bitwise_not(resized)
    return inverted

def train_model(X_train, y_train, X_test, y_test):
    model = models.create_model()
    st.write("Training the model...")
    history = model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test), callbacks=[print_callback])
    model.save("model.keras")
    st.write("Model trained successfully!")
    plt.plot(history.epoch, history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.epoch, history.history['val_accuracy'], label = 'Validation Accuracy')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    st.pyplot(plt)
    st.session_state['trained_model'] = model
    return model

if uploaded_file is not None:
    dataframe = pd.read_csv(uploaded_file)
    # Add dropdown
    target = st.selectbox("Review mode", ["CSV", "Image"])
    if target == "CSV":
        st.write(dataframe.head())
    else:
        columns = st.columns(6)
        images = dataframe.iloc[:10, 1:].values.reshape(-1, 28, 28, 1)
        for i in range(len(images)):
            with columns[i % 6]:
                st.image(
                    images[i], 
                    caption=f"Label: {dataframe.iloc[i, 0]}", 
                    use_column_width="auto")
    label_col = st.selectbox("Select the label column", dataframe.columns, )
    # Train / test split
    st.divider()
    st.header("Train / Test Split", "h2")
    train_size = st.slider("Train size", 0.1, 0.9, 0.7)
    X_train, X_test, y_train, y_test = split_data(dataframe, label_col, train_size)
    # Display h2 header
    st.divider()
    st.header("Model Training", "h2")
    model = st.session_state['model'] if 'model' in st.session_state else None
    if model is not None:
        model.summary(print_fn=st.write)

    # Load model
    if st.button("Load Model"):
        model = models.load_model()
        st.session_state['model'] = model
    if st.button("Train Model"):
        model = train_model(X_train, y_train, X_test, y_test)
        st.session_state['model'] = model
    st.divider()
    if model is not None:
        predict_component(model)