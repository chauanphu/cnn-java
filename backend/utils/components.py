from keras.models import Sequential
import streamlit as st
import keras
# UI to manually create a Keras model
def input_layer(id=0):
    _shape = st.text_input("Input Shape", "28,28,1", key=id)
    _shape = tuple(map(int, _shape.split(",")))
    return keras.layers.InputLayer(key=f"input_{id}", shape=_shape)

