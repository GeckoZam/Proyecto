import streamlit as st
from PIL import Image
import plotly.express as px
import pandas as pd
import keras
from keras.preprocessing.image import img_to_array
import tensorflow as tf
import numpy as np
import cv2

# Configuración de la página
st.set_page_config(layout="wide")
st.title("Detección de Emociones en Imágenes")

# Cargar el modelo preentrenado
model = tf.keras.models.load_model("modelEmocion2.keras")
emotion_labels = ['disgust', 'sad', 'fear', 'angry', 'neutral', 'happy', 'surprise']
faces = []


@st.cache_data
def cargar_imagen(image_file):
    img = Image.open(image_file)
    return img

# Función para preprocesar la imagen antes de pasársela al modelo
def preprocess_image(image):
    face = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, (48, 48))
    face2 = img_to_array(face)
    face2 = np.expand_dims(face2,axis=0)  # Añadir una dimensión extra para el batch size  
    face2 = np.expand_dims(face2, axis=-1)  # Añadir una dimensión extra para la profundidad de color
    faces.append(face2)
    return faces


# Subir la imagen
archivo_imagen = st.file_uploader("Sube una imagen para detectar emociones", type=["png", "jpg", "jpeg"])
if archivo_imagen is not None:
    st.image(archivo_imagen, width=250)
    img = cargar_imagen(archivo_imagen)

    # Convertir la imagen cargada a un formato adecuado para la predicción
    img_array = np.array(img)
    processed_image = preprocess_image(img_array)
    
    # Realizar la predicción de emociones
    emotion_probs = model.predict(processed_image)
    
    # Crear un DataFrame para las emociones detectadas
    emotion_counts = {emotion_labels[i]: emotion_probs[0][i] for i in range(len(emotion_labels))}
    emotion_data = pd.DataFrame(list(emotion_counts.items()), columns=['Emoción', 'Probabilidad'])

    # Crear el gráfico de barras para mostrar las emociones detectadas
    fig = px.bar(
        emotion_data,
        x='Emoción',
        y='Probabilidad',
        title='Probabilidades de Emociones Detectadas',
        labels={'Probabilidad': 'Probabilidad', 'Emoción': 'Emoción'},
        color='Probabilidad',
        color_continuous_scale='Viridis'
    )

    # Mostrar el gráfico
    st.title("Emociones Detectadas")
    st.plotly_chart(fig)
