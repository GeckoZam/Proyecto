import streamlit as st
from PIL import Image
import plotly.express as px
import pandas as pd
import cv2
import keras
from keras.preprocessing.image import img_to_array
import time
import tensorflow as tf
import numpy as np

# Título de la aplicación
st.title("Página de la Webcam con Detección de Emociones en Tiempo Real")

# Cargar el modelo preentrenado
model = tf.keras.models.load_model("modelEmocion1000.keras")
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'neutral', 'surprise']

# Placeholder para reproducir la cámara
frame_placeholder = st.empty()

# Crear un placeholder para el gráfico que se actualizará en tiempo real
graph_placeholder = st.empty()

# Inicia la cámara
cap = cv2.VideoCapture(0)

# Lista para almacenar las emociones detectadas
emotion_counts = {emotion: 0 for emotion in emotion_labels}

# Loop para tomar frames de la cámara y analizarlos
if cap.isOpened():
    while True:  # Bucle infinito para captura de frames
        ret, frame = cap.read()
        if not ret:
            st.write("Error: No frame available.")
            break
        
        # Preprocesamiento de la imagen
        face = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convertir a escala de grises
        face = cv2.resize(face, (48, 48))  # Redimensionar la imagen a 48x48
        face = img_to_array(face)  # Convertir la imagen a un array numpy
        face = np.expand_dims(face, axis=-1)  # Añadir una dimensión extra para la profundidad de color
        face = np.expand_dims(face, axis=0)  # Añadir una dimensión extra para el batch size
        
        # Realizar la predicción de emociones
        emotion_probs = model.predict(face)
        
        # Obtener la emoción con la mayor probabilidad
        predicted_emotion_idx = np.argmax(emotion_probs[0])
        predicted_emotion = emotion_labels[predicted_emotion_idx]
        predicted_probability = emotion_probs[0][predicted_emotion_idx]

        # Actualizar las probabilidades acumuladas
        emotion_counts[predicted_emotion] += predicted_probability

        # Mostrar el frame y la emoción detectada
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convertir el frame a RGB
        frame_pil = Image.fromarray(frame_rgb)  # Convertir a imagen de PIL para Streamlit
        caption = f"Emoción Detectada: {predicted_emotion} - Probabilidad: {predicted_probability:.2f}"
        
        # Mostrar la imagen con la emoción detectada
        frame_placeholder.image(frame_pil, caption=caption)

        # Convertir los resultados de la detección de emociones en un DataFrame
        emotion_data = pd.DataFrame(list(emotion_counts.items()), columns=['Emotion', 'Probability Sum'])

        # Crear gráfico de barras para las emociones detectadas
        fig = px.bar(
            emotion_data,
            x='Emotion',
            y='Probability Sum',
            title='Frecuencia y Probabilidad de Emociones Detectadas en Tiempo Real',
            labels={'Emotion': 'Emoción', 'Probability Sum': 'Suma de Probabilidades'},
            color='Probability Sum',
            color_continuous_scale='Spectral'
        )

        # Mostrar el gráfico en tiempo real
        graph_placeholder.plotly_chart(fig, use_container_width=True)

        # Pausar brevemente para permitir que otros elementos se rendericen
        time.sleep(0.1)

else:
    st.write("Error: Unable to open webcam.")

# Liberar la cámara al finalizar
cap.release()
