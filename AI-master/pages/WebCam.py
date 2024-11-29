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

# Cargar el clasificador de rostros Haar
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Placeholder para reproducir la cámara
frame_placeholder = st.empty()

# Crear un placeholder para el gráfico que se actualizará en tiempo real
graph_placeholder = st.empty()

# Inicia la cámara
cap = cv2.VideoCapture(0)

# Lista para almacenar las emociones detectadas
emotion_counts = {emotion: 0 for emotion in emotion_labels}

# Bucle infinito para capturar frames de la cámara
if cap.isOpened():
    while True:
        try:
            ret, frame = cap.read()

            if not ret:
                # Si no se obtiene un frame, continuar buscando sin interrumpir la aplicación
                st.write("No se pudo obtener un frame. Intentando nuevamente...")
                continue

            # Convertir la imagen a escala de grises
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detectar rostros en la imagen usando el clasificador Haar
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Si se detectan rostros
            if len(faces) > 0:
                # Seleccionamos el primer rostro detectado
                x, y, w, h = faces[0]

                # Dibujar un rectángulo alrededor del rostro detectado
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

                # Recortar la región del rostro de la imagen
                face_region = gray_frame[y:y+h, x:x+w]

                # Preprocesar la imagen del rostro para la predicción
                face_region = cv2.resize(face_region, (48, 48))  # Redimensionar a 48x48
                face_region = img_to_array(face_region)  # Convertir a un array numpy
                face_region = np.expand_dims(face_region, axis=-1)  # Añadir una dimensión extra para la profundidad de color
                face_region = np.expand_dims(face_region, axis=0)  # Añadir una dimensión extra para el batch size

                # Realizar la predicción de emociones
                emotion_probs = model.predict(face_region)

                # Obtener la emoción con la mayor probabilidad
                predicted_emotion_idx = np.argmax(emotion_probs[0])
                predicted_emotion = emotion_labels[predicted_emotion_idx]
                predicted_probability = emotion_probs[0][predicted_emotion_idx]

                # Actualizar las probabilidades acumuladas
                emotion_counts[predicted_emotion] += predicted_probability

                # Mostrar la emoción detectada y el rostro en el frame
                caption = f"Emoción Detectada: {predicted_emotion} - Probabilidad: {predicted_probability:.2f}"
            else:
                caption = "Buscando rostro..."  # Cambiar el mensaje cuando no se detecta rostro

            # Convertir el frame a RGB para Streamlit y a imagen de PIL para mostrarla
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convertir la imagen a PIL
            frame_pil = Image.fromarray(frame_rgb)

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

            # Mostrar el gráfico en tiempo real con un key único para cada iteración
            timestamp = int(time.time() * 1000)  # Generar un timestamp único
            graph_placeholder.plotly_chart(fig, use_container_width=True, key=f"graph_{timestamp}")  # Usamos un timestamp para crear un key único

            # Pausar brevemente para permitir que otros elementos se rendericen, pero no detener por completo
            time.sleep(0.05)  # Reducido para que la actualización sea más rápida

        except Exception as e:
            # Captura cualquier excepción y muestra un mensaje de error
            st.error(f"Ocurrió un error: {e}")
            break  # Rompe el bucle si ocurre un error, para evitar que se repita indefinidamente
else:
    st.write("Error: Unable to open webcam.")

