import streamlit as st
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
import matplotlib.pyplot as plt
import numpy as np
import pathlib

# Mostrar versiones
print("TF version:", tf.__version__)
print("TFDS version:", tfds.__version__)

st.title("🌷 Clasificador de Flores con Transfer Learning")

# Configuración inicial en session_state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'history' not in st.session_state:
    st.session_state.history = None

# Sidebar para parámetros
with st.sidebar:
    st.header("⚙️ Configuración del Modelo")
    
    # Parámetros de datos
    st.subheader("Datos")
    train_size = st.slider("Número de imágenes de entrenamiento", 10, 100, 32)
    val_size = st.slider("Número de imágenes de validación", 5, 30, 8)
    batch_size = st.selectbox("Tamaño del batch", [2, 4, 8, 16, 32], index=0)
    
    # Parámetros del modelo
    st.subheader("Arquitectura")
    dense_units = st.slider("Neuronas en capa densa", 64, 1024, 512, step=64)
    dropout_rate = st.slider("Tasa de Dropout", 0.0, 0.9, 0.5, step=0.1)
    
    # Parámetros de entrenamiento
    st.subheader("Entrenamiento")
    epochs = st.slider("Épocas", 1, 20, 5)
    fine_tune_layers = st.slider("Capas para fine-tuning", 0, 10, 4)
    
    run_button = st.button("🚀 Entrenar Modelo")

# Cargar dataset tf_flowers
@st.cache_resource
def load_data():
    try:
        st.info("Cargando dataset tf_flowers...")
        (train_ds, val_ds), info = tfds.load(
            'tf_flowers',
            split=['train[:80%]', 'train[80%:]'],
            with_info=True,
            as_supervised=True
        )
        st.success("Dataset tf_flowers cargado exitosamente!")
        
        # Mostrar información del dataset
        st.write(f"**Número de clases:** {info.features['label'].num_classes}")
        st.write(f"**Clases disponibles:** {info.features['label'].names}")
        return (train_ds, val_ds), info
        
    except Exception as e:
        st.error(f"Error al cargar el dataset: {e}")
        st.stop()

# Cargar los datos
(ds_train, ds_val), info = load_data()

# Preprocesamiento
def preprocess_image(image, label):
    image = tf.image.resize(image, (224, 224))
    image = image / 255.0
    return image, label

# Preparar datos
train_data = ds_train.take(train_size).map(preprocess_image).batch(batch_size)
val_data = ds_val.take(val_size).map(preprocess_image).batch(batch_size)

# Función para mostrar imágenes
def mostrar_imagenes(dataset, titulo):
    plt.figure(figsize=(12, 5))
    class_names = info.features['label'].names
    
    for images, labels in dataset.take(1):
        for i in range(min(len(images), 10)):
            ax = plt.subplot(2, 5, i + 1)
            plt.imshow(images[i].numpy())
            plt.title(class_names[labels[i]])
            plt.axis("off")
    plt.suptitle(titulo)
    plt.tight_layout()
    st.pyplot(plt)

# Mostrar imágenes de ejemplo
st.subheader("📸 Muestra de Imágenes")
col1, col2 = st.columns(2)
with col1:
    if st.button("Mostrar imágenes de entrenamiento"):
        mostrar_imagenes(train_data, "Imágenes de Entrenamiento")
with col2:
    if st.button("Mostrar imágenes de validación"):
        mostrar_imagenes(val_data, "Imágenes de Validación")

# Construir modelo
def build_model(dense_units, dropout_rate, fine_tune_layers):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Congelar todas las capas inicialmente
    for layer in base_model.layers:
        layer.trainable = False
    
    model = Sequential([
        base_model,
        Flatten(),
        Dense(dense_units, activation='relu'),
        Dropout(dropout_rate),
        Dense(info.features['label'].num_classes, activation='softmax')
    ])
    
    # Descongelar capas para fine-tuning
    if fine_tune_layers > 0:
        for layer in base_model.layers[-fine_tune_layers:]:
            layer.trainable = True
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

if run_button:
    st.subheader("🏗 Construyendo Modelo")
    with st.spinner('Creando modelo...'):
        model = build_model(dense_units, dropout_rate, fine_tune_layers)
        st.session_state.model = model
    
    st.success("Modelo creado exitosamente!")
    model.summary(print_fn=lambda x: st.text(x))
    
    st.subheader("🔧 Resumen de Arquitectura")
    st.write(f"- Modelo base: VGG16 (congelado excepto últimas {fine_tune_layers} capas)")
    st.write(f"- Capa densa: {dense_units} neuronas con activación ReLU")
    st.write(f"- Dropout: {dropout_rate*100}%")
    st.write(f"- Capa de salida: {info.features['label'].num_classes} neuronas con activación softmax")
    
    st.subheader("🏋️ Entrenando Modelo")
    with st.spinner('Entrenando... Esto puede tomar unos minutos...'):
        history = model.fit(
            train_data, 
            epochs=epochs, 
            validation_data=val_data,
            verbose=1
        )
        st.session_state.history = history
        st.session_state.model_trained = True
    
    st.success("Entrenamiento completado!")
    
    # Mostrar resultados finales
    st.subheader("📊 Resultados del Entrenamiento")
    loss, accuracy = model.evaluate(val_data, verbose=0)
    st.metric("Precisión en validación", f"{accuracy*100:.2f}%")
    st.metric("Pérdida en validación", f"{loss:.4f}")
    
    # Gráficos de entrenamiento
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(history.history['accuracy'], label='Precisión entrenamiento')
    ax1.plot(history.history['val_accuracy'], label='Precisión validación')
    ax1.set_title('Precisión durante el entrenamiento')
    ax1.set_xlabel('Épocas')
    ax1.set_ylabel('Precisión')
    ax1.legend()
    
    ax2.plot(history.history['loss'], label='Pérdida entrenamiento')
    ax2.plot(history.history['val_loss'], label='Pérdida validación')
    ax2.set_title('Pérdida durante el entrenamiento')
    ax2.set_xlabel('Épocas')
    ax2.set_ylabel('Pérdida')
    ax2.legend()
    
    st.pyplot(fig)

# Sección de predicción
if st.session_state.model_trained:
    st.subheader("🔮 Probar con Nuevas Imágenes")
    
    uploaded_file = st.file_uploader("Sube una imagen de flor", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Procesar imagen
        image = tf.image.decode_image(uploaded_file.read(), channels=3)
        image = tf.image.resize(image, (224, 224))
        image = image / 255.0
        image = tf.expand_dims(image, axis=0)
        
        # Mostrar imagen
        st.image(uploaded_file, caption="Imagen subida", width=300)
        
        # Hacer predicción
        predictions = st.session_state.model.predict(image)
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        class_names = info.features['label'].names
        
        st.subheader("Resultado de la Predicción")
        st.metric("Clase predicha", class_names[predicted_class])
        st.metric("Confianza", f"{confidence*100:.2f}%")
        
        # Gráfico de probabilidades
        fig, ax = plt.subplots()
        ax.bar(class_names, predictions[0])
        ax.set_ylabel('Probabilidad')
        ax.set_title('Distribución de Probabilidades por Clase')
        plt.xticks(rotation=45)
        st.pyplot(fig)