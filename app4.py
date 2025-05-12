import numpy as np
import streamlit as st

st.title("🧠 Red Neuronal para la función XOR")

with st.sidebar:
    st.header("Parámetros del modelo")
    epochs = st.number_input("Épocas de entrenamiento", min_value=100, max_value=100000, value=10000, step=100)
    lr = st.slider("Tasa de aprendizaje", 0.001, 1.0, 0.1)
    hidden_neurons = st.slider("Neuronas en la capa oculta", 1, 10, 2)

# Datos XOR
st.subheader("📥 Datos de entrada y salida esperada")
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
expected_output = np.array([[0], [1], [1], [0]])

st.write("**Entradas:**")
st.table(inputs)

st.write("**Salidas esperadas (XOR):**")
st.table(expected_output)

# Activaciones
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

if st.button("Entrenar Red Neuronal"):
    # Inicialización de pesos y sesgos
    input_neurons, output_neurons = 2, 1
    hidden_weights = np.random.uniform(size=(input_neurons, hidden_neurons))
    hidden_bias = np.random.uniform(size=(1, hidden_neurons))
    output_weights = np.random.uniform(size=(hidden_neurons, output_neurons))
    output_bias = np.random.uniform(size=(1, output_neurons))

    # Entrenamiento
    for _ in range(epochs):
        # Forward
        hidden_input = np.dot(inputs, hidden_weights) + hidden_bias
        hidden_output = sigmoid(hidden_input)

        final_input = np.dot(hidden_output, output_weights) + output_bias
        predicted_output = sigmoid(final_input)

        # Backpropagation
        error = expected_output - predicted_output
        d_output = error * sigmoid_derivative(predicted_output)

        error_hidden = d_output.dot(output_weights.T)
        d_hidden = error_hidden * sigmoid_derivative(hidden_output)

        # Actualización
        output_weights += hidden_output.T.dot(d_output) * lr
        output_bias += np.sum(d_output, axis=0, keepdims=True) * lr
        hidden_weights += inputs.T.dot(d_hidden) * lr
        hidden_bias += np.sum(d_hidden, axis=0, keepdims=True) * lr

    st.subheader("✅ Resultados del entrenamiento")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Pesos ocultos (finales):**")
        st.table(hidden_weights)
        st.write("**Bias ocultos:**")
        st.table(hidden_bias)
    with col2:
        st.write("**Pesos de salida:**")
        st.table(output_weights)
        st.write("**Bias de salida:**")
        st.table(output_bias)

    st.subheader("🔮 Predicciones de la red neuronal")
    for i in range(len(inputs)):
        entrada = inputs[i]
        salida = predicted_output[i][0]
        st.write(f"Entrada {entrada} ➡️ Predicción: `{salida:.4f}`")

