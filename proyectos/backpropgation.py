import numpy as np
import streamlit as st

def run():
    # Función de activación sigmoidal
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(x):
        return x * (1 - x)

    # Datos de entrada y salida esperada
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    expected_output = np.array([[0], [1], [1], [0]])

    # Configuración de la red
    epochs = 10000
    lr = 0.1
    inputLayerNeurons, hiddenLayerNeurons, outputLayerNeurons = 2, 2, 1

    # Inicialización de pesos y sesgos
    hidden_weights = np.random.uniform(size=(inputLayerNeurons, hiddenLayerNeurons))
    hidden_bias = np.random.uniform(size=(1, hiddenLayerNeurons))
    output_weights = np.random.uniform(size=(hiddenLayerNeurons, outputLayerNeurons))
    output_bias = np.random.uniform(size=(1, outputLayerNeurons))

    # Entrenamiento
    for _ in range(epochs):
        # Propagación hacia adelante
        hidden_layer_activation = np.dot(inputs, hidden_weights)
        hidden_layer_activation += hidden_bias
        hidden_layer_output = sigmoid(hidden_layer_activation)

        output_layer_activation = np.dot(hidden_layer_output, output_weights)
        output_layer_activation += output_bias
        predicted_output = sigmoid(output_layer_activation)

        # Backpropagación
        error = expected_output - predicted_output
        d_predicted_output = error * sigmoid_derivative(predicted_output)

        error_hidden_layer = d_predicted_output.dot(output_weights.T)
        d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

        # Actualización de pesos y sesgos
        output_weights += hidden_layer_output.T.dot(d_predicted_output) * lr
        output_bias += np.sum(d_predicted_output) * lr
        hidden_weights += inputs.T.dot(d_hidden_layer) * lr
        hidden_bias += np.sum(d_hidden_layer) * lr

    # Mostrar resultados en Streamlit
    st.write("**Pesos iniciales:**")
    st.write(hidden_weights)
    st.write("**Bias iniciales:**")
    st.write(hidden_bias)
    st.write("**Pesos finales después de entrenar:**")
    st.write(output_weights)
    st.write("**Bias finales después de entrenar:**")
    st.write(output_bias)
    st.write("**Predicción:**")
    st.write(predicted_output)

# Llamar a la función run() en la app de Streamlit
if __name__ == "__main__":
    run()
