import streamlit as st
import pathlib
import importlib.util

st.set_page_config(page_title="Proyectos de Python", layout="wide")
st.title("Algoritmos y Proyectos de Software Inteligente")

# Diccionario de proyectos
proyectos = {
    "Ejemplo Algoritmos Genéticos": "proyectos/ag_reinas.py",
    "Algoritmos Genéticos": "proyectos/algoritmo_genetico_pizzas.py",
    "Clasificación Bayesiana y Naive Bayes": "proyectos/naive_bayes.py",
    "Backpropagation": "proyectos/backpropgation.py",
    "Modelo de predicción de la calidad del agua": "proyectos/water_pred.py",
}

# Menú lateral
st.sidebar.title("📁 Selecciona un proyecto")
proyecto_seleccionado = st.sidebar.selectbox("Proyectos disponibles:", list(proyectos.keys()))

# Mostrar código fuente
ruta_archivo = proyectos[proyecto_seleccionado]
st.subheader(f"📄 Código del proyecto: {proyecto_seleccionado}")
path = pathlib.Path(ruta_archivo)

if path.exists():
    with open(path, "r", encoding="utf-8") as f:
        codigo = f.read()
    st.code(codigo, language="python")
else:
    st.error("⚠️ Archivo no encontrado. Verifica la ruta.")

# Botón para ejecutar
st.divider()
st.subheader("▶️ Ejecución del proyecto")

if st.button("Ejecutar Proyecto"):
    try:
        module_name = f"mod_{proyecto_seleccionado.replace(' ', '_').lower()}"
        spec = importlib.util.spec_from_file_location(module_name, ruta_archivo)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if hasattr(module, "run"):
            module.run()
        else:
            st.warning("Este archivo no tiene una función `run()` para ejecutar.")
    except Exception as e:
        st.error(f"❌ Error al ejecutar el módulo: {e}")
