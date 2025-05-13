
# Software Inteligente - Índice de Proyectos

Este repositorio contiene una página web con un índice de proyectos interactivos realizados en **Streamlit** sobre temas de inteligencia artificial y aprendizaje automático. Puedes ver la página estática en:

🔗 [https://joxaqc.github.io/software-inteligente/](https://joxaqc.github.io/software-inteligente/)

Desde la sección **"Programas"**, puedes acceder a los siguientes proyectos desplegados en **Streamlit Cloud**:

## 📦 Proyectos en la nube

| Proyecto | Tema | Link |
|---------|------|------|
| 🧬 Algoritmo Genético | Optimización de combinación de pizzas | [Abrir](https://software-inteligente-pizzas.streamlit.app/) |
| 📊 Clasificación Naive Bayes | Modelos de clasificación probabilística | [Abrir](https://software-inteligente-naive-bayes.streamlit.app/) |
| 🧠 Backpropagation | Entrenamiento de redes neuronales simples | [Abrir](https://software-inteligente-backpropagation.streamlit.app/) |
| 💧 Calidad del Agua (NN) | Predicción de potabilidad con redes neuronales | [Abrir](https://software-inteligente-water-quality-pred.streamlit.app/) |

---

## 💻 Alternativas de ejecución local

### Opción 1: Abrir el archivo `index.html` en un navegador

Esto te permitirá navegar de forma visual por los proyectos de manera local.

### Opción 2: Ejecutar cada app por separado

1. Instala las dependencias:
```bash
pip install -r requirements.txt
```

2. Ejecuta el archivo correspondiente:

| Proyecto | Archivo |
|---------|---------|
| Algoritmo Genético | `app2.py` |
| Naive Bayes | `app3.py` |
| Backpropagation | `app4.py` |
| Calidad del Agua (NN) | `app5.py` |

```bash
streamlit run app2.py  # o app3.py, app4.py, app5.py según el caso
```

### 📂 Datos de entrada

- Para los proyectos **Backpropagation (app4.py)** y **Calidad del Agua (app5.py)** puedes cargar tus propios datasets. En la carpeta `/proyectos` hay archivos CSV de ejemplo que puedes subir desde la interfaz.

---

## 🛠️ Requisitos

- Python 3.8+
- Streamlit
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- plotly
- tensorflow (para proyectos de redes neuronales)

---

## 📄 Licencia

Este proyecto está destinado exclusivamente a fines educativos.
