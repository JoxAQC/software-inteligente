import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, roc_auc_score
)
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

st.title("📊 Clasificación de Ingresos con Múltiples Modelos")

# Inicializar variables en session_state si no existen
if 'modelo_entrenado' not in st.session_state:
    st.session_state.modelo_entrenado = False
if 'mejor_modelo' not in st.session_state:
    st.session_state.mejor_modelo = None
if 'le' not in st.session_state:
    st.session_state.le = None
if 'columnas_dummies' not in st.session_state:
    st.session_state.columnas_dummies = None
if 'columnas_originales' not in st.session_state:
    st.session_state.columnas_originales = None

archivo = st.file_uploader("📂 Carga el archivo CSV", type=["csv"])
tiene_header = st.checkbox("El archivo tiene encabezado", value=True)

nombres_columnas = []
if not tiene_header:
    columnas_raw = st.text_input("Nombres de columnas (separados por coma)", 
                                value="age,workclass,fnlwgt,education,education-num,marital-status,occupation,relationship,race,sex,capital-gain,capital-loss,hours-per-week,native-country,income")
    if columnas_raw:
        nombres_columnas = [col.strip() for col in columnas_raw.split(",")]

if archivo is not None:
    try:
        df = pd.read_csv(archivo, header=0 if tiene_header else None)

        if not tiene_header:
            if len(nombres_columnas) != df.shape[1]:
                st.error(f"❌ El número de columnas ingresadas ({len(nombres_columnas)}) no coincide con el del archivo ({df.shape[1]} columnas).")
                st.stop()
            df.columns = nombres_columnas

        df = df.replace("?", np.nan).dropna()
        df = df.apply(lambda col: col.map(lambda x: x.strip() if isinstance(x, str) else x))

        if "income" not in df.columns:
            st.error("❌ El conjunto de datos debe contener una columna llamada `income`.")
            st.stop()
            
        X = df.drop("income", axis=1)
        y = df["income"]

        # Guardar las columnas originales
        st.session_state.columnas_originales = X.columns.tolist()
        
        # Preprocesamiento
        X_encoded = pd.get_dummies(X)
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        # Guardar las columnas después del one-hot encoding
        st.session_state.columnas_dummies = X_encoded.columns.tolist()
        st.session_state.le = le

        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)

        if st.button("🚀 Ejecutar clasificación"):
            modelos = {
                "Naive Bayes (Gaussian)": GaussianNB(),
                "Naive Bayes (Multinomial)": MultinomialNB(),
                "Naive Bayes (Bernoulli)": BernoulliNB(),
                "Random Forest": RandomForestClassifier(),
                "KNN": KNeighborsClassifier()
            }

            resultados = []
            mejores_modelos = {}

            for nombre, modelo in modelos.items():
                modelo.fit(X_train, y_train)
                y_pred = modelo.predict(X_test)
                y_prob = modelo.predict_proba(X_test)[:, 1] if hasattr(modelo, "predict_proba") else None

                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred)
                rec = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None

                resultados.append({
                    "Modelo": nombre,
                    "Accuracy": acc,
                    "Precision": prec,
                    "Recall": rec,
                    "F1 Score": f1,
                    "AUC": auc
                })

                mejores_modelos[nombre] = modelo

                st.subheader(f"🔍 Resultados: {nombre}")
                st.write(f"**Accuracy**:  {acc:.4f}")
                st.write(f"**Precisión**: {prec:.4f}")
                st.write(f"**Recall**:    {rec:.4f}")
                st.write(f"**F1 Score**:  {f1:.4f}")
                if auc is not None:
                    st.write(f"**AUC**:       {auc:.4f}")

                fig, ax = plt.subplots()
                cm = confusion_matrix(y_test, y_pred)
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
                st.pyplot(fig)

                st.text(classification_report(y_test, y_pred, target_names=le.classes_))

                if y_prob is not None:
                    fpr, tpr, _ = roc_curve(y_test, y_prob)
                    fig, ax = plt.subplots()
                    ax.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
                    ax.plot([0, 1], [0, 1], 'k--')
                    ax.set_xlabel("Falsos Positivos")
                    ax.set_ylabel("Verdaderos Positivos")
                    ax.set_title(f"Curva ROC - {nombre}")
                    ax.legend()
                    st.pyplot(fig)

            st.subheader("📊 Comparación de Modelos")
            resultados_df = pd.DataFrame(resultados).set_index("Modelo")
            st.dataframe(resultados_df.style.format("{:.4f}").highlight_max(axis=0, color="lightgreen"))

            # Identificar el mejor modelo
            mejor_modelo_nombre = resultados_df['Accuracy'].idxmax()
            st.session_state.mejor_modelo = mejores_modelos[mejor_modelo_nombre]
            st.session_state.modelo_entrenado = True
            
            st.success(f"🌟 El mejor modelo es: {mejor_modelo_nombre} (Accuracy: {resultados_df.loc[mejor_modelo_nombre, 'Accuracy']:.4f})")

            st.subheader("📦 Distribución de Edad por Nivel de Ingreso")
            fig, ax = plt.subplots()
            sns.boxplot(data=df, x="income", y="age", ax=ax)
            st.pyplot(fig)

            st.subheader("🔥 Mapa de Calor de Variables Numéricas")
            fig, ax = plt.subplots(figsize=(10, 8))
            numerical_df = df.select_dtypes(include=['int64', 'float64'])
            sns.heatmap(numerical_df.corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Error al procesar el archivo: {e}")

# Sección de predicción (siempre visible si hay un modelo entrenado)
if st.session_state.modelo_entrenado:
    st.subheader("🔮 Predecir Income con Nuevos Datos")
    
    # Crear un formulario para ingresar los valores
    with st.form("prediccion_form"):
        st.write("Ingrese los valores para predecir el income:")
        
        # Crear inputs para cada columna original (sin income)
        valores = {}
        for col in st.session_state.columnas_originales:
            if df[col].dtype == 'object':
                # Para variables categóricas, mostrar un selectbox con las opciones disponibles
                opciones = df[col].unique().tolist()
                valores[col] = st.selectbox(f"{col}", opciones)
            else:
                # Para variables numéricas, mostrar un input numérico
                min_val = float(df[col].min())
                max_val = float(df[col].max())
                valores[col] = st.number_input(f"{col}", min_value=min_val, max_value=max_val, value=min_val)
        
        submitted = st.form_submit_button("Predecir Income")
        
        if submitted:
            # Crear un DataFrame con los valores ingresados
            nuevo_dato = pd.DataFrame([valores])
            
            # Aplicar one-hot encoding al nuevo dato
            nuevo_dato_encoded = pd.get_dummies(nuevo_dato)
            
            # Asegurarse de que todas las columnas de entrenamiento estén presentes
            for col in st.session_state.columnas_dummies:
                if col not in nuevo_dato_encoded.columns:
                    nuevo_dato_encoded[col] = 0
            
            # Ordenar las columnas en el mismo orden que los datos de entrenamiento
            nuevo_dato_encoded = nuevo_dato_encoded[st.session_state.columnas_dummies]
            
            # Hacer la predicción
            prediccion_encoded = st.session_state.mejor_modelo.predict(nuevo_dato_encoded)
            prediccion = st.session_state.le.inverse_transform(prediccion_encoded)
            
            # Mostrar la predicción
            st.success(f"Predicción de Income: {prediccion[0]}")
            
            # Si el modelo soporta probabilidades, mostrarlas
            if hasattr(st.session_state.mejor_modelo, 'predict_proba'):
                probabilidades = st.session_state.mejor_modelo.predict_proba(nuevo_dato_encoded)[0]
                for i, clase in enumerate(st.session_state.le.classes_):
                    st.write(f"Probabilidad de {clase}: {probabilidades[i]:.2f}")