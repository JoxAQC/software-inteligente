# proyectos/clasificacion_ingresos.py

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

def run():
    st.title("📊 Clasificación de Ingresos con Múltiples Modelos")

    # Cargar datos
    df = pd.read_csv("proyectos/adult.csv", header=None, names=[
        "age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
    ])
    
    df = df.replace("?", np.nan).dropna()
    df = df.apply(lambda col: col.map(lambda x: x.strip() if isinstance(x, str) else x))

    X = df.drop("income", axis=1)
    y = df["income"]

    X_encoded = pd.get_dummies(X)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)

    modelos = {
        "Naive Bayes (Gaussian)": GaussianNB(),
        "Naive Bayes (Multinomial)": MultinomialNB(),
        "Naive Bayes (Bernoulli)": BernoulliNB(),
        "Random Forest": RandomForestClassifier(),
        "KNN": KNeighborsClassifier()
    }

    resultados = []

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

        st.subheader(f"🔍 Resultados: {nombre}")
        st.write(f"**Accuracy**:  {acc:.4f}")
        st.write(f"**Precisión**: {prec:.4f}")
        st.write(f"**Recall**:    {rec:.4f}")
        st.write(f"**F1 Score**:  {f1:.4f}")
        if auc is not None:
            st.write(f"**AUC**:       {auc:.4f}")

        # Matriz de confusión
        fig, ax = plt.subplots()
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
        st.pyplot(fig)

        # Reporte de clasificación
        st.text(classification_report(y_test, y_pred, target_names=le.classes_))

        # Curva ROC si aplica
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

    # Comparativa general
    st.subheader("📊 Comparación de Modelos")
    resultados_df = pd.DataFrame(resultados).set_index("Modelo")
    st.dataframe(resultados_df.style.format("{:.4f}").highlight_max(axis=0, color="lightgreen"))

    # Distribución de edad
    st.subheader("📦 Distribución de Edad por Nivel de Ingreso")
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x="income", y="age", ax=ax)
    st.pyplot(fig)

    # Heatmap de correlación
    st.subheader("🔥 Mapa de Calor de Variables Numéricas")
    fig, ax = plt.subplots(figsize=(10, 8))
    numerical_df = df.select_dtypes(include=['int64', 'float64'])
    sns.heatmap(numerical_df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
