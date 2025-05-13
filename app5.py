import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_score

# Configuración inicial
st.set_page_config(layout="wide", page_title="Análisis de Potabilidad del Agua")
st.title("🚰 Análisis de Potabilidad del Agua con Machine Learning")

# Sidebar para configuración
with st.sidebar:
    st.header("⚙️ Configuración General")
    
    # Carga de archivo
    uploaded_file = st.file_uploader("Subir archivo de datos", 
                                   type=["xls", "xlsx", "csv"],
                                   help="Formatos soportados: XLS, XLSX, CSV")
    
    if uploaded_file:
        try:
            # Determinar el tipo de archivo y cargarlo
            if uploaded_file.name.endswith(('.xls', '.xlsx')):
                # Para archivos .xls (Excel 97-2003)
                if uploaded_file.name.endswith('.xls'):
                    data = pd.read_excel(uploaded_file, engine='xlrd')
                # Para archivos .xlsx (Excel 2007+)
                else:
                    data = pd.read_excel(uploaded_file, engine='openpyxl')
            else:
                data = pd.read_csv(uploaded_file)
                
            # Verificar que el DataFrame no esté vacío
            if data.empty:
                st.error("El archivo está vacío")
                st.stop()
                
            # Selector de columna objetivo
            target_col = st.selectbox("Seleccionar columna objetivo", 
                                    data.columns,
                                    index=len(data.columns)-1 if 'Potability' in data.columns else 0)
            
            # Renombrar columna objetivo si es necesario
            if target_col != 'Potability':
                data = data.rename(columns={target_col: 'Potability'})
                
        except Exception as e:
            st.error(f"Error al cargar el archivo: {str(e)}")
            st.stop()
    else:
        st.warning("Por favor sube un archivo de datos")
        st.stop()

    # Configuración de preprocesamiento
    st.header("🔧 Preprocesamiento")
    test_size = st.slider("Tamaño del conjunto de prueba", 0.1, 0.5, 0.2, 0.05)
    random_state = st.number_input("Semilla aleatoria", 0, 1000, 42)
    
    # Manejo de valores faltantes
    st.subheader("Valores Faltantes")
    fill_method = st.radio("Método para completar valores faltantes",
                          ["Mediana (recomendado)", 
                           "Media", 
                           "Eliminar filas",
                           "Valor personalizado"],
                          index=0)
    
    if fill_method == "Valor personalizado":
        custom_value = st.number_input("Ingresa el valor a usar para rellenar")
    
    # Balanceo de clases
    st.subheader("Balanceo de Clases")
    use_smote = st.checkbox("Usar SMOTE para balancear clases", value=True)
    if use_smote:
        smote_ratio = st.slider("Ratio de muestreo para SMOTE", 0.1, 2.0, 1.0, 0.1)
    
    # Escalado de características
    st.subheader("Escalado de Características")
    scaling_method = st.radio("Método de escalado",
                            ["StandardScaler (recomendado)",
                             "MinMaxScaler",
                             "Sin escalado"])
    
    # Configuración de modelos
    st.header("🤖 Configuración de Modelos")
    
    # Modelos clásicos
    st.subheader("Modelos Clásicos")
    models_options = {
        "Regresión Logística": {"active": st.checkbox("Regresión Logística", True),
                               "params": {
                                   "max_iter": st.number_input("Iteraciones máx (LR)", 100, 5000, 1000),
                                   "C": st.slider("Parámetro C (LR)", 0.01, 10.0, 1.0)
                               }},
        "Random Forest": {"active": st.checkbox("Random Forest", True),
                         "params": {
                             "n_estimators": st.number_input("N° estimadores (RF)", 10, 500, 100),
                             "max_depth": st.number_input("Profundidad máx (RF)", 2, 50, 10)
                         }},
        "SVM": {"active": st.checkbox("SVM", False),
                "params": {
                    "C": st.slider("Parámetro C (SVM)", 0.01, 10.0, 1.0),
                    "kernel": st.selectbox("Kernel (SVM)", ["rbf", "linear", "poly"])
                }},
        "KNN": {"active": st.checkbox("KNN", False),
                "params": {
                    "n_neighbors": st.number_input("N° vecinos (KNN)", 1, 50, 5)
                }}
    }
    
    # Red Neuronal
    st.subheader("Red Neuronal")
    use_nn = st.checkbox("Incluir Red Neuronal", False)
    if use_nn:
        nn_params = {
            "epochs": st.number_input("Épocas de entrenamiento", 10, 500, 50),
            "batch_size": st.number_input("Tamaño de lote", 8, 256, 32),
            "learning_rate": st.slider("Tasa de aprendizaje", 0.0001, 0.1, 0.001, 0.0001, format="%.4f"),
            "hidden_units": st.number_input("Neuronas en capa oculta", 4, 128, 16),
            "dropout_rate": st.slider("Tasa de Dropout", 0.0, 0.5, 0.2)
        }

# Pestañas principales
tab1, tab2, tab3, tab4 = st.tabs(["📊 Exploración", "📈 Visualización", "🛠 Preprocesamiento", "🤖 Modelado"])

with tab1:
    st.header("Exploración de Datos")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Datos del Archivo")
        st.write(f"📄 Nombre: {uploaded_file.name}")
        st.write(f"📊 Forma: {data.shape[0]} filas × {data.shape[1]} columnas")
        
        st.subheader("Primeras filas")
        st.dataframe(data.head())
    
    with col2:
        st.subheader("Estadísticas Descriptivas")
        st.dataframe(data.describe())
        
        st.subheader("Balance de Clases")
        class_counts = data['Potability'].value_counts()
        fig = px.pie(values=class_counts, 
                    names=['No Potable', 'Potable'],
                    color=['No Potable', 'Potable'],
                    color_discrete_map={'No Potable':'red', 'Potable':'green'})
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Visualización Interactiva")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.subheader("Configuración")
        feature = st.selectbox("Variable a visualizar", 
                             data.columns.drop('Potability'))
        plot_type = st.selectbox("Tipo de gráfico",
                               ["Histograma", "Boxplot", "Violín"])
        color_palette = st.selectbox("Paleta de colores",
                                   ["Tema claro", "Tema oscuro", "Rojo/Verde"])
    
    with col1:
        st.subheader(f"Distribución de {feature}")
        
        if color_palette == "Tema claro":
            colors = ['#1F77B4', '#FF7F0E']  # Azul/Naranja
        elif color_palette == "Tema oscuro":
            colors = ['#2CA02C', '#D62728']  # Verde/Rojo
        else:
            colors = ['#FF0000', '#00FF00']  # Rojo/Verde
        
        if plot_type == "Histograma":
            fig = px.histogram(data, x=feature, color='Potability',
                             nbins=30, barmode='overlay',
                             color_discrete_sequence=colors,
                             marginal='box')
        elif plot_type == "Boxplot":
            fig = px.box(data, x='Potability', y=feature,
                       color='Potability', color_discrete_sequence=colors)
        else:  # Violín
            fig = px.violin(data, x='Potability', y=feature,
                          color='Potability', color_discrete_sequence=colors,
                          box=True)
        
        # Líneas de referencia según variable
        if feature == 'ph':
            fig.add_vline(x=7, line_dash="dash", line_color="black")
        elif feature == 'Hardness':
            fig.add_vline(x=151, line_dash="dash", line_color="black")
            fig.add_vline(x=301, line_dash="dash", line_color="black")
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Correlaciones")
    corr_method = st.radio("Método de correlación",
                         ["Pearson", "Spearman", "Kendall"],
                         horizontal=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(data.corr(method=corr_method.lower()), 
               annot=True, ax=ax, cmap='coolwarm',
               fmt=".2f", linewidths=.5)
    st.pyplot(fig)

with tab3:
    st.header("Preprocesamiento de Datos")
    
    # Manejo de valores faltantes
    st.subheader("Tratamiento de Valores Faltantes")
    data_processed = data.copy()
    
    if fill_method.startswith("Mediana"):
        data_processed = data.fillna(data.median())
        st.success(f"Valores faltantes rellenados con la mediana")
    elif fill_method == "Media":
        data_processed = data.fillna(data.mean())
        st.success(f"Valores faltantes rellenados con la media")
    elif fill_method == "Valor personalizado":
        data_processed = data.fillna(custom_value)
        st.success(f"Valores faltantes rellenados con {custom_value}")
    else:
        initial_rows = data.shape[0]
        data_processed = data.dropna()
        removed_rows = initial_rows - data_processed.shape[0]
        st.warning(f"Se eliminaron {removed_rows} filas con valores faltantes")
    
    st.dataframe(data_processed.isnull().sum().to_frame("Valores faltantes restantes"))
    
    # División de datos
    st.subheader("División de Datos")
    X = data_processed.drop('Potability', axis=1)
    y = data_processed['Potability']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y
    )
    
    # Balanceo de clases
    if use_smote:
        smote = SMOTE(sampling_strategy=smote_ratio, random_state=random_state)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        st.success(f"Datos balanceados - Clase 0: {sum(y_train==0)}, Clase 1: {sum(y_train==1)}")
    else:
        st.warning(f"Datos desbalanceados - Clase 0: {sum(y_train==0)}, Clase 1: {sum(y_train==1)}")
    
    # Escalado de características
    st.subheader("Escalado de Características")
    if scaling_method == "StandardScaler (recomendado)":
        scaler = StandardScaler()
    elif scaling_method == "MinMaxScaler":
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
    else:
        scaler = None
    
    if scaler:
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        st.success(f"Características escaladas usando {scaling_method.split(' ')[0]}")
    else:
        X_train_scaled = X_train.values
        X_test_scaled = X_test.values
        st.warning("No se aplicó escalado a las características")
    
    st.success("Preprocesamiento completado!")

with tab4:
    st.header("Modelado y Evaluación")
    
    # Importar todos los modelos necesarios al inicio (FUERA de los bloques condicionales)
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    
    # Función mejorada para evaluar modelos
    def evaluate_model(model, model_name, X_test, y_test, is_nn=False):
        try:
            # Obtener predicciones
            if is_nn:
                y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
            else:
                y_pred = model.predict(X_test)
            
            # Asegurar que los datos sean numpy arrays
            y_test = np.array(y_test).ravel()
            y_pred = np.array(y_pred).ravel()
            
            # Verificación de seguridad
            if len(y_test) != len(y_pred):
                min_length = min(len(y_test), len(y_pred))
                y_test = y_test[:min_length]
                y_pred = y_pred[:min_length]
                st.warning(f"Ajustadas las longitudes a {min_length} muestras")
            
            # Calcular métricas con manejo robusto
            try:
                report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                cm = confusion_matrix(y_test, y_pred)
                
                # Mostrar resultados
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Accuracy", f"{report['accuracy']:.2%}")
                    if '1' in report:
                        st.metric("Precisión (Clase 1)", f"{report['1']['precision']:.2%}")
                        st.metric("Recall (Clase 1)", f"{report['1']['recall']:.2%}")
                
                with col2:
                    fig, ax = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                              xticklabels=['No Potable', 'Potable'],
                              yticklabels=['No Potable', 'Potable'])
                    ax.set_title('Matriz de Confusión')
                    st.pyplot(fig)
                
                return report['accuracy']  # Usamos accuracy como métrica principal
                
            except Exception as e:
                st.error(f"Error en métricas: {str(e)}")
                return 0
                
        except Exception as e:
            st.error(f"Error en predicción: {str(e)}")
            return 0

    # Entrenamiento y evaluación de modelos
    results = {}
    
    # Modelos clásicos - Versión simplificada y robusta
    model_classes = {
        "Regresión Logística": LogisticRegression,
        "Random Forest": RandomForestClassifier,
        "SVM": SVC,
        "KNN": KNeighborsClassifier
    }
    
    for model_name, config in models_options.items():
        if config["active"] and model_name in model_classes:
            try:
                st.subheader(model_name)
                
                with st.spinner(f"Entrenando {model_name}..."):
                    # Crear instancia del modelo con parámetros
                    model = model_classes[model_name](**{
                        'random_state': random_state,
                        **config["params"]
                    })
                    
                    # Entrenar y evaluar
                    model.fit(X_train_scaled, y_train)
                    score = evaluate_model(model, model_name, X_test_scaled, y_test)
                    results[model_name] = score
                
            except Exception as e:
                st.error(f"Error entrenando {model_name}: {str(e)}")
                results[model_name] = 0

    # Red Neuronal con manejo robusto
    if use_nn:
        st.subheader("Red Neuronal")
        
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense, Dropout
            from tensorflow.keras.optimizers import Adam
            from tensorflow.keras.callbacks import Callback
            
            # Clase para monitorear el entrenamiento
            class TrainingMonitor(Callback):
                def on_epoch_end(self, epoch, logs=None):
                    if epoch % 10 == 0:
                        st.write(f"Época {epoch}: loss={logs['loss']:.4f}, accuracy={logs['accuracy']:.4f}")
            
            model = Sequential([
                Dense(nn_params["hidden_units"], activation='relu', input_shape=(X_train_scaled.shape[1],)),
                Dropout(nn_params["dropout_rate"]),
                Dense(1, activation='sigmoid')
            ])
            
            optimizer = Adam(learning_rate=nn_params["learning_rate"])
            model.compile(optimizer=optimizer, 
                         loss='binary_crossentropy', 
                         metrics=['accuracy'])
            
            with st.spinner("Entrenando red neuronal..."):
                history = model.fit(
                    X_train_scaled, y_train,
                    epochs=nn_params["epochs"],
                    batch_size=nn_params["batch_size"],
                    validation_data=(X_test_scaled, y_test),
                    verbose=0,
                    callbacks=[TrainingMonitor()]
                )
            
            # Mostrar historia de entrenamiento
            try:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                ax1.plot(history.history.get('accuracy', []), label='Train Accuracy')
                ax1.plot(history.history.get('val_accuracy', []), label='Validation Accuracy')
                ax1.set_title('Accuracy')
                ax1.legend()
                
                ax2.plot(history.history.get('loss', []), label='Train Loss')
                ax2.plot(history.history.get('val_loss', []), label='Validation Loss')
                ax2.set_title('Loss')
                ax2.legend()
                
                st.pyplot(fig)
            except:
                st.warning("No se pudo graficar el historial de entrenamiento")
            
            # Evaluación
            precision = evaluate_model(model, "Red Neuronal", X_test_scaled, y_test, is_nn=True)
            results["Red Neuronal"] = precision
            
        except Exception as e:
            st.error(f"Error en red neuronal: {str(e)}")
            results["Red Neuronal"] = 0

    # Comparativa final
    if len(results) > 1:
        st.subheader("Resultados Finales")
        df_results = pd.DataFrame({
            "Modelo": results.keys(),
            "Accuracy": results.values()
        }).sort_values("Accuracy", ascending=False)
        
        fig = px.bar(df_results, x='Modelo', y='Accuracy',
                    color='Accuracy', text='Accuracy',
                    color_continuous_scale='Blues')
        fig.update_layout(yaxis_tickformat=".0%")
        fig.update_traces(texttemplate='%{text:.2%}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)