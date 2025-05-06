import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.express as px
import plotly.tools as tls
import plotly.figure_factory as ff
import streamlit as st

from warnings import filterwarnings
filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation,Dropout

# Modelling Libraries
from sklearn.linear_model import LogisticRegression,RidgeClassifier,SGDClassifier,PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC,LinearSVC,NuSVC
from sklearn.neighbors import KNeighborsClassifier,NearestCentroid
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB,BernoulliNB
from sklearn.ensemble import VotingClassifier

# Evaluation & CV Libraries
from sklearn.metrics import precision_score,accuracy_score
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV,RepeatedStratifiedKFold

def run():
    data = pd.read_csv('water_potability.xls')

    data.head()

    fig = msno.matrix(data,color=(0,0.5,0.5))

    data.isnull().sum()

    fig, ax = plt.subplots(figsize = (18,18))
    sns.heatmap(data.corr(), ax = ax, annot = True)

    colors_blue = ["#132C33", "#264D58", '#17869E', '#51C4D3', '#B4DBE9']
    colors_dark = ["#1F1F1F", "#313131", '#636363', '#AEAEAE', '#DADADA']
    colors_green = ['#01411C','#4B6F44','#4F7942','#74C365','#D0F0C0']
    sns.palplot(colors_blue)
    sns.palplot(colors_green)
    sns.palplot(colors_dark)

    """# Visualizacion

    **Hardenss of water**: La definición simple de dureza del agua es la cantidad de calcio y magnesio disueltos en el agua. El agua dura es rica en minerales disueltos, principalmente calcio y magnesio. Probablemente hayas sentido los efectos del agua dura, literalmente, la última vez que te lavaste las manos. Dependiendo de la dureza del agua, después de usar jabón para lavarte, podrías haber sentido como si quedara una película de residuos en tus manos. En agua dura, el jabón reacciona con el calcio (que es relativamente alto en agua dura) para formar "escamas de jabón". Al usar agua dura, se necesita más jabón o detergente para limpiar las cosas, ya sean tus manos, cabello o ropa.
    """

    fig = px.histogram(data,x='Hardness',y=Counter(data['Hardness']),color='Potability',template='plotly_white',
                    marginal='box',opacity=0.7,nbins=100,color_discrete_sequence=[colors_green[3],colors_blue[3]],
                    barmode='group',histfunc='count')

    fig.add_vline(x=151, line_width=1, line_color=colors_dark[1],line_dash='dot',opacity=0.7)
    fig.add_vline(x=301, line_width=1, line_color=colors_dark[1],line_dash='dot',opacity=0.7)
    fig.add_vline(x=76, line_width=1, line_color=colors_dark[1],line_dash='dot',opacity=0.7)

    fig.add_annotation(text='<76 mg/L is<br> considered soft',x=40,y=130,showarrow=False,font_size=9)
    fig.add_annotation(text='Between 76 and 150<br> (mg/L) is<br>moderately hard',x=113,y=130,showarrow=False,font_size=9)
    fig.add_annotation(text='Between 151 and 300 (mg/L)<br> is considered hard',x=250,y=130,showarrow=False,font_size=9)
    fig.add_annotation(text='>300 mg/L is<br> considered very hard',x=340,y=130,showarrow=False,font_size=9)

    fig.update_layout(
        font_family='monospace',
        title=dict(text='Hardness Distribution',x=0.53,y=0.95,
                font=dict(color=colors_dark[2],size=20)),
        xaxis_title_text='Hardness (mg/L)',
        yaxis_title_text='Count',
        legend=dict(x=1,y=0.96,bordercolor=colors_dark[4],borderwidth=0,tracegroupgap=5),
        bargap=0.3,
    )
    st.pyplot(fig)

    """**pH level:** El pH del agua es una medida del equilibrio ácido-base y, en la mayoría de las aguas naturales, está controlado por el sistema de equilibrio dióxido de carbono–bicarbonato–carbonato. Una mayor concentración de dióxido de carbono disminuirá el pH, mientras que una disminución hará que aumente. La temperatura también afectará el equilibrio y el pH. En el agua pura, el pH disminuye aproximadamente 0.45 cuando la temperatura aumenta 25 °C. En agua con una capacidad tampón proporcionada por iones de bicarbonato, carbonato y oxidrilo, este efecto de temperatura se modifica (APHA, 1989). El pH de la mayoría del agua potable está dentro del rango de 6.5–8.5. Las aguas naturales pueden tener un pH más bajo debido, por ejemplo, a la lluvia ácida, o un pH más alto en áreas con presencia de piedra caliza."""

    fig = px.histogram(data,x='ph',y=Counter(data['ph']),color='Potability',template='plotly_white',
                    marginal='box',opacity=0.7,nbins=100,color_discrete_sequence=[colors_green[3],colors_blue[3]],
                    barmode='group',histfunc='count')

    fig.add_vline(x=7, line_width=1, line_color=colors_dark[1],line_dash='dot',opacity=0.7)

    fig.add_annotation(text='<7 is Acidic',x=4,y=70,showarrow=False,font_size=10)
    fig.add_annotation(text='>7 is Basic',x=10,y=70,showarrow=False,font_size=10)


    fig.update_layout(
        font_family='monospace',
        title=dict(text='pH Level Distribution',x=0.5,y=0.95,
                font=dict(color=colors_dark[2],size=20)),
        xaxis_title_text='pH Level',
        yaxis_title_text='Count',
        legend=dict(x=1,y=0.96,bordercolor=colors_dark[4],borderwidth=0,tracegroupgap=5),
        bargap=0.3,
    )
    st.pyplot(fig)

    """**TDS**: TDS significa concentración de partículas o sólidos disueltos en el agua. TDS está compuesto por sales inorgánicas como calcio, magnesio, cloruros, sulfatos, bicarbonatos, entre otros, junto con muchos otros compuestos inorgánicos que se disuelven fácilmente en el agua."""

    fig = px.histogram(data,x='Solids',y=Counter(data['Solids']),color='Potability',template='plotly_white',
                    marginal='box',opacity=0.7,nbins=100,color_discrete_sequence=[colors_green[3],colors_blue[3]],
                    barmode='group',histfunc='count')

    fig.update_layout(
        font_family='monospace',
        title=dict(text='Distribution Of Total Dissolved Solids',x=0.5,y=0.95,
                font=dict(color=colors_dark[2],size=20)),
        xaxis_title_text='Dissolved Solids (ppm)',
        yaxis_title_text='Count',
        legend=dict(x=1,y=0.96,bordercolor=colors_dark[4],borderwidth=0,tracegroupgap=5),
        bargap=0.3,
    )
    st.pyplot(fig)

    """**Chloramines**: Las cloraminas (también conocidas como desinfección secundaria) son desinfectantes utilizados para tratar el agua potable y:

    * Se forman comúnmente cuando se añade amoníaco al cloro para tratar el agua potable.
    * Proporcionan una desinfección más duradera a medida que el agua se mueve a través de las tuberías hacia los consumidores.

    Las cloraminas han sido utilizadas por las empresas de servicios de agua desde la década de 1930.
    """

    fig = px.histogram(data,x='Chloramines',y=Counter(data['Chloramines']),color='Potability',template='plotly_white',
                    marginal='box',opacity=0.7,nbins=100,color_discrete_sequence=[colors_green[3],colors_blue[3]],
                    barmode='group',histfunc='count')

    fig.add_vline(x=4, line_width=1, line_color=colors_dark[1],line_dash='dot',opacity=0.7)

    fig.add_annotation(text='<4 ppm is considered<br> safe for drinking',x=1.8,y=90,showarrow=False)

    fig.update_layout(
        font_family='monospace',
        title=dict(text='Chloramines Distribution',x=0.53,y=0.95,
                font=dict(color=colors_dark[2],size=20)),
        xaxis_title_text='Chloramines (ppm)',
        yaxis_title_text='Count',
        legend=dict(x=1,y=0.96,bordercolor=colors_dark[4],borderwidth=0,tracegroupgap=5),
        bargap=0.3,
    )
    st.pyplot(fig)

    """**Sulfate**: El sulfato (SO4) se puede encontrar en casi toda el agua natural. El origen de la mayoría de los compuestos de sulfato es la oxidación de minerales de sulfito, la presencia de lutitas o los desechos industriales.
    El sulfato es uno de los principales componentes disueltos en la lluvia. Altas concentraciones de sulfato en el agua que bebemos pueden tener un efecto laxante cuando se combinan con calcio y magnesio, los dos componentes más comunes de la dureza.
    """

    fig = px.histogram(data,x='Sulfate',y=Counter(data['Sulfate']),color='Potability',template='plotly_white',
                    marginal='box',opacity=0.7,nbins=100,color_discrete_sequence=[colors_green[3],colors_blue[3]],
                    barmode='group',histfunc='count')

    fig.add_vline(x=250, line_width=1, line_color=colors_dark[1],line_dash='dot',opacity=0.7)

    fig.add_annotation(text='<250 mg/L is considered<br> safe for drinking',x=175,y=90,showarrow=False)

    fig.update_layout(
        font_family='monospace',
        title=dict(text='Sulfate Distribution',x=0.53,y=0.95,
                font=dict(color=colors_dark[2],size=20)),
        xaxis_title_text='Sulfate (mg/L)',
        yaxis_title_text='Count',
        legend=dict(x=1,y=0.96,bordercolor=colors_dark[4],borderwidth=0,tracegroupgap=5),
        bargap=0.3,
    )
    st.pyplot(fig)

    """**Conductivity**: La conductividad es una medida de la capacidad del agua para transmitir una corriente eléctrica. Dado que las sales disueltas y otros químicos inorgánicos conducen la corriente eléctrica, la conductividad aumenta a medida que aumenta la salinidad. Los compuestos orgánicos como el aceite no conducen muy bien la corriente eléctrica y, por lo tanto, tienen una baja conductividad cuando están en el agua. La conductividad también se ve afectada por la temperatura: cuanto más caliente es el agua, mayor es la conductividad.

    """

    fig = px.histogram(data,x='Conductivity',y=Counter(data['Conductivity']),color='Potability',template='plotly_white',
                    marginal='box',opacity=0.7,nbins=100,color_discrete_sequence=[colors_green[3],colors_blue[3]],
                    barmode='group',histfunc='count')

    fig.add_annotation(text='The Conductivity range <br> is safe for both (200-800),<br> Potable and Non-Potable water',
                    x=600,y=90,showarrow=False)

    fig.update_layout(
        font_family='monospace',
        title=dict(text='Conductivity Distribution',x=0.5,y=0.95,
                font=dict(color=colors_dark[2],size=20)),
        xaxis_title_text='Conductivity (μS/cm)',
        yaxis_title_text='Count',
        legend=dict(x=1,y=0.96,bordercolor=colors_dark[4],borderwidth=0,tracegroupgap=5),
        bargap=0.3,
    )
    st.pyplot(fig)

    """**Organic Carbon**: Los contaminantes orgánicos (sustancias orgánicas naturales, insecticidas, herbicidas y otros químicos agrícolas) ingresan a las vías fluviales a través del agua de escorrentía de las lluvias. Las aguas residuales domésticas e industriales también aportan contaminantes orgánicos en diferentes cantidades. Como resultado de derrames accidentales o fugas, los desechos orgánicos industriales pueden llegar a los ríos y arroyos. Algunos de estos contaminantes pueden no ser completamente eliminados por los procesos de tratamiento, por lo que podrían convertirse en un problema para las fuentes de agua potable. Es importante conocer el contenido orgánico en una vía fluvial."""

    fig = px.histogram(data,x='Organic_carbon',y=Counter(data['Organic_carbon']),color='Potability',template='plotly_white',
                    marginal='box',opacity=0.7,nbins=100,color_discrete_sequence=[colors_green[3],colors_blue[3]],
                    barmode='group',histfunc='count')

    fig.add_vline(x=10, line_width=1, line_color=colors_dark[1],line_dash='dot',opacity=0.7)

    fig.add_annotation(text='Typical Organic Carbon<br> level is upto 10 ppm',x=5.3,y=110,showarrow=False)

    fig.update_layout(
        font_family='monospace',
        title=dict(text='Organic Carbon Distribution',x=0.5,y=0.95,
                font=dict(color=colors_dark[2],size=20)),
        xaxis_title_text='Organic Carbon (ppm)',
        yaxis_title_text='Count',
        legend=dict(x=1,y=0.96,bordercolor=colors_dark[4],borderwidth=0,tracegroupgap=5),
        bargap=0.3,
    )
    st.pyplot(fig)

    """**Trihalomethanes**: Los trihalometanos (THMs) son el resultado de una reacción entre el cloro utilizado para desinfectar el agua del grifo y la materia orgánica natural en el agua. En niveles elevados, los THMs se han asociado con efectos negativos para la salud, como el cáncer y resultados adversos en la reproducción."""

    fig = px.histogram(data,x='Trihalomethanes',y=Counter(data['Trihalomethanes']),color='Potability',template='plotly_white',
                    marginal='box',opacity=0.7,nbins=100,color_discrete_sequence=[colors_green[3],colors_blue[3]],
                    barmode='group',histfunc='count')

    fig.add_vline(x=80, line_width=1, line_color=colors_dark[1],line_dash='dot',opacity=0.7)

    fig.add_annotation(text='Upper limit of Trihalomethanes<br> level is 80 μg/L',x=115,y=90,showarrow=False)

    fig.update_layout(
        font_family='monospace',
        title=dict(text='Trihalomethanes Distribution',x=0.5,y=0.95,
                font=dict(color=colors_dark[2],size=20)),
        xaxis_title_text='Trihalomethanes (μg/L)',
        yaxis_title_text='Count',
        legend=dict(x=1,y=0.96,bordercolor=colors_dark[4],borderwidth=0,tracegroupgap=5),
        bargap=0.3,
    )
    st.pyplot(fig)

    """**Turbidity**: La turbidez es la medida de la claridad relativa de un líquido. Es una característica óptica del agua y es una medida de la cantidad de luz que se dispersa por el material en el agua cuando se ilumina una muestra de agua. Cuanto mayor sea la intensidad de la luz dispersada, mayor será la turbidez. El material que provoca que el agua sea turbia incluye arcilla, limo, materia inorgánica y orgánica muy pequeña, algas, compuestos orgánicos disueltos coloreados, y plancton y otros organismos microscópicos.

    """

    fig = px.histogram(data,x='Turbidity',y=Counter(data['Turbidity']),color='Potability',template='plotly_white',
                    marginal='box',opacity=0.7,nbins=100,color_discrete_sequence=[colors_green[3],colors_blue[3]],
                    barmode='group',histfunc='count')

    fig.add_vline(x=5, line_width=1, line_color=colors_dark[1],line_dash='dot',opacity=0.7)

    fig.add_annotation(text='<5 NTU Turbidity is<br> considered safe',x=6,y=90,showarrow=False)

    fig.update_layout(
        font_family='monospace',
        title=dict(text='Turbidity Distribution',x=0.5,y=0.95,
                font=dict(color=colors_dark[2],size=20)),
        xaxis_title_text='Turbidity (NTU)',
        yaxis_title_text='Count',
        legend=dict(x=1,y=0.96,bordercolor=colors_dark[4],borderwidth=0,tracegroupgap=5),
        bargap=0.3,
    )
    st.pyplot(fig)

    trace = go.Pie(labels = ['Potable', 'Not Potable'], values = data['Potability'].value_counts(),
                textfont=dict(size=15), opacity = 0.8,
                marker=dict(colors=['lightskyblue','gold'],
                            line=dict(color='#000000', width=1.5)))


    layout = dict(title =  'Distribution of Drinkable Water')

    fig = dict(data = [trace], layout=layout)
    py.iplot(fig)

    plt.figure(figsize = (15,10), tight_layout = True)

    for i, feature in enumerate(data.columns):
        if feature != 'Potability':

            plt.subplot(3,3,i+1)
            sns.histplot(data = data, x =feature, palette = 'mako', hue = 'Potability',alpha = 0.5, element="step",hue_order=[1,0] )

    sns.pairplot(data = data,hue = 'Potability',palette='mako_r', corner=True)

    data[data['Potability']==0].describe()

    data[data['Potability']==1].describe()

    data['ph'].fillna(value=data['ph'].median(),inplace=True)
    data['Sulfate'].fillna(value=data['Sulfate'].median(),inplace=True)
    data['Trihalomethanes'].fillna(value=data['Trihalomethanes'].median(),inplace=True)

    data.isnull().sum()

    """## Data Splitting"""

    X = data.drop('Potability',axis=1).values
    y = data['Potability'].values

    from imblearn.over_sampling import SMOTE
    smote = SMOTE(sampling_strategy='auto')  # Generar ejemplos sintéticos
    X_resampled, y_resampled = smote.fit_resample(X, y)  # Nuevo conjunto de datos equilibrado
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    """## Model Machine"""

    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix, classification_report
    from sklearn.model_selection import train_test_split
    import pandas as pd

    # Modelos a probar
    models = [
        ("LR", LogisticRegression(max_iter=1000)),
        ("SVC", SVC()),
        ('KNN', KNeighborsClassifier(n_neighbors=10)),
        ("DTC", DecisionTreeClassifier()),
        ("GNB", GaussianNB()),
        ("SGDC", SGDClassifier()),
        ("Perc", Perceptron()),
        ("NC", NearestCentroid()),
        ("Ridge", RidgeClassifier()),
        ("NuSVC", NuSVC()),
        ("BNB", BernoulliNB()),
        ('RF', RandomForestClassifier()),
        ('ADA', AdaBoostClassifier()),
        ('XGB', GradientBoostingClassifier()),
        ('PAC', PassiveAggressiveClassifier())
    ]

    # Lista para almacenar resultados y nombres
    results = []
    names = []
    finalResults = []
    confusion_matrices = {}
    classification_reports = {}

    # Entrenar y evaluar cada modelo
    for name, model in models:
        # Entrenamiento
        model.fit(X_train, y_train)

        # Predicción
        model_results = model.predict(X_test)

        # Evaluación de precisión
        score = precision_score(y_test, model_results, average='macro')
        results.append(score)
        names.append(name)
        finalResults.append((name, score))

        # Matriz de confusión y reporte de clasificación
        cm = confusion_matrix(y_test, model_results)
        confusion_matrices[name] = cm  # Guardar en un diccionario

        report = classification_report(y_test, model_results, output_dict=True)  # Guardar como dict
        classification_reports[name] = report

    # Ordenar resultados finales
    finalResults.sort(key=lambda k: k[1], reverse=True)

    from sklearn.metrics import confusion_matrix
    # Función para plotear la matriz de confusión
    def plot_confusion_matrix(name, class_labels):
        cm = confusion_matrices[name]  # Obtener la matriz de confusión por nombre del modelo
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
        plt.title(f'Confusion Matrix for {name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

    class_labels = np.unique(y_test)

    # Ejemplo de uso de la función
    plot_confusion_matrix("SVC", class_labels) # Plotea la matriz de confusión para el modelo "LR"

    # Mostrar resultados finales
    for result in finalResults:
        st.write(f"{result[0]}: {result[1]}")

    """## Model Neural Network

    ### Model 1
    """

    def evaluate_model(model, X_test, y_test, threshold=0.5):
        # Realizar predicciones y convertirlas a etiquetas binarias según el umbral
        y_pred = model.predict(X_test)
        y_pred = [1 if y >= threshold else 0 for y in y_pred]

        # Obtener el classification_report
        report = classification_report(y_test, y_pred)

        # Generar la matriz de confusión
        cm = confusion_matrix(y_test, y_pred)

        # Calcular el accuracy usando la matriz de confusión
        accuracy = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])

        return report, cm, accuracy  # Devolver el reporte, la matriz y el accuracy

    def model_base():
        model = Sequential()  # Inicialización de la ANN

        # Capas densas con diferentes unidades y activaciones
        model.add(Dense(units=16, kernel_initializer='uniform', activation='relu'))  # Primera capa
        model.add(Dense(units=9, kernel_initializer='uniform', activation='relu'))  # Segunda capa
        model.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))  # Capa de salida

        # Compilación del modelo
        model.compile(optimizer='adam', loss='binary_crossentropy')  # Compilar con Adam y pérdida binaria

        return model  # Devolver el modelo construido

    model_1 = model_base()
    model_1.fit(x=X_train, y=y_train, epochs=300, validation_data=(X_test, y_test), verbose=0)

    model_loss = pd.DataFrame(model_1.history.history)
    model_loss.plot()

    report, cm, accuracy = evaluate_model(model_1, X_test, y_test, threshold=0.5)

    # Imprimir resultados
    st.write("Classification Report:\n", report)
    st.write("Confusion Matrix:\n", cm)
    st.write("Accuracy: " + str(accuracy * 100) + "%")

    """### Model 2 (Dropout and optimizar):"""

    from tensorflow.keras.optimizers import Adam

    def model_Dropout():
        model = Sequential()  # Inicialización de la ANN
        model.add(Dropout(0.2))  # Reducir Dropout para mejorar el aprendizaje
        # Ajustar capas y regularización
        model.add(Dense(units=16, activation='relu'))

        model.add(Dense(units=5, activation='relu'))
        model.add(Dense(units=1, activation='sigmoid'))  # Capa de salida para clasificación binaria

        learning_rate = 0.001  # Reducir la tasa de aprendizaje para ajustes más finos
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='binary_crossentropy',  metrics=['accuracy'])

        return model  # Devolver el modelo construido

    model_2 = model_Dropout()
    model_2.fit(x=X_train, y=y_train, epochs=350, validation_data=(X_test, y_test), verbose=0)

    model_loss = pd.DataFrame(model_2.history.history)
    model_loss.plot()

    report_2, cm_2, accuracy_2 = evaluate_model(model_2, X_test, y_test, threshold=0.5)

    # Imprimir resultados
    st.write("Classification Report:\n", report_2)
    st.write("Confusion Matrix:\n", cm_2)
    st.write("Accuracy: " + str(accuracy_2 * 100) + "%")

    """### Model 3 (Menor cantidad de neuronas)"""

    def model_minor():
        model = Sequential()
        model.add(Dropout(0.2))  # Para reducir el sobreajuste
        model.add(Dense(units=16,activation='relu'))

        model.add(Dropout(0.2))  # Para reducir el sobreajuste

        model.add(Dense(units=1,activation='sigmoid'))
        learning_rate=0.001
        # Compilación del modelo
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='binary_crossentropy',  metrics=['accuracy'])  # Compilar con Adam y pérdida binaria
        return model

    model_3 = model_minor()
    model_3.fit(x=X_train, y=y_train, epochs=350, validation_data=(X_test, y_test), verbose=0)

    model_loss = pd.DataFrame(model_3.history.history)
    model_loss.plot()

    report_3, cm_3, accuracy_3 = evaluate_model(model_3, X_test, y_test, threshold=0.5)

    # Imprimir resultados
    st.write("Classification Report:\n", report_3)
    st.write("Confusion Matrix:\n", cm_3)
    st.write("Accuracy: " + str(accuracy_3 * 100) + "%")