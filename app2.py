import random
import streamlit as st

st.title("🍕 Optimizador de Pizzas con Algoritmos Genéticos")

with st.sidebar:
    st.header("Parámetros del Algoritmo")

    num_ingredientes = st.slider("Número de ingredientes", 4, 20, 8)
    poblacion_size = st.number_input("Tamaño de la población", min_value=4, max_value=200, value=20, step=2)
    tasa_mutacion = st.slider("Tasa de mutación", 0.0, 1.0, 0.1)
    min_ingredientes = st.slider("Ingredientes mínimos por pizza", 1, num_ingredientes, 3)
    max_ingredientes = st.slider("Ingredientes máximos por pizza", min_ingredientes, num_ingredientes, 6)
    n_estancamiento = st.number_input("Generaciones sin mejora (estancamiento)", min_value=1, value=100)

    ingredientes_nombres = st.text_area(
        "Nombres de ingredientes (separados por coma)", 
        "Pepperoni,Champiñones,Jamon,Pimientos,Aceitunas,Cebolla,Tocino,Extra Queso"
    ).split(",")

    clientes_raw = st.text_area(
        "Preferencias de clientes (binario y peso, separados por coma)\nEj: 10101011,1.5",
        "10101011,1.5\n11000101,1.0\n01101100,0.8\n11100011,1.2"
    )

if st.button("Ejecutar algoritmo"):
    if len(ingredientes_nombres) != num_ingredientes:
        st.error("El número de nombres de ingredientes debe coincidir con el número de ingredientes definido.")
    else:
        try:
            clientes = []
            for linea in clientes_raw.strip().split("\n"):
                binario, peso = linea.strip().split(",")
                if len(binario) != num_ingredientes:
                    st.error(f"Error: la cadena {binario} no tiene {num_ingredientes} dígitos.")
                    st.stop()
                clientes.append((binario, float(peso)))

            peso_total = sum(p for _, p in clientes)
            preferencias_totales = [0] * num_ingredientes
            for cliente, peso in clientes:
                for i, pref in enumerate(cliente):
                    if pref == '1':
                        preferencias_totales[i] += peso
            max_fitness = sum(preferencias_totales) / peso_total

            def generar_pizza():
                while True:
                    pizza = "".join(str(random.randint(0, 1)) for _ in range(num_ingredientes))
                    if min_ingredientes <= sum(int(i) for i in pizza) <= max_ingredientes:
                        return pizza

            def evaluar_pizza(pizza):
                total = 0
                for cliente, peso in clientes:
                    satisfaccion = sum(int(p) for p, pref in zip(pizza, cliente) if pref == '1')
                    total += satisfaccion * peso
                return total / peso_total

            def encontrar_pizza_ideal(poblacion):
                return min(poblacion, key=lambda p: abs(evaluar_pizza(p) - max_fitness))

            def seleccion_por_torneo(poblacion):
                seleccionados = []
                for _ in range(poblacion_size // 2):
                    a, b = random.sample(poblacion, 2)
                    mejor = a if evaluar_pizza(a) > evaluar_pizza(b) else b
                    seleccionados.append(mejor)
                return seleccionados

            def cruzar_pizzas(p1, p2):
                punto = random.randint(1, num_ingredientes - 1)
                hijo = p1[:punto] + p2[punto:]
                return hijo if min_ingredientes <= sum(int(i) for i in hijo) <= max_ingredientes else p1

            def mutar_pizza(pizza):
                ingredientes = list(pizza)
                i = random.randint(0, num_ingredientes - 1)
                ingredientes[i] = '1' if ingredientes[i] == '0' else '0'
                mutada = "".join(ingredientes)
                return mutada if min_ingredientes <= sum(int(i) for i in mutada) <= max_ingredientes else pizza

            def binario_a_ingredientes(pizza):
                return [nombre for i, nombre in enumerate(ingredientes_nombres) if pizza[i] == '1']

            poblacion = [generar_pizza() for _ in range(poblacion_size)]
            mejor_pizza = None
            mejor_score = float('inf')
            sin_mejora = 0

            while sin_mejora < n_estancamiento:
                puntuaciones = [(p, evaluar_pizza(p)) for p in poblacion]
                actual = encontrar_pizza_ideal([p for p, _ in puntuaciones])
                score_actual = evaluar_pizza(actual)

                if abs(score_actual - max_fitness) < abs(mejor_score - max_fitness):
                    mejor_pizza = actual
                    mejor_score = score_actual
                    sin_mejora = 0
                else:
                    sin_mejora += 1

                seleccionados = seleccion_por_torneo(poblacion)
                nueva = seleccionados[:]
                while len(nueva) < poblacion_size:
                    padre, madre = random.sample(seleccionados, 2)
                    hijo = cruzar_pizzas(padre, madre)
                    nueva.append(hijo)

                for i in range(len(nueva)):
                    if random.random() < tasa_mutacion:
                        nueva[i] = mutar_pizza(nueva[i])

                poblacion = nueva

            st.subheader("🍕 Resultados del Algoritmo Genético")
            st.write(f"Mejor pizza encontrada: `{mejor_pizza}`")
            st.write(f"Puntaje obtenido: `{mejor_score:.2f}`")
            st.write(f"Fitness objetivo: `{max_fitness:.2f}`")
            st.success("Ingredientes recomendados:")
            st.markdown(", ".join(binario_a_ingredientes(mejor_pizza)))

        except Exception as e:
            st.error(f"Ocurrió un error: {e}")
