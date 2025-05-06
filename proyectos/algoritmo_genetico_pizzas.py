import random

def run():
    NUM_INGREDIENTES = 8
    POBLACION_SIZE = 20
    TASA_MUTACION = 0.1
    MIN_INGREDIENTES = 3
    MAX_INGREDIENTES = 6
    N_ESTANCAMIENTO = 100

    clientes = [
        ("10101011", 1.5),
        ("11000101", 1.0),
        ("01101100", 0.8),
        ("11100011", 1.2)
    ]

    ingredientes_nombres = ["Pepperoni", "Champiñones", "Jamon", "Pimientos", "Aceitunas", "Cebolla", "Tocino", "Extra Queso"]

    def calcular_max_fitness(clientes, num_ingredientes):
        peso_total = sum(peso for _, peso in clientes)
        preferencias_totales = [0] * num_ingredientes

        for cliente, peso in clientes:
            for i, pref in enumerate(cliente):
                if pref == '1':
                    preferencias_totales[i] += peso

        max_fit = sum(preferencias_totales) / peso_total
        return max_fit

    def generar_pizza():
        while True:
            pizza = "".join(str(random.randint(0, 1)) for _ in range(NUM_INGREDIENTES))
            if MIN_INGREDIENTES <= sum(int(i) for i in pizza) <= MAX_INGREDIENTES:
                return pizza

    def evaluar_pizza(pizza, clientes):
        total_satisfaccion = 0
        for cliente, peso in clientes:
            satisfaccion = sum(int(p) for p, pref in zip(pizza, cliente) if pref == '1')
            total_satisfaccion += satisfaccion * peso
        return total_satisfaccion / sum(peso for _, peso in clientes)

    def encontrar_pizza_ideal(poblacion, clientes, max_fitness):
        return min(poblacion, key=lambda pizza: abs(evaluar_pizza(pizza, clientes) - max_fitness))

    def seleccion_por_torneo(poblacion, clientes):
        seleccionados = []
        for _ in range(POBLACION_SIZE // 2):
            a, b = random.sample(poblacion, 2)
            mejor = a if evaluar_pizza(a, clientes) > evaluar_pizza(b, clientes) else b
            seleccionados.append(mejor)
        return seleccionados

    def cruzar_pizzas(padre, madre):
        punto_corte = random.randint(1, NUM_INGREDIENTES - 1)
        hijo = padre[:punto_corte] + madre[punto_corte:]
        return hijo if MIN_INGREDIENTES <= sum(int(i) for i in hijo) <= MAX_INGREDIENTES else padre

    def mutar_pizza(pizza):
        ingredientes = list(pizza)
        indice_mutacion = random.randint(0, NUM_INGREDIENTES - 1)
        ingredientes[indice_mutacion] = '1' if ingredientes[indice_mutacion] == '0' else '0'
        mutada = "".join(ingredientes)
        return mutada if MIN_INGREDIENTES <= sum(int(i) for i in mutada) <= MAX_INGREDIENTES else pizza

    def binario_a_texto(pizza):
        return [nombre for i, nombre in enumerate(ingredientes_nombres) if pizza[i] == '1']

    poblacion = [generar_pizza() for _ in range(POBLACION_SIZE)]
    max_fitness = calcular_max_fitness(clientes, NUM_INGREDIENTES)

    mejor_pizza = None
    mejor_puntuacion = float('inf')
    generacion = 0
    generaciones_sin_mejora = 0

    while generaciones_sin_mejora < N_ESTANCAMIENTO:
        puntuaciones = [(pizza, evaluar_pizza(pizza, clientes)) for pizza in poblacion]
        mejor_actual = encontrar_pizza_ideal([p for p, _ in puntuaciones], clientes, max_fitness)
        mejor_actual_puntuacion = evaluar_pizza(mejor_actual, clientes)

        if abs(mejor_actual_puntuacion - max_fitness) < abs(mejor_puntuacion - max_fitness):
            mejor_pizza = mejor_actual
            mejor_puntuacion = mejor_actual_puntuacion
            generaciones_sin_mejora = 0
        else:
            generaciones_sin_mejora += 1

        seleccionados = seleccion_por_torneo(poblacion, clientes)

        nueva_poblacion = seleccionados[:]
        while len(nueva_poblacion) < POBLACION_SIZE:
            padre, madre = random.sample(seleccionados, 2)
            hijo = cruzar_pizzas(padre, madre)
            nueva_poblacion.append(hijo)

        for i in range(len(nueva_poblacion)):
            if random.random() < TASA_MUTACION:
                nueva_poblacion[i] = mutar_pizza(nueva_poblacion[i])

        poblacion = nueva_poblacion
        generacion += 1

    ingredientes_mejor_pizza = binario_a_texto(mejor_pizza)

    # Mostrar resultados en Streamlit
    import streamlit as st
    st.subheader("Resultados del Algoritmo Genético")
    st.write(f"Mejor pizza encontrada: `{mejor_pizza}`")
    st.write(f"Puntaje obtenido: `{mejor_puntuacion:.2f}`")
    st.write(f"Fitness objetivo: `{max_fitness:.2f}`")
    st.success("Ingredientes recomendados:")
    st.markdown(", ".join(ingredientes_mejor_pizza))

