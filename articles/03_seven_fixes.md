# 3. Siete Correcciones que Transformaron un Hopfield Roto en un Solver Robusto

> *Veinticinco años después de mi tesis, volví a implementar el solver de shortest path con Hopfield. El resultado: una fiabilidad del 40-60%. Este capítulo documenta las siete correcciones específicas que elevaron esa cifra al 95-100%.*

## El Problema de Raíz: Restricciones Incorrectas

El modelo original codificaba las restricciones de un **ciclo Hamiltoniano** (el problema del viajante, TSP) en lugar de las de **conservación de flujo** (Shortest Path). Esto es fundamentalmente incorrecto para el SPP.

Las restricciones correctas para Shortest Path son:

- **Origen**: una unidad más de flujo sale que entra
- **Destino**: una unidad más de flujo entra que sale
- **Intermedios**: el flujo entrante iguala al saliente

Las restricciones de TSP exigen que cada nodo tenga exactamente una arista entrante y una saliente — un ciclo completo. El modelo buscaba soluciones en el espacio de solución equivocado.

La corrección (`train_model_improved.py`):

```python
def energy(self, source, destination, temperature=0.5):
    x = tf.nn.sigmoid(self.logits / temperature) * self.valid_arcs

    flow_penalty = 0.0
    for i in range(self.n):
        out_flow = tf.reduce_sum(x[i, :])
        in_flow = tf.reduce_sum(x[:, i])

        if i == source:
            flow_penalty += tf.square(out_flow - in_flow - 1.0)
        elif i == destination:
            flow_penalty += tf.square(in_flow - out_flow - 1.0)
        else:
            flow_penalty += tf.square(out_flow - in_flow)
```

Este único cambio es responsable del salto del 40% al 95% de fiabilidad.

## Corrección 1: Cero Entrenamiento Offline

El modelo original ejecutaba 1000 épocas de entrenamiento offline antes de cualquier consulta. Esto es computación inútil: los pesos de Hopfield son la matriz de costos, no parámetros aprendibles.

```python
# Original: 1000 épocas de nada
modelo = HopfieldModel(n, distance_matrix)
modelo.fit(dummy_data, epochs=1000)

# Mejorado: cero entrenamiento offline
modelo = ImprovedHopfieldModel(n, distance_matrix)
# Sin training. La optimización ocurre en tiempo de consulta.
```

Ahorro: 30-60 segundos de computación sin sentido.

## Corrección 2: Optimizador Fresco por Consulta

El optimizador Adam mantiene vectores de momento (estimaciones de primer y segundo orden). Reutilizar el mismo optimizador entre consultas conserva momento de caminos anteriores, contaminando la búsqueda actual.

```python
# Original (contaminación de estado):
self.optimizer = tf.optimizers.Adam(learning_rate=0.01)

# Mejorado:
def optimize(self, source, destination, ...):
    optimizer = tf.optimizers.Adam(learning_rate=0.02)  # Fresco
    self.logits.assign(tf.random.normal(...))           # Logits frescos
```

## Corrección 3: Fallback a Dijkstra

Ningún algoritmo heurístico es 100% fiable. El nuevo modelo ejecuta Dijkstra como validación después de la optimización Hopfield. Si Hopfield produce una solución >5% peor que la óptima conocida, se usa Dijkstra.

```python
if best_path is None or best_cost > dijkstra_cost * 1.05:
    return dijkstra_path  # Solución garantizada
return best_path  # Solución Hopfield (dentro del 5% de la óptima)
```

## Corrección 4: Parada Temprana

El modelo original siempre ejecutaba el número completo de iteraciones. El nuevo detiene la ejecución cuando la energía se estabiliza (20 iteraciones sin mejora >1e-6). Reducción del 40-60% en tiempo de consulta.

## Corrección 5: Múltiples Reinicios

El paisaje de energía de Hopfield tiene mínimos locales. Ejecutar tres reinicios desde puntos de partida aleatorios y quedarse con el mejor resultado mejora significativamente la calidad.

## Corrección 6: Extracción BFS vs. Argmax

La extracción greedy (argmax) se queda en callejones sin salida. La nueva implementación usa BFS sobre el conjunto de aristas con activación >0.5, ordenadas por peso. Si hay un camino en las aristas activas, BFS lo encuentra.

## Corrección 7: Caché de Modelo

La API original cargaba el modelo de disco en cada petición (~2s). La nueva usa un caché en memoria. Primera llamada: ~2-3s. Siguientes: ~50-100ms.

## Resultado

| Métrica | Original | Mejorado |
|---------|----------|----------|
| Tiempo de consulta | 5-10s | 1-3s |
| Soluciones óptimas | 40-60% | 95-100% |
| Fiabilidad | 80-90% | 100% |
| Máximo tamaño grafo | ~100 | ~500 (5000+ con sparse) |

## Lección General

Siete correcciones, de las cuales solo una — las restricciones correctas — fue responsable de la mayor parte de la mejora. Las otras seis fueron _disciplina de ingeniería_: caché, estado fresco, parada temprana, múltiples intentos, fallback.

El tesis de 1998 no falló por la idea. Falló por la implementación. Veinticinco años después, con mejores herramientas y más experiencia, la idea funciona.

---

**Siguiente: [Capítulo 4 — Modern Hopfield Networks: La Revolución Silenciosa](04_modern_hopfield.md)**
