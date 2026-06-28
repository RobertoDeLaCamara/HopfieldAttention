# 6. Subspace Attention: De la Atención Estándar al Espacio Subdimensional

> *Una vez establecida la equivalencia entre Hopfield Networks y Attention, surge la pregunta: ¿qué otras formas de atención — o de actualización Hopfield — podemos diseñar? El AdvancedHopfieldModel de mi solver de shortest path apunta en una dirección: atención en subespacios.*

## El Problema de Escalabilidad

La atención estándar tiene complejidad $O(n^2)$ en el número de tokens. Para secuencias largas, esto es prohibitivo. Pero hay otro problema más sutil: en un grafo de 5000 nodos, la mayoría de las aristas no existen (el grafo es sparse). La matriz de adyacencia completa $n \times n$ desperdicia memoria y computación.

Con Hopfield SPP, abordé este problema con **tensores sparse**: solo almacenar las aristas existentes ($E$ en lugar de $n^2$), reduciendo la memoria de $O(n^2)$ a $O(E)$.

## Subespacios en Hopfield SPP

El `AdvancedHopfieldModel` en mi solver de shortest path implementa varias técnicas que tienen paralelos directos en atención:

### Tensores Sparse

```python
# Solo almacenar aristas existentes en lugar de matriz completa
# Memoria: O(E) en lugar de O(n²)
sparse_indices = tf.constant(edge_list, dtype=tf.int64)  # shape (E, 2)
sparse_values = tf.Variable(initial_values, shape=(E,))
```

### Mecanismo de Atención

El modelo avanzado incluye un mecanismo de atención que pondera la influencia de nodos vecinos según su relevancia:

```python
attention_weights = tf.nn.softmax(
    beta * tf.reduce_sum(node_features * target_features, axis=-1)
)
update = tf.reduce_sum(attention_weights * neighbor_values, axis=0)
```

### Hiperparámetros Adaptativos

En lugar de temperatura fija, el modelo ajusta dinámicamente:

- **Temperatura**: controla la nitidez de la atención/activación
- **Tasa de aprendizaje**: se adapta según la convergencia
- **Número de reinicios**: se incrementa para problemas difíciles

### Beam Search para Extracción

El modelo avanzado usa beam search (similar a decoding en transformers) para extraer la ruta óptima de la matriz de activación:

```python
# Beam search: mantener top-k caminos parciales
beam = [(source, [source], 0.0)]
for step in range(max_steps):
    candidates = []
    for node, path, score in beam:
        for next_node in adjacency[node]:
            candidates.append((next_node, path + [next_node], score + cost))
    beam = sorted(candidates, key=lambda x: x[2])[:beam_width]
```

## De Hopfield SPP a Atención en Subespacios

Las técnicas que implementé para escalar Hopfield a grafos grandes son directamente aplicables a atención en transformers:

| Técnica Hopfield SPP | Equivalente en Atención |
|---------------------|------------------------|
| Tensores sparse | Atención sparse (Reformer, BigBird) |
| Atención local por vecindad | Atención de ventana deslizante |
| Beam search | Decoding autorregresivo |
| Hiperparámetros adaptativos | Temperature scaling en inference |
| Energía por subgrafo | Atención por parches (patch-based) |

## Subspace Attention como Principio General

La idea central que emerge es: **no toda atención necesita operar en el espacio completo**. Podemos restringir la atención a subespacios relevantes:

- **Subespacio de tokens**: atención local, ventanas, vecindades
- **Subespacio de características**: atención factorizada (low-rank), proyecciones
- **Subespacio temporal**: atención en ventanas de tiempo, memoria comprimida

Cada una de estas es una Hopfield Network que opera en un subespacio del espacio completo. Y cada una hereda las propiedades de convergencia y estabilidad de la teoría de Hopfield.

---

**Siguiente: [Capítulo 7 — Hacia el Futuro: Hopfield Layers en Deep Learning Moderno](07_future.md)**
