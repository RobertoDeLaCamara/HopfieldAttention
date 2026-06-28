# 1. La Hopfield Original: Energía, Estabilidad y Puntos Fijos

> *Antes de que el deep learning fuera deep, antes de que "atención" fuera un concepto en machine learning, existía una red recurrente simple que convergía a puntos fijos minimizando una función de energía.*

## El Modelo

En 1982, John Hopfield publicó un artículo que revitalizó las redes neuronales en un momento en que el campo estaba estancado. Su idea era elegante: una red de neuronas binarias conectadas entre sí que evolucionaba hacia estados estables — atractores — minimizando una función de energía.

Una Hopfield Network es una red recurrente totalmente conectada donde cada neurona está conectada a todas las demás (pero no a sí misma). Las neuronas son binarias (activadas o desactivadas) y la red actualiza su estado de forma asíncrona, una neurona a la vez.

## La Función de Energía

La contribución clave de Hopfield fue definir una función de energía (o Lyapunov) que siempre decrece con cada actualización:

$$E = -\frac{1}{2} \sum_{i,j} w_{ij} s_i s_j + \sum_i \theta_i s_i$$

Donde:
- $w_{ij}$ es el peso de la conexión entre las neuronas $i$ y $j$
- $s_i$ es el estado (0 o 1) de la neurona $i$
- $\theta_i$ es el umbral de la neurona $i$

Cada actualización de estado garantiza que la energía nunca aumente. Como la energía está acotada inferiormente, la red converge inevitablemente a un mínimo local — un atractor.

## Atractores como Memoria

Las Hopfield Networks se popularizaron como **memorias asociativas**: se almacenan patrones como mínimos de energía, y la red recupera el patrón completo a partir de una versión parcial o ruidosa.

Si entrenamos la red con imágenes de dígitos, cada dígito se convierte en un atractor. Al presentar un dígito ruidoso, la red converge al patrón más cercano — "recupera" la imagen limpia.

## Dos Limitaciones Fundamentales

1. **Capacidad limitada**: Una Hopfield Network con $n$ neuronas puede almacenar aproximadamente $0.15n$ patrones antes de que los atractores espurios comiencen a dominar.

2. **Tendencia a mínimos espurios**: La red converge al *mínimo local* más cercano, que no siempre corresponde a un patrón almacenado. Especialmente problemático cuando se usa para optimización combinatoria, como veremos en el siguiente capítulo.

## Por Qué Esto Importa

El mecanismo de actualización de Hopfield — calcular una suma ponderada de entradas y aplicar una no-linealidad — es estructuralmente idéntico al mecanismo de atención que define los Transformers. La diferencia está en la no-linealidad (signo/sigmoides vs. softmax) y en cómo se estructuran las conexiones.

Pero estamos adelantando la historia. Primero, veamos cómo aplicar Hopfield a la optimización combinatoria.

---

**Siguiente: [Capítulo 2 — Mi tesis de 1998: Hopfield para el Shortest Path Problem](02_thesis_1998.md)**
