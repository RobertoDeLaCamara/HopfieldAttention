---
title: "De mi tesis de 1998 a Transformers: 25 años, 7 correcciones, una conexión inesperada"
date: 2026-05-11
author: "Roberto de la Cámara"
tags: [hopfield-networks, transformers, attention, personal-story, history-of-ml, neural-networks]
platforms: [devto]
---

# De mi tesis de 1998 a Transformers: 25 años, 7 correcciones, una conexión inesperada

En 1998 estaba en mi último año de carrera, sentado frente a un monitor CRT, debugando una red neuronal que se negaba a converger. Mi tesis: aplicar Hopfield Networks al problema del camino más corto.

La red funcionaba a veces. El 40-60% de las veces. Y cuando fallaba, no sabía por qué.

Veinticinco años después, trabajo definiendo la estrategia AI-native para Core Networks 5G en Ericsson. Y un día, revisando el mecanismo de atención de los Transformers, me di cuenta de algo que me dejó helado:

**La atención de un Transformer y la actualización de una Hopfield Network son la misma operación matemática.**

No "inspiradas en". No "relacionadas con". La misma.

---

## El momento "aha"

Mira estas dos fórmulas:

**Hopfield Network (moderna):**
```
V_new = softmax(β · Ξ · V) · Ξ
```

**Transformer Attention:**
```
Attention(Q, K, V) = softmax(Q · K^T / √d) · V
```

Son idénticas en estructura. En Hopfield, buscas en tu memoria (Ξ) el patrón más cercano a tu estado actual (V), y recuperas una combinación ponderada. En Attention, buscas en tus keys (K) lo más relevante para tu query (Q), y recuperas una combinación ponderada de values (V).

La única diferencia real: en Transformers aprendes a proyectar Q, K, V desde la misma entrada con matrices de peso distintas. En Hopfield clásico, los patrones Ξ son fijos.

---

## Lo que mi tesis me enseñó (y que sigue siendo cierto)

Debugando Hopfield Networks en 1998, aprendí lecciones que hoy aplico a diario:

**1. Las restricciones correctas son todo.** Mi implementación original tenía un bug sutil: codificaba las restricciones del Traveling Salesman Problem (ciclo Hamiltoniano) en lugar de Shortest Path (conservación de flujo). Corregir esto llevó la fiabilidad del 40% al 95%. Una sola línea de código.

**2. Sin estado compartido entre consultas.** Reutilizar el optimizador Adam entre consultas contaminaba el momento de queries anteriores. Cada consulta necesita estado fresco. Esto es aún más relevante hoy con modelos que sirven a millones de usuarios.

**3. Siempre ten un fallback.** Mi solver ahora ejecuta Dijkstra en paralelo. Si la solución Hopfield está dentro del 5% de la óptima, la uso. Si no, caigo a Dijkstra. Esto da 100% de fiabilidad. En producción, siempre hay que tener un plan B.

**4. La parada temprana ahorra el 40-60% del tiempo.** Cuando la energía se estabiliza, parar. No sigues iterando sin mejora. Este principio está en todas partes hoy: early stopping en training, speculative decoding en inference.

---

## Las 7 correcciones (resumen)

1. Restricciones de flujo vs. TSP (40% → 95% fiabilidad)
2. Cero entrenamiento offline (30-60s ahorrados)
3. Optimizador fresco por consulta
4. Fallback a Dijkstra (100% fiabilidad)
5. Parada temprana por energía estable
6. BFS en lugar de argmax greedy
7. Caché de modelo en memoria (2-3s → 50ms)

---

## La demo que lo visualiza

He construido un dashboard interactivo donde puedes ajustar la temperatura, el ruido, y ver cómo la misma operación se visualiza como Hopfield o como Attention:

- **β bajo** → atención difusa, todos los patrones pesan similar
- **β alto** → atención enfocada, un solo patrón domina

Es exactamente el mismo principio que el softmax temperature en LLMs.

---

## Por qué esto importa (y no es solo nostalgia)

1. La conexión Hopfield-Attention proporciona un **marco teórico sólido** para entender por qué los transformers funcionan
2. Sugiere **nuevas arquitecturas**: atención iterativa (múltiples pasos Hopfield), memoria jerárquica, temperatura dinámica
3. Cierra un círculo personal de 25 años

La Hopfield Network de 1982, que parecía un callejón sin salida, resultó ser el embrión de la arquitectura que define la IA moderna. No se necesitó una idea nueva. Se necesitaron mejores funciones de activación, mejor hardware, y 25 años de ingeniería.

---

*El código, el dashboard y los 7 capítulos completos están aquí:*  
**GitHub:** [RobertoDeLaCamara/HopfieldAttention](https://github.com/RobertoDeLaCamara/HopfieldAttention)

*Hardware usado en el homelab: RTX 5060 Ti + WSL2 (Hawkeye) + Raspberry Pi cluster para monitoreo de red.*

*Si quieres leer el artículo técnico completo (con matemáticas, código y benchmarks): [Hashnode — De Hopfield Networks a Transformers](URL_AQUI)*
