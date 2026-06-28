# HopfieldAttention

**De Hopfield Networks a Transformers: 25 años de optimización a atención.**

Una exploración interactiva de la línea evolutiva que conecta las Hopfield Networks (1982) — que resolvían el problema del camino más corto en mi tesis universitaria (1998) — con el mecanismo de atención que impulsa los Transformers modernos (2017) y su extensión a Subspace Attention.

## Estructura

```
articles/          → Serie de 7 capítulos trazando la conexión
dashboard/         → Streamlit app interactiva (energía vs atención)
bridge/            → Código que muestra la equivalencia formal
notebooks/         → Jupyter notebooks explicativos
publications/      → Drafts para Hashnode, Dev.to, HF Forums
```

## La Tesis Central

La actualización de estado de una Hopfield Network y el mecanismo de atención de un Transformer **son la misma operación matemática** con distintas funciones de normalización y softmax.

| Aspecto | Hopfield Network | Transformer Attention |
|---------|-----------------|---------------------|
| Representación | Estado de neuronas V | Valores V |
| Memoria / Claves | Patrones almacenados ξ | Claves K |
| Similaridad | Producto punto ξ·V | Producto punto Q·K^T |
| Actualización | V_new = softmax(β·ξ·V) · ξ | Attention = softmax(Q·K^T/√d) · V |
| Atractor | Punto fijo de energía | Promedio ponderado por atención |

## Autor

**Roberto de la Cámara** — Senior Technical Product Manager en Ericsson.
Tesis universitaria (1998): *Hopfield Neural Network for solving the Shortest Path Problem (SPP)*.
Hoy: definiendo la estrategia de transformación AI-native para Core Networks 5G.
