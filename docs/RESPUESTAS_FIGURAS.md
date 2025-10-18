# Respuestas a Preguntas sobre las Figuras

## Figura 1: Comparative Correlations (Barplot)

### Pregunta: "El corte weak vs. strong parece arbitrario. ¿Cómo has obtenido ese 'max absolute correlation'?"

**Cálculo del "Max Absolute Correlation":**

Para cada feature (e.g., Temporal Variance), calculamos la correlación de Pearson con **cada una de las 25 dimensiones latentes**:

```python
correlations = []
for latent_dim in range(25):
    r, p_value = pearsonr(latent_codes[:, latent_dim], feature_values)
    correlations.append(abs(r))

max_correlation = max(correlations)
```

Es decir, tomamos la dimensión latente que **mejor** codifica esa feature (en valor absoluto).

**El umbral 0.3:**
- Es convención en estadística: r < 0.3 = débil, 0.3-0.7 = moderada, > 0.7 = fuerte
- Pero tienes razón que es algo arbitrario

**Mejora propuesta:**
- Añadir p-values y intervalos de confianza
- Mostrar la distribución completa de correlaciones, no solo el máximo
- Usar un test de permutación para establecer un umbral empírico

---

## Figura 2: Correlation Heatmap

### Pregunta: "¿Cómo defines cada cantidad en las filas?"

**Definiciones de Features Dinámicas:**

1. **Temporal Variance**:
   ```
   var_temporal = mean_over_species(variance_over_time(X[species, :]))
   ```
   Cuánto fluctúa cada especie a lo largo del tiempo, promediado sobre todas las especies.

2. **Steady State**:
   ```
   steady_state = mean(X[:, -20:])  # últimos 20 timesteps
   ```
   Valor de equilibrio al final de la serie temporal.

3. **Peak Timing**:
   ```
   peak_timing = mean_over_species(argmax_over_time(X[species, :]))
   ```
   En qué timestep ocurre el pico de población (promedio sobre especies).

4. **Overshoot**:
   ```
   overshoot = mean((max(X) - steady_state) / steady_state)
   ```
   Cuánto sobrepasa el pico el valor de equilibrio final.

5. **Initial Growth**:
   ```
   initial_growth = mean(X[:, 5] - X[:, 0])  # primeros 5 timesteps
   ```
   Tasa de crecimiento al inicio.

6. **Species Correlation**:
   ```
   species_corr = mean(abs(corrcoef(X)))  # correlación entre especies
   ```
   Cuánto co-varían las dinámicas de diferentes especies.

---

## Figura 3: Example Time Series - VARIANZA

### Pregunta: "Las med-low variance parecen tener más varianza que high-variance"

**El problema:** La varianza temporal es **per-species averaged**, no visual.

**Cálculo actual:**
```python
temporal_variance = mean_over_species(variance_over_time(X[i, :]))
```

**Por qué es confuso:**
- Una serie con gran pico pero rápido equilibrio puede tener ALTA varianza
- Una serie con oscilaciones pequeñas pero constantes puede tener BAJA varianza
- Lo que ves visualmente es la AMPLITUD, no la varianza temporal

**Ejemplo:**
- Serie A: [0, 1, 0.5, 0.5, 0.5, ...] → var = 0.19
- Serie B: [0.3, 0.4, 0.35, 0.38, ...] → var = 0.0013

Serie A tiene más "drama" pero menos varianza si el equilibrio es estable.

**Mejora propuesta:** Cambiar a features más intuitivos:
- "Stable dynamics" vs "Transient oscillations"
- "Boom-bust" vs "Gradual equilibrium"
- Mostrar la serie temporal normalizada Y su varianza numérica

---

## Figura 4: Latent Space (UMAP) con colores

### Pregunta: "Necesito más info sobre correlaciones y clustering"

**Qué muestra UMAP:**
- Reducción no-lineal de 25D → 2D
- Preserva estructura de vecindarios (puntos cercanos en 25D están cercanos en 2D)
- **NO** preserva distancias globales ni correlaciones lineales

**Clustering observado:**
- Hay gradientes suaves, no clusters discretos
- Esto indica que las features varían continuamente en el espacio latente
- La falta de clusters = buena cobertura del espacio, no modos colapsados

**Mejora propuesta:**
- Añadir análisis de clusters (k-means, DBSCAN)
- Mostrar PCA en vez de UMAP para ver correlaciones lineales
- Cuantificar la "estructura" con silhouette scores

---

## Figura 5: Latent Dimension Interpretation

### Pregunta: "¿Son direcciones puras? Curioso que no sea una mezcla. ¿PCA por target?"

**Sí, son direcciones puras:**
- z19 codifica temporal variance (r=0.713)
- z12 codifica steady state (r=0.533)
- Cada dimensión latente tiene una interpretación dominante

**Por qué no es mezcla:**
- El VAE aprende representaciones **disentangled** por la regularización KL
- Beta-VAE fuerza independencia entre dimensiones latentes
- Resultado: cada z_i captura un factor de variación

**Tu idea de "PCA supervisado por target" es excelente:**

Esto se llama **Canonical Correlation Analysis (CCA)** o **Partial Least Squares (PLS)**:

```python
# Para cada target (temporal_variance, steady_state, etc.)
# Encontrar combinación lineal óptima de dimensiones latentes
w_optimal = argmax_w( corr(Z @ w, target) )
```

Esto nos diría:
- ¿Es z19 SOLO temporal variance, o también tiene steady state?
- ¿Qué combinaciones de dimensiones explican mejor cada target?

---

## Figura 6: Latent Space Key Features (UMAP colormap)

### Pregunta: "No hay mucha estructura salvo Peak Timing. ¿UMAP no es bueno aquí?"

**Tienes razón:**
- UMAP es no-lineal y no preserva correlaciones globales
- Peak Timing muestra gradiente → está bien capturado por vecindarios locales
- Temporal Variance está más distribuido → UMAP no lo captura bien

**Problema:** UMAP optimiza para **estructura local**, no para **gradientes globales**.

**Mejora propuesta:**
- Usar **PCA** o **t-SNE** para comparar
- Graficar las 2 dimensiones latentes MÁS correlacionadas con cada feature
  - Para Temporal Variance: graficar (z19, z17) directamente
  - Para Steady State: graficar (z12, z21) directamente
- Esto mostrará la estructura **real** sin compresión no-lineal

---

## Figura 7: Variance Explained (PCA en latent space)

### Pregunta: "La info está muy repartida. ¿Esperarías dirección con poca PVE si faltaran dimensiones?"

**Interpretación actual:**
- 90% varianza en 22 dimensiones → espacio casi uniforme
- Primera componente PCA: solo 8.2% varianza

**Tu hipótesis:**
> "Si nos faltaran dimensiones, esperaría alguna con muy poca PVE"

**Esto es CORRECTO pero al revés:**

- **Si hay dimensiones con MUY poca varianza** → están "desperdiciadas", no se usan
- **Si todas tienen varianza similar** → todas se usan, espacio eficiente
- **Si faltaran dimensiones** → veríamos **colapso posterior** (KL loss → 0)

**Lo que vemos (varianza uniforme) indica:**
1. Las 25 dimensiones SÍ son necesarias (no sobran)
2. El modelo usa el espacio latente eficientemente
3. **PERO:** ¿Son 25 suficientes? Para saberlo:
   - Ver si reconstruction loss sigue bajando con más dimensiones
   - Probar latent_dim = 30, 35, 40 y comparar

**Test definitivo:**
```python
for latent_dim in [15, 20, 25, 30, 35, 40]:
    train_VAE(latent_dim=latent_dim)
    plot(reconstruction_loss vs latent_dim)
```

Si se estanca en 25 → suficiente
Si sigue bajando → necesitamos más

---

## Resumen de Mejoras Propuestas

1. **Fig 1 (Barplot):** Añadir p-values, bootstrap CI, permutation test
2. **Fig 2 (Heatmap):** Añadir definiciones matemáticas como anotaciones
3. **Fig 3 (Examples):** Reemplazar "variance" con features más intuitivos
4. **Fig 4 (UMAP):** Añadir análisis de clustering cuantitativo
5. **Fig 5 (Interpretation):** Implementar CCA/PLS para encontrar combinaciones óptimas
6. **Fig 6 (Key Features):** Reemplazar UMAP con PCA o plot directo de dimensiones relevantes
7. **Fig 7 (Variance):** Añadir ablation study de latent_dim

¿Quieres que implemente alguna de estas mejoras?
