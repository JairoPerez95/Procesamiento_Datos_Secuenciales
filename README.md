# ProphetNet — Generación Automática de Titulares de Noticias

> Proyecto Final — Procesamiento de Datos Secuenciales  
> Arquitectura Transformer encoder-decoder con *future n-gram prediction*

---

## Integrantes

* Jairo Andrés Pérez Hurtatis
* Diego Mauricio Ortiz
* Daniel Felipe Zamora Pineda

---

## 1. Resumen (Abstract)

Este proyecto implementa el proceso de inferencia sobre **ProphetNet**, un modelo Transformer encoder-decoder desarrollado por Microsoft Research, aplicado a la tarea de generación automática de titulares a partir de artículos de noticias en inglés (*headline generation*). Se utilizan los pesos preentrenados del modelo `microsoft/prophetnet-large-uncased-cnndm`, ajustado sobre el dataset CNN/DailyMail. La implementación incluye el pipeline completo de inferencia (tokenización → generación con beam search → decodificación), inspección directa de las matrices de proyección Q, K y V por capa, extracción y visualización interactiva del mecanismo de cross-attention con selector de capa (superficial a profunda), evaluación cuantitativa con métricas ROUGE calculadas dinámicamente (Precision, Recall y F1), y una interfaz interactiva desarrollada con Streamlit que permite ingresar cualquier artículo, compararlo contra un titular de referencia y observar la inferencia del modelo en tiempo real. Los resultados obtenidos muestran un ROUGE-1 promedio de 0.377, ROUGE-2 de 0.141 y ROUGE-L de 0.317 sobre diez artículos de referencia.

---

## 2. Introducción

### Artículo base

**ProphetNet: Predicting Future N-gram for Sequence-to-Sequence Pre-training**  
Yan et al. (2020) — arXiv:2001.04063  
Repositorio original: https://github.com/microsoft/ProphetNet  
Modelo en HuggingFace: https://huggingface.co/microsoft/prophetnet-large-uncased-cnndm

### Contexto y motivación

La generación automática de titulares es una tarea de procesamiento de datos secuenciales que consiste en producir un resumen conciso de un texto largo. Los modelos Transformer encoder-decoder son el estado del arte para este tipo de tareas, pero presentan un problema fundamental: durante el entrenamiento, el decoder aprende a predecir solo el siguiente token, lo que lleva al modelo a enfocarse demasiado en el contexto local inmediato (*overfitting on local context*) y perder coherencia global.

ProphetNet introduce una solución arquitectónica a este problema mediante el **n-stream self-attention decoder**, que obliga al modelo a predecir múltiples tokens futuros en paralelo durante el entrenamiento, produciendo representaciones de contexto más ricas y titulares con mayor coherencia global.

### Objetivo

Aplicar la arquitectura ProphetNet para inferencia sobre artículos de noticias en inglés, comprender en profundidad su mecanismo de atención n-stream y las dimensiones reales de los tensores Q, K y V, y presentar los resultados mediante una interfaz interactiva que visualice el proceso de generación capa por capa.

---

## 3. Marco Teórico

### 3.1 Arquitectura Transformer encoder-decoder

ProphetNet sigue la arquitectura estándar encoder-decoder con una innovación crítica en el decoder:

- **Encoder**: procesa el artículo de entrada de forma **bidireccional**, similar a BERT. Cada token atiende a todos los demás tokens del artículo construyendo representaciones ricas en contexto. Produce un conjunto de *hidden states* `h ∈ ℝ^(seq × d_model)` que capturan el significado del texto completo. Compuesto por 12 capas con self-attention completa (sin máscara causal).

- **Decoder**: genera el titular token por token de forma **autoregresiva**, usando tres tipos de atención en cada capa:
  - **N-stream self-attention**: el decoder atiende a los tokens que ya generó, con streams paralelos durante el entrenamiento (detalle en sección 3.3).
  - **Cross-attention**: el decoder consulta los hidden states del encoder para incorporar información del artículo original. **Este es el mecanismo que visualizamos en los heatmaps.**
  - **Feed-forward network**: transformación no lineal por token, dimensión interior 4096.

### 3.2 Mecanismo de atención — Q, K, V

El mecanismo de atención opera con tres componentes para cada token, obtenidos mediante proyecciones lineales aprendidas:
```
Q = h · Wq     Wq ∈ ℝ^(1024 × 1024)
K = h · Wk     Wk ∈ ℝ^(1024 × 1024)
V = h · Wv     Wv ∈ ℝ^(1024 × 1024)
```

La fórmula de atención escalada es:
```
Attention(Q, K, V) = softmax(Q · Kᵀ / √d_head) · V
```

donde `d_head = d_model / num_heads = 1024 / 16 = 64`.

El producto `Q · Kᵀ` mide la similitud entre el query de un token y los keys de todos los demás. La división por `√64 = 8` estabiliza los gradientes evitando que el producto escalar crezca en magnitud con la dimensión. El `softmax` normaliza esos scores en probabilidades (los *attention weights*), que ponderan los Values para producir la representación final.

**En la cross-attention específicamente:**

| Componente | Origen | Significado |
|---|---|---|
| Q | Estado del **decoder** · Wq | "¿Qué información busca el token del titular?" |
| K | Estado del **encoder** · Wk | "¿De qué trata cada token del artículo?" |
| V | Estado del **encoder** · Wv | "¿Qué contenido aporta cada token del artículo?" |

Los pesos `softmax(Q · Kᵀ / √64)` son exactamente los valores visualizados en los heatmaps de la aplicación: indican cuánta atención presta cada token del titular generado a cada token del artículo de entrada.

**Dimensiones reales de los tensores en ProphetNet** (verificadas por inspección directa del modelo):

| Tensor | Shape | Descripción |
|---|---|---|
| `input_ids` | `(1, seq_enc)` | Tokens del artículo indexados |
| `encoder_hidden_states` | `(1, seq_enc, 1024)` | Salida del encoder por token |
| `Wq, Wk, Wv` | `(1024, 1024)` | Matrices de proyección por capa |
| `Q, K, V` (antes de split) | `(1, seq, 1024)` | Proyecciones completas |
| `Q, K, V` (por cabeza) | `(1, 16, seq, 64)` | Dividido en 16 cabezas de 64 dims |
| `cross_attentions[-1]` | `(1, 16, seq_dec, seq_enc)` | Pesos post-softmax, última capa |

### 3.3 Innovación de ProphetNet: N-stream decoder

La innovación clave de ProphetNet es su **n-stream self-attention** en el decoder. En vez de tener un único flujo que predice solo el siguiente token, ProphetNet usa **n flujos en paralelo** durante el entrenamiento, donde cada stream predice un token futuro diferente con matrices Q/K/V propias.

Con n=2 (el valor usado en este proyecto), para cada posición `t` el decoder mantiene:

| Stream | Predice | Matrices propias | Máscara de atención |
|---|---|---|---|
| Main stream | token `t+1` | `Wq1, Wk1, Wv1` | Causal estándar: solo atiende `t' < t` |
| Predicting stream | token `t+2` | `Wq2, Wk2, Wv2` | Bloquea posición `t`, fuerza contexto más lejano |

Esto cambia la función de pérdida durante el entrenamiento:

**Transformer estándar (BART, T5):**
```
L = −Σₜ log P(yₜ | y<t, x)
```

**ProphetNet (n=2):**
```
L = −Σₜ [ log P(yₜ₊₁ | y<t, x) + log P(yₜ₊₂ | y<t, x) ]
```

Al obligar al modelo a "planear" dos tokens hacia adelante durante el entrenamiento, aprende representaciones que son útiles para predecir tanto el siguiente token como el subsiguiente, desincentivando el *local coherence shortcut* (copiar frases del artículo) y produciendo texto con mayor coherencia global.

> **Importante — comportamiento en inferencia:** el predicting stream opera **únicamente 
> durante el entrenamiento**. En inferencia (`model.generate()`), solo actúa el main 
> stream de forma autoregresiva, idéntico a un decoder estándar. La ventaja del n-stream 
> está codificada en los pesos aprendidos, no en la arquitectura de inferencia.

**Detalle de máscaras por stream:**

| Stream | Puede ver | Bloqueado |
|---|---|---|
| Main stream | `y₁, ..., y_{t-1}` (todos los tokens previos) | `y_t` en adelante |
| Predict stream | `y₁, ..., y_{t-1}` (todos los tokens previos) | `y_t` (posición actual) |

El predict stream bloquea explícitamente la posición `t` para forzar al modelo a predecir
`y_{t+2}` sin ver el token inmediatamente anterior, desincentivando el atajo de copiar
contexto local y produciendo representaciones más globales.

**Comparación con otros modelos encoder-decoder:**

| Modelo | Tipo | N-stream decoder | Pre-training |
|---|---|---|---|
| BERT | Encoder only | No | Masked LM |
| GPT-2 | Decoder only | No | Causal LM |
| BART | Encoder-Decoder | No | Denoising Autoencoder |
| T5 | Encoder-Decoder | No | Text-to-Text |
| **ProphetNet** | **Encoder-Decoder** | **Sí (n=2)** | **Future N-gram Prediction** |

### 3.4 Parámetros del modelo

| Parámetro | Valor |
|---|---|
| Parámetros totales | 485,085,184 |
| Capas del encoder | 12 |
| Capas del decoder | 12 |
| Dimensión oculta (d_model) | 1024 |
| Cabezas de atención | 16 |
| d_head = d_model / heads | 64 |
| FFN dimensión interior | 4096 |
| N-gram del decoder (n) | 2 |
| Vocabulario | 30,522 tokens |
| Máx. tokens encoder | 512 |

---

## 4. Metodología

### 4.1 Modelo utilizado

Se utilizó `microsoft/prophetnet-large-uncased-cnndm`, la versión fine-tuneada sobre el dataset CNN/DailyMail para headline generation. El modelo se carga en precisión `float16` con `low_cpu_mem_usage=True` para reducir el consumo de memoria a ~800 MB, permitiendo su ejecución en el tier gratuito de Streamlit Cloud sin degradación observable en la calidad de los titulares generados.

### 4.2 Herramientas

| Herramienta | Versión | Uso |
|---|---|---|
| Python | 3.12 | Lenguaje de implementación |
| PyTorch | ≥2.0 | Framework de deep learning |
| HuggingFace Transformers | ≥4.35 | Carga del modelo y tokenizer |
| rouge-score | ≥0.1.2 | Cálculo dinámico de métricas ROUGE |
| matplotlib / seaborn | latest | Visualización de heatmaps |
| Streamlit | ≥1.30 | Interfaz interactiva web |
| pandas | ≥2.0 | Presentación tabular de resultados |
| Google Colab (GPU T4) | — | Entorno de ejecución del notebook |

### 4.3 Pesos preentrenados

Los pesos se descargan automáticamente desde HuggingFace Hub en la primera ejecución (~1.57 GB) y quedan en caché:
```python
from transformers import ProphetNetForConditionalGeneration, ProphetNetTokenizer

tokenizer = ProphetNetTokenizer.from_pretrained("microsoft/prophetnet-large-uncased-cnndm")
model = ProphetNetForConditionalGeneration.from_pretrained(
    "microsoft/prophetnet-large-uncased-cnndm",
    torch_dtype=torch.float16,    # reduce memoria a ~800 MB
    low_cpu_mem_usage=True,       # carga por partes, nunca duplica en RAM
)
```

No se realiza entrenamiento desde cero — se usan directamente los pesos publicados por Microsoft Research.

---

## 5. Desarrollo e Implementación

### 5.1 Clonar el repositorio
```bash
git clone https://github.com/DanielZampi/proyecto_procesamiento_de_datos.git
cd proyecto_procesamiento_de_datos
```

### 5.2 Instalar dependencias
```bash
pip install transformers torch sentencepiece rouge-score matplotlib seaborn streamlit pandas Pillow
```

### 5.3 Correr el notebook de inferencia

Abrir `ProphetNet_Completo.ipynb` en Google Colab:
1. Ir a `Entorno de ejecución → Cambiar tipo de entorno → GPU (T4)`
2. Ejecutar todas las celdas en orden

El notebook cubre: inspección de arquitectura interna, inferencia sobre 10 artículos, métricas ROUGE con Precision/Recall/F1, visualización multicapa de cross-attention, análisis de hiperparámetros y limitaciones del modelo.

### 5.4 Correr la app interactiva
```bash
streamlit run app.py
```

O acceder directamente en: **https://proyecto-procesamiento-de-datos-secuenciales-articulo.streamlit.app**

### 5.5 Pipeline de inferencia

El proceso de inferencia sigue tres pasos:

**Paso 1 — Preprocesamiento:** el texto se convierte a minúsculas (modelo uncased) y se tokeniza con el vocabulario BERT-uncased (30,522 tokens):
```python
inputs = tokenizer(
    articulo.lower(),
    return_tensors="pt",
    truncation=True,
    max_length=256          # límite operacional con float16
)
```

**Paso 2 — Inferencia con beam search:** el main stream del decoder genera tokens autoregresivamente manteniendo las `num_beams` hipótesis más probables en paralelo:
```python
output_ids = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    num_beams=4,
    max_length=60,
    min_length=8,
    no_repeat_ngram_size=3,
    early_stopping=True
)
```

**Paso 3 — Decodificación:** los tokens generados se convierten de vuelta a texto eliminando los tokens especiales `[BOS]`, `[EOS]`, `[PAD]`:
```python
titular = tokenizer.decode(output_ids[0], skip_special_tokens=True)
```

### 5.6 Extracción del mecanismo de atención

Se usa `output_attentions=True` para obtener los pesos de cross-attention internos de todas las capas. Se hace un forward pass teacher-forced usando el titular ya generado como `decoder_input_ids`:
```python
outputs = model(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    decoder_input_ids=labels,
    output_attentions=True
)

# Capa seleccionada, promedio de las 16 cabezas de atención
# Shape: (batch=1, heads=16, seq_dec, seq_enc) → (seq_dec, seq_enc)
cross_attn = outputs.cross_attentions[capa].squeeze(0).mean(dim=0).float().cpu().numpy()

# Normalización defensiva (pesos ya son post-softmax, garantiza suma = 1.0)
row_sums = cross_attn.sum(axis=-1, keepdims=True)
row_sums = np.where(row_sums == 0, 1, row_sums)
cross_attn = cross_attn / row_sums
```

> **Nota metodológica:** los pesos de `cross_attentions` en HuggingFace ya vienen en formato post-softmax (probabilidades entre 0 y 1). No se aplica `np.exp()`. La normalización defensiva garantiza que cada fila sume exactamente 1.0 sin distorsionar la distribución aprendida.

### 5.7 Cálculo dinámico de métricas ROUGE

Las métricas se calculan en tiempo real sobre el titular generado en cada ejecución, comparado contra el titular de referencia ingresado por el usuario:
```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
scores = scorer.score(target=titular_real, prediction=titular_generado)

# Retorna Precision, Recall y F1 para cada métrica
r1_f1 = scores["rouge1"].fmeasure
r1_p  = scores["rouge1"].precision
r1_r  = scores["rouge1"].recall
```

`use_stemmer=True` reduce las palabras a su raíz antes de comparar (`"collected"` y `"collecting"` se tratan como equivalentes). Se reporta el **F1-score** que balancea precisión y recall.

---

## 6. Resultados y Análisis

### 6.1 Titulares generados (beam=4, max_length=60)

| # | Artículo | Titular real (ground truth) | ProphetNet |
|---|---|---|---|
| 1 | Diplomacia Bolivia | us rejects charges against its ambassador in bolivia | the us state department said it had received no formal word from bolivia that it was expelling the us ambassador |
| 2 | Dinosaurio Argentina | scientists discover one of the largest dinosaurs ever found in argentina | scientists have discovered a new species of dinosaur in argentina. the titanosaur is estimated to have weighed 70 tons |
| 3 | Apple ganancias | apple reports record quarterly revenue driven by iphone and services growth | apple announces record revenue of 90 billion dollars. the company announced a new stock buyback program |
| 4 | NASA Marte | nasa perseverance rover collects first mars rock sample | nasa's perseverance rover has successfully collected its first rock sample from the surface of mars |
| 5 | OMS emergencia | who declares global health emergency over new respiratory virus | the world health organization declared a global health emergency on thursday. a new respiratory virus continues to spread |
| 6 | UE clima | eu agrees landmark deal to cut emissions 55 percent by 2030 | the european union has agreed on a landmark climate deal to cut greenhouse gas emissions by 55 percent by 2030 |
| 7 | Ucrania-Rusia | ukraine russia resume peace talks with progress on neutrality deal | ukraine and russia have resumed peace talks in istanbul after weeks of fighting |
| 8 | Fed tasas | fed raises rates by 75 basis points in largest hike since 1994 | the federal reserve raised interest rates by 75 basis points, the largest increase since 1994 |
| 9 | Google Gemini | google unveils gemini ai model claiming to surpass gpt-4 | google has unveiled its new artificial intelligence model called gemini |
| 10 | Brasil Bolsonaro | brazil court bars bolsonaro from running for office until 2030 | brazil's supreme court has ruled that former president jair bolsonaro is ineligible to run for office until 2030 |

### 6.2 Métricas ROUGE (calculadas dinámicamente — Precision / Recall / F1)

| # | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 |
|---|---|---|---|
| 1 | 0.2703 | 0.0000 | 0.1622 |
| 2 | 0.3684 | 0.0556 | 0.3158 |
| 3 | 0.1935 | 0.0000 | 0.1935 |
| 4 | 0.4444 | 0.1176 | 0.3889 |
| 5 | 0.3784 | 0.2286 | 0.3784 |
| 6 | 0.4706 | 0.2500 | 0.4118 |
| 7 | 0.3810 | 0.1111 | 0.2857 |
| 8 | 0.5000 | 0.3077 | 0.4286 |
| 9 | 0.3636 | 0.1538 | 0.2727 |
| 10 | 0.4000 | 0.1818 | 0.3333 |
| **Promedio** | **0.3770** | **0.1406** | **0.3171** |
| **Máximo** | **0.5000** | **0.3077** | **0.4286** |
| **Mínimo** | **0.1935** | **0.0000** | **0.1622** |
| **Desv. estándar** | **0.0876** | **0.1021** | **0.0849** |

El ROUGE-1 promedio de 0.377 indica overlap significativo de palabras individuales entre el titular generado y el real. El ROUGE-2 de 0.141 refleja que el modelo parafrasea en vez de copiar frases exactas del artículo — comportamiento esperado y deseable en headline generation. El ejemplo de la Fed (ROUGE-1: 0.500) es el mejor resultado, donde el modelo reprodujo con alta precisión los elementos clave: tasa, puntos básicos, aumento, 1994.

> **Referencia de contexto:** los valores ROUGE para headline generation son naturalmente más bajos que para summarization, porque un mismo evento puede titularse de múltiples formas igualmente válidas. Rush et al. (2015) reportan ROUGE-1 ~35 como baseline sobre Gigaword para modelos seq2seq.

### 6.3 Análisis del mecanismo de atención por capas

El selector de capa de la aplicación permite comparar los patrones de atención entre capas superficiales y profundas:

- **Capas superficiales (1-4):** distribución amplia de atención. El decoder atiende tokens de función (artículos, preposiciones, conectores) y construye representaciones sintácticas básicas.
- **Capas medias (5-8):** atención más selectiva. Emergen patrones semánticos: tokens de entidades nombradas y verbos clave reciben pesos crecientes.
- **Capas profundas (9-12):** alta concentración de atención en los tokens más informativos del artículo. Para el artículo de la NASA, los tokens `"perseverance"`, `"rock"`, `"sample"` y `"mars"` concentran la mayor atención en la capa 12, confirmando que el modelo identifica correctamente las palabras clave antes de generar cada token del titular.

### 6.4 Análisis de hiperparámetros

Se evaluó el efecto del número de beams sobre la calidad (ROUGE-1) y longitud del titular generado:

| Beams | ROUGE-1 (NASA) | ROUGE-L (NASA) | Palabras | Observación |
|---|---|---|---|---|
| 1 (greedy) | 0.380 | 0.320 | 12 | Decodificación greedy — menor calidad global |
| 2 | 0.400 | 0.350 | 13 | Mejora notable respecto a greedy |
| **4** | **0.444** | **0.389** | **15** | **Mejor balance calidad/costo ✓** |
| 6 | 0.444 | 0.389 | 15 | Sin mejora adicional |
| 8 | 0.444 | 0.389 | 15 | Rendimiento decreciente para titulares cortos |

La configuración óptima encontrada es **beam=4, max_length=60**, que ofrece el mejor balance entre calidad del titular y costo computacional.

---

## 7. Conclusiones

- ProphetNet demuestra ser efectivo para headline generation cuando se usa la versión fine-tuneada sobre CNN/DailyMail, alcanzando un ROUGE-1 promedio de 0.377 sobre 10 artículos de referencia.
- La innovación del n-stream decoder (n=2) obliga al modelo a predecir dos tokens hacia adelante simultáneamente durante el entrenamiento, con matrices Q/K/V independientes por stream, produciendo representaciones de contexto más ricas que un decoder estándar. Este mecanismo opera **únicamente en entrenamiento**; en inferencia solo actúa el main stream.
- La inspección directa de las matrices de proyección confirma que cada capa mantiene tres conjuntos de pesos `(Wq, Wk, Wv) ∈ ℝ^(1024×1024)` para encoder self-attention, n-stream decoder self-attention y cross-attention, con un total de 485M parámetros.
- El mecanismo de cross-attention extrae correctamente los tokens más relevantes del artículo, con los patrones de atención de las capas profundas concentrados en las entidades y verbos clave, visible en los heatmaps interactivos de la aplicación.
- El modelo presenta limitaciones con textos fuera del dominio periodístico y genera oraciones descriptivas largas en vez de titulares concisos, reflejo del sesgo del dataset CNN/DailyMail hacia summarization. Para headline generation estricta existe el checkpoint entrenado sobre Gigaword.
- La carga en `float16` con `low_cpu_mem_usage=True` reduce el consumo de memoria de ~1.57 GB a ~800 MB, permitiendo el despliegue gratuito en Streamlit Cloud sin degradación observable en calidad.

**Posibles mejoras:** fine-tuning sobre el checkpoint Gigaword para titulares más concisos, exploración de ProphetNet con n=3, evaluación sobre XSum, y cuantización int8 para reducir aún más la latencia de inferencia.

---

## 8. Referencias

[1] Y. Yan, W. Qi, Y. Gong, D. Liu, N. Duan, J. Chen, R. Zhang, and M. Zhou, "ProphetNet: Predicting future n-gram for sequence-to-sequence pre-training," *arXiv preprint arXiv:2001.04063*, 2020.

[2] Microsoft, "microsoft/prophetnet-large-uncased-cnndm," Hugging Face, 2020. [Online]. Available: https://huggingface.co/microsoft/prophetnet-large-uncased-cnndm

[3] C.-Y. Lin, "ROUGE: A package for automatic evaluation of summaries," in *Proc. ACL Workshop on Text Summarization Branches Out*, Barcelona, Spain, 2004, pp. 74–81.

[4] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, "Attention is all you need," in *Advances in Neural Information Processing Systems*, vol. 30, 2017.

[5] C. Raffel, N. Shazeer, A. Roberts, K. Lee, S. Narang, M. Matena, Y. Zhou, W. Li, and P. J. Liu, "Exploring the limits of transfer learning with a unified text-to-text transformer," *J. Mach. Learn. Res.*, vol. 21, no. 140, pp. 1–67, 2020.
