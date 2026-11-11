# Guía de preparación para la sustentación

El 10 % individual lo deciden las respuestas a las preguntas, y **cualquiera** de los tres puede recibir cualquier pregunta.
Estudien todas las secciones, no solo la que exponen. Las cifras exactas están en `runs/metrics.json` y en el informe.

## Reparto sugerido de la exposición (15 min ≈ 5 min por persona)
| Expositor | Diapositivas | Tema |
|---|---|---|
| Cristobal Rodriguez Rojas | 1–6 | Problema, objetivos, datos, preprocesamiento, cambio de dominio |
| Juan Esteban Villada | 7–12 | Arquitecturas (EfficientNet, ViT), configuración, curvas, test, matriz de confusión |
| Cristian David Cabezas Jimenez | 13–18 | Generalización (RoCoLe), Grad-CAM, severidad/OOD/ablación, conclusiones, referencias |

Cada diapositiva trae en las **notas del orador** el guion sugerido.

### Cifras que deben saberse de memoria
| Qué | Valor |
|---|---|
| Datos | BRACOL 1685 hojas (1179/253/253); RoCoLe 1393 sana/roya + 167 arañita roja |
| Mejor F1 en validación | ViT-Tiny 0,935 · EfficientNet-B0 0,915 · CNN base 0,877 · EffNet sin aumento 0,913 |
| Test BRACOL (acc / F1-macro) | ViT 87,4 % / 0,843 · EffNet 88,1 % / 0,864 · CNN 84,2 % / 0,808 · sin aumento 88,1 % / 0,851 |
| Generalización ViT (acc 5 clases) | test 87 % → raw 31 % → crop 35 % → mask 69 % |
| AUROC sana/enferma ViT | test 1,00 → raw 0,75 → crop 0,80 → mask 0,93 |
| Hojas sanas de campo predichas enfermas (raw) | 96 % |
| Errores por multiestrés (ViT, test) | un estrés 92 % de acierto vs. varios estreses 66 % |
| Ablación (EffNet sin aumento) | F1 test 0,851 vs. 0,864, pero AUROC mask 0,59 vs. 0,90 |
| Arañita roja | 55 % → roya, 22 % → minador; confianza 0,54 ≈ 0,51 de clases conocidas; AUROC detección 0,45 |
| Cómputo | CPU i5-1155G7, 20 GB RAM, sin GPU; 5,4 h para los 4 modelos |

---

## A. Problema y datos
**1. ¿Por qué este problema importa?**
La roya del café (*Hemileia vastatrix*) es la enfermedad foliar más dañina del cultivo. La epidemia de 2008–2011 en Colombia
redujo fuertemente la producción nacional, y el diagnóstico temprano permite aplicar control focalizado. El diagnóstico visual
lo hace hoy un experto en campo; un clasificador en el celular abarata y escala ese diagnóstico.

**2. ¿Por qué Deep Learning y no reglas o ML clásico?**
Los síntomas son patrones de textura y color muy variables (iluminación, tamaño, cantidad de manchas, enfermedades mezcladas).
Las CNN aprenden directamente esos descriptores jerárquicos a partir de los píxeles, sin diseñar características a mano,
y con transfer learning funcionan bien incluso con unos pocos miles de imágenes.

**3. ¿Por qué BRACOL para entrenar y RoCoLe para evaluar?**
BRACOL tiene 5 clases etiquetadas por expertos y es el dataset de referencia (Esgario et al., 2020). RoCoLe es independiente:
otra variedad (robusta vs. arábica), otro país (Ecuador vs. Brasil), otro celular y, sobre todo, otra condición de captura
(hoja en la planta con fondo de campo vs. hoja desprendida sobre fondo blanco). Eso permite medir el cambio de dominio real.

**4. ¿Por qué descartaron las 62 hojas "indeterminadas"?**
No tienen una clase objetivo definida para la tarea multiclase; incluirlas metería ruido en las etiquetas.

**5. ¿Cómo evitaron la fuga de información?**
(i) La partición 70/15/15 estratificada se hizo una sola vez con semilla fija y se verifica con una huella MD5.
(ii) En BRACOL cada imagen es una hoja física distinta, así que no hay fotos casi duplicadas de la misma hoja entre conjuntos.
(iii) La normalización usa μ/σ de ImageNet, no estadísticas calculadas con validación o prueba.
(iv) El aumento de datos solo se aplica en entrenamiento.
(v) El modelo se elige con validación; el test se usa una sola vez al final.
(vi) RoCoLe nunca se usa para entrenar ni para escoger hiperparámetros.

**6. ¿Por qué letterbox y no estirar la imagen a 224×224?**
Las hojas de BRACOL son 2:1 y las de RoCoLe 9:16. Estirarlas deformaría las lesiones (las manchas redondas de roya
se volverían elipses) y lo haría de forma distinta en cada dominio. El letterbox conserva la geometría y rellena con
el color medio de ImageNet, que tras normalizar vale ≈ 0. **Costo:** la hoja de BRACOL ocupa la mitad del lienzo, así que la
resolución efectiva baja; es una limitación reconocida.

**7. ¿Por qué esos aumentos de datos?**
- Rotaciones de 90° y volteos: la orientación de la hoja es arbitraria (horizontal en BRACOL, vertical en campo).
- `RandomResizedCrop` (escala 0.7–1): robustez a encuadre y escala.
- `ColorJitter`: robustez a la iluminación y a la cámara.
No usamos transformaciones que cambien el color de las lesiones de forma drástica (por ejemplo, el *hue* va con ±0.03),
porque el color es justamente lo que diferencia la roya (naranja) de otras manchas.

**8. ¿Cómo manejaron el desbalance?**
Hay 3.6× entre roya (531) y cercospora (147). Usamos **entropía cruzada ponderada** con w_c = N / (K·n_c), calculada solo con
entrenamiento, y reportamos **F1-macro**, que pesa igual a todas las clases, en vez de solo accuracy.

## B. Arquitecturas
**9. ¿Qué es un bloque MBConv?**
Es el bloque *inverted residual* de MobileNetV2, al que EfficientNet le añade Squeeze-and-Excitation:
- Conv 1×1 que **expande** los canales (×6).
- Conv **depthwise** 3×3 o 5×5: un filtro por canal, mucho más barata que una convolución estándar.
- **SE**: pooling global → FC → sigmoide, que re-pondera los canales (atención por canal).
- Conv 1×1 de **proyección** lineal a pocos canales.
- Conexión **residual** cuando la entrada y la salida coinciden.
- Activación SiLU/Swish.

**10. ¿Qué es el escalamiento compuesto de EfficientNet?**
En lugar de agrandar solo la profundidad, solo el ancho o solo la resolución, se escalan los tres a la vez con coeficientes
fijos (α·β²·γ² ≈ 2) hallados por búsqueda de arquitectura. B0 es el modelo base: unos 4 M de parámetros (con nuestra cabeza de 5 clases)
y unos 0.39 GFLOPs, con mejor exactitud por parámetro que ResNet-50 (unos 25 M).

**11. ¿Cómo funciona ViT?**
- La imagen 224×224 se divide en 196 *patches* de 16×16.
- Cada patch se aplana y se proyecta linealmente a d = 192.
- Se agrega un token [CLS] aprendible y *embeddings* posicionales.
- 12 bloques Transformer: LayerNorm → **multi-head self-attention** (3 cabezas) → residual → LayerNorm → MLP (768) → residual.
- El [CLS] final pasa a la capa de clasificación.
- La atención es global desde la primera capa, pero el ViT no tiene los sesgos inductivos de localidad e invariancia a la
  traslación de las CNN, por eso necesita más datos o preentrenamiento (aquí: ImageNet-21k).

**12. ¿Qué modificaron respecto a los modelos originales?**
Reemplazamos la capa de clasificación (1000 → 5 salidas) y añadimos dropout antes de ella (0.3 en EfficientNet, 0.1 en ViT).
Todos los pesos se reentrenan (*fine-tuning* completo).

**13. ¿Por qué transfer learning?**
Con unas 1.200 imágenes de entrenamiento, aprender desde cero filtros de bajo nivel (bordes, texturas) es ineficiente. Los
pesos de ImageNet ya traen esos detectores genéricos y solo hay que adaptarlos. La CNN base desde cero lo demuestra
comparativamente.

**14. ¿Por qué una CNN propia como línea base?**
Para medir cuánto aporta el preentrenamiento y la arquitectura moderna frente a una red convencional del tamaño visto en clase.

**15. ¿Qué es Global Average Pooling y por qué usarlo?**
Promedia cada mapa de activación en un solo número. Reduce drásticamente los parámetros frente a un `Flatten` + FC denso
(menos sobreajuste) y hace la red independiente del tamaño espacial de la entrada.

## C. Entrenamiento
**16. ¿Por qué AdamW?**
Adam con *weight decay* desacoplado del gradiente (Loshchilov & Hutter, 2019). En Adam, la regularización L2 se escala por la tasa
adaptativa y no regulariza bien; AdamW la aplica directamente a los pesos. Es el estándar para fine-tuning.

**17. ¿Por qué *warmup* + coseno?**
Al inicio la cabeza es aleatoria y produce gradientes grandes que podrían dañar los pesos preentrenados; el *warmup* lineal
(2 épocas) sube el LR gradualmente. Después, el coseno lo baja suavemente hasta ≈ 0 para converger a un mínimo más estable.

**18. ¿Por qué distintos learning rates por modelo?**
- Desde cero se tolera un LR alto (1e-3).
- Con pesos preentrenados, un LR alto destruye lo aprendido (*catastrophic forgetting*): 3e-4.
- Los ViT son más sensibles a la optimización: 1e-4.

**19. ¿Qué es *label smoothing*?**
Reemplaza la etiqueta one-hot por 0.9 en la clase correcta y 0.1/(K−1) en las demás. Evita que la red empuje los logits a
infinito, reduce la sobreconfianza y suele mejorar la calibración. Por eso la pérdida mínima no es 0.

**20. ¿Cómo eligieron el mejor modelo?**
Con la época de mayor F1-macro en **validación** (*early stopping* con paciencia de 8–10 épocas). El test no participa en
ninguna decisión.

**21. ¿Cómo se identifica sobreajuste en sus curvas?**
La pérdida de entrenamiento sigue bajando mientras la de validación se estanca o sube, y el F1 de entrenamiento se separa del
de validación (la brecha está en la tabla de resumen de entrenamiento). El early stopping toma la época anterior al deterioro.

**22. ¿Por qué la pérdida de validación a veces es menor que la de entrenamiento?**
El entrenamiento usa aumento de datos y dropout (ejemplos más difíciles, red con ruido) y su pérdida es un promedio durante la
época, mientras el modelo todavía cambia. La validación se mide al final de la época, sin aumento y sin dropout.

**23. ¿Qué es la precisión mixta (AMP)?**
Calcular en FP16 en la GPU (Tensor Cores) manteniendo los pesos maestros en FP32. Un `GradScaler` multiplica la pérdida para
evitar que los gradientes pequeños se vuelvan 0 en FP16. Acelera unas 2× y ahorra memoria sin perder exactitud.

## D. Resultados y generalización
**24. ¿Por qué F1-macro y no solo accuracy?**
Con desbalance, la accuracy puede ser alta acertando solo las clases grandes. El F1-macro promedia el F1 de cada clase con el
mismo peso, así que penaliza ignorar cercospora.

**25. ¿Qué mide la AUROC sana vs. enferma?**
La probabilidad de que una hoja enferma elegida al azar reciba un puntaje de enfermedad (1 − p(sana)) mayor que una hoja sana
elegida al azar. No depende del umbral 0.5: 0.5 es azar y 1.0 es separación perfecta. Es útil en RoCoLe, donde el modelo puede
detectar *que hay* enfermedad aunque confunda *cuál*.

**26. ¿Por qué cae el rendimiento en RoCoLe? (pregunta casi segura)**
Hay un cambio de dominio con varias causas:
1. **Fondo:** blanco uniforme vs. vegetación, suelo y otras hojas (el EDA lo cuantifica: casi 0 % de píxeles blancos).
2. **Iluminación:** luz de estudio vs. sol y sombra (el brillo medio baja de 0.62 a 0.41).
3. **Variedad:** robusta tiene hojas más grandes, onduladas y de otro tono que arábica.
4. **Cámara y escala:** distintos celulares y la hoja ocupa una fracción distinta de la foto.
5. **Etiquetas:** RoCoLe solo tiene sana y roya, y grados de severidad propios.
El experimento raw → crop → mask aísla el efecto del fondo, y Grad-CAM muestra a qué píxeles responde el modelo.

**27. ¿Qué demuestra el experimento raw / crop / mask?**
Si la métrica mejora al recortar la hoja o al poner fondo blanco, una parte de la caída se debe al **contexto** (fondo y encuadre)
y no a la enfermedad: el modelo aprendió atajos ligados a la condición de captura de BRACOL. Lo que no se recupera con mask
se atribuye a la variedad, la iluminación y la cámara.

**28. ¿Qué es Grad-CAM?**
Es un mapa de calor de importancia (Selvaraju et al., 2017):
- Se toma el gradiente del logit de la clase predicha respecto a las activaciones de la última capa convolucional.
- Se promedian los gradientes por canal para obtener los pesos α_k.
- Se combinan los mapas de activación como ReLU(Σ α_k · A_k) y se interpola a la resolución de la imagen.
- En el ViT se usan los tokens del último bloque, reorganizados en una cuadrícula de 14×14.

**29. ¿Qué pasa con la arañita roja (una clase que el modelo nunca vio)?**
Un clasificador softmax **siempre** asigna alguna de sus 5 clases; no puede decir "no sé". Revisamos a qué clase la asigna y si
su confianza es menor que en las clases conocidas (detección fuera de distribución por *maximum softmax probability*,
Hendrycks & Gimpel 2017). Es una limitación para el despliegue real.

**30. ¿Cómo se comparan con la literatura?**
Esgario et al. (2020) reportan un 95,24 % de accuracy en la clasificación del estrés predominante sobre las hojas de BRACOL, con
ResNet-50 multitarea (≈ 25 M parámetros). Nosotros obtenemos 87–88 % con modelos de 4–5,5 M. ¿Por qué quedamos por debajo?
- Con el letterbox la hoja (2:1) ocupa la mitad del lienzo, así que la resolución efectiva es menor.
- Entrenamos pocas épocas en CPU, sin búsqueda de hiperparámetros.
- La partición es distinta y hay una sola semilla.
- Su modelo es multitarea y además usa la severidad.

**30b. Si EfficientNet tiene más F1 en test, ¿por qué eligieron el ViT?**
Porque el modelo se elige con **validación**, no con test. Elegir según el test sería usarlo para tomar decisiones (una forma de fuga).
Además, la diferencia en test es de 1–2 puntos, es decir, 3–5 imágenes de 253, dentro del ruido estadístico.

**30c. ¿Por qué el ViT en Grad-CAM mira el fondo y aun así acierta?**
Hay dos posibilidades:
1. Usa atajos correlacionados con la clase en BRACOL (iluminación, sombras, bordes), que en test interno siguen funcionando porque
   tienen la misma distribución.
2. Grad-CAM es menos fiel en Transformers: la atención mezcla información global y los gradientes sobre los tokens no se localizan
   tan bien.

Para distinguir entre ambas haría falta *attention rollout* o pruebas de oclusión.

**30d. ¿Por qué con fotos de campo casi todas las hojas sanas salen «enfermas»?**
El modelo nunca vio suelo, sombras ni otras hojas, porque en BRACOL «sana» equivale a una hoja verde uniforme sobre fondo blanco.
Cualquier textura extra (suelo, manchas de luz, otras hojas) se parece más a los patrones de las clases enfermas. Al poner el fondo
en blanco, el recall de las sanas sube de 4 % a 81 %.

**30e. ¿Por qué el aumento de datos no mejora el test pero sí la generalización?**
El test interno tiene la misma distribución que el entrenamiento, así que memorizar sus condiciones basta. Las rotaciones y los cambios
de color obligan a aprender rasgos invariantes a la orientación y la iluminación, que son justamente los que cambian en campo.
Sin aumento: F1 train 0,99 vs. val 0,91 (sobreajuste) y AUROC en RoCoLe mask 0,59 vs. 0,90.

**31. ¿Cómo mejorarían la generalización?**
- Entrenar con imágenes de campo o mezclar dominios.
- Segmentar la hoja antes de clasificar (pipeline de dos etapas, como sugiere el experimento mask).
- Aumentos de fondo (pegar la hoja sobre fondos naturales: *copy-paste* o *background augmentation*).
- Adaptación de dominio no supervisada.
- Mayor resolución de entrada.
- Clasificación multietiqueta, porque el 18.7 % de las hojas tiene varios estreses.
- Un umbral de rechazo para las entradas fuera de distribución.

**32. ¿Qué limitaciones tiene el trabajo?**
- Una sola partición y una sola semilla (sin intervalos de confianza).
- Etiqueta "predominante" en hojas con varios estreses.
- RoCoLe solo permite evaluar sana y roya.
- Resolución efectiva reducida por el letterbox.
- El modelo no detecta clases desconocidas.
- Sin validación con fotos de fincas colombianas.

**33. ¿Qué hardware usaron y cuánto tardó?**
Intentamos Google Colab, pero no asignó GPU (cuota gratuita agotada), así que todo se entrenó en CPU: un portátil Intel Core i5-1155G7
(4 núcleos / 8 hilos) con 20 GB de RAM, PyTorch 2.15 y timm.
- Tiempo por época: ViT ≈ 2,1 min, EfficientNet-B0 ≈ 3,5 min, CNN base ≈ 3,6 min.
- Total: unas 5,4 h para los 4 modelos.
- Inferencia en CPU: 45 ms/imagen (ViT), 53 ms (EfficientNet), 67 ms (CNN).

**34. ¿Por qué la CNN pequeña es más lenta que EfficientNet en CPU?**
Porque hace convoluciones 3×3 densas a resolución completa (224×224 con 32–64 canales). EfficientNet reduce la resolución
rápidamente y usa convoluciones *depthwise*, que son mucho más baratas en FLOPs.

---

## Correo de entrega
**Para:** profesor del curso · **Asunto:** `ProyectoFinal_Grupo#` (reemplazar # por el número del grupo)

> Buen día, profesor.
> Adjuntamos los entregables del proyecto final del grupo #, "Clasificación de enfermedades foliares del café y
> generalización ante cambio de dominio": (1) diapositivas, (2) código Python (notebooks y carpeta `src/`) y (3) informe final en PDF.
> Integrantes: Cristobal Rodriguez Rojas, Juan Esteban Villada, Cristian David Cabezas Jimenez.
> Saludos.
