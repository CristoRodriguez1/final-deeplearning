// ---------------------------------------------------------------- textos que dependen de los resultados
// Redactados a partir de entregables/resultados.json (ejecución real). Todas las cifras se leen de R.
const pct = (v, d = 1) => (100 * v).toFixed(d).replace(".", ",") + " %";
const num = (v, d = 3) => Number(v).toFixed(d).replace(".", ",");
const NOMDF = { cnn_base: "CNN base", effnet_b0: "EfficientNet-B0", vit_tiny: "ViT-Tiny", effnet_b0_sinaug: "EffNet-B0 sin aum." };
const NOM = { cnn_base: "CNN base", effnet_b0: "EfficientNet-B0", vit_tiny: "ViT-Tiny", effnet_b0_sinaug: "EffNet-B0 sin aumento" };
const E = (R, run, dom) => R.externo.find((e) => e.modelo === NOMDF[run] && e.dominio === dom);

function CURVE_NOTES(R) {
  const r = R.runs;
  return [
    `Con aumento no hay sobreajuste: el F1 de train (medido con aumento y dropout) queda en o por debajo del de validación.`,
    `ViT converge más rápido (F1 val ${num(r.vit_tiny.history.val_f1[1], 2)} en la época 2) y llega a ${num(r.vit_tiny.best_val_f1)}.`,
    `La CNN desde cero oscila mucho (caídas a ${num(Math.min(...r.cnn_base.history.val_f1.slice(5)), 2)}): sin preentrenamiento la optimización es inestable.`,
    `Piso de pérdida ≈ 0,55 por el label smoothing.`,
  ];
}
function CURVE_SPEECH(R) {
  const r = R.runs;
  return `Las líneas sólidas son validación y las punteadas entrenamiento. Con aumento de datos no observamos sobreajuste: el F1 de entrenamiento queda incluso por debajo del de validación, porque en entrenamiento la red ve imágenes aumentadas y con dropout. El ViT converge más rápido y alcanza el mejor F1 de validación, ${num(r.vit_tiny.best_val_f1)}, en la época ${r.vit_tiny.best_epoch}; EfficientNet llega a ${num(r.effnet_b0.best_val_f1)}. La CNN desde cero es la más inestable, con caídas bruscas de la validación, y el early stopping la detuvo en la época ${r.cnn_base.epochs_run}. Las curvas siguen subiendo levemente al final: con más épocas o GPU podría ganarse algo más, es decir, hay un leve subajuste. La pérdida no baja de 0,55 por el label smoothing.`;
}
function TEST_TITLE(R) {
  return "En el test interno los modelos preentrenados alcanzan 87–88 % de accuracy";
}
function TEST_NOTES(R) {
  const t = R.test, b = R.best;
  return [
    `Seleccionado por validación: ${NOM[b]} (accuracy ${pct(t[b].accuracy)}, F1-macro ${pct(t[b].f1_macro)}).`,
    `EfficientNet-B0 queda 1–2 puntos arriba en test: ~3–5 imágenes de 253, dentro del ruido.`,
    `Preentrenar suma ${num(100 * (t.effnet_b0.f1_macro - t.cnn_base.f1_macro), 1)} puntos de F1 sobre la CNN desde cero.`,
    `Sana vs. enferma: AUROC 1,00 en test. Esgario et al. (2020): 95,2 % con ResNet-50.`,
  ];
}
function TEST_SPEECH(R) {
  const t = R.test, b = R.best;
  return `En el conjunto de prueba, que nunca participó en ninguna decisión, los modelos preentrenados alcanzan entre 87 y 88 por ciento de accuracy. El modelo que seleccionamos por validación, ${NOM[b]}, obtiene ${pct(t[b].accuracy)} de accuracy y ${pct(t[b].f1_macro)} de F1-macro; EfficientNet queda uno o dos puntos arriba en test, pero con 253 imágenes eso son tres a cinco imágenes, así que no lo consideramos una diferencia significativa y respetamos la selección por validación. El preentrenamiento sí marca diferencia: unos ${num(100 * (t.effnet_b0.f1_macro - t.cnn_base.f1_macro), 0)} puntos de F1 sobre la CNN desde cero. Distinguir sana de enferma es trivial en este dominio: AUROC de 1,0. Nuestro resultado queda por debajo del 95 por ciento de Esgario y colegas, que usaron una ResNet-50 seis veces más grande, otra partición y la hoja a mayor resolución; con el letterbox nuestra hoja ocupa la mitad de la imagen.`;
}
function CONF_NOTES(R) {
  const me = Object.fromEntries(R.multiestres.map((m) => [m.tipo, m]));
  const pc = R.test_por_clase[R.best];
  return [
    `Sana: ${pct(pc.sana.recall, 0)} de recall; phoma: F1 ${num(pc.phoma["f1-score"], 2)}.`,
    `Cercospora es la más difícil (F1 ${num(pc.cercospora["f1-score"], 2)}, solo ${pc.cercospora.support} imágenes): se confunde con minador y roya.`,
    `Accuracy con un solo estrés: ${pct(me["un estrés"].accuracy, 0)}; con varios estreses: ${pct(me["varios estreses"].accuracy, 0)}.`,
  ];
}
function CONF_SPEECH(R) {
  const me = Object.fromEntries(R.multiestres.map((m) => [m.tipo, m]));
  const pc = R.test_por_clase[R.best];
  return `La matriz de confusión muestra que el modelo nunca confunde una hoja sana con una enferma: los errores son entre enfermedades. La clase más difícil es cercospora, con F1 de ${num(pc.cercospora["f1-score"], 2)}: es la minoritaria y sus manchas se parecen a las de minador y roya. El hallazgo más interesante: en hojas con una sola enfermedad el modelo acierta el ${pct(me["un estrés"].accuracy, 0)}, pero en hojas con varias enfermedades solo el ${pct(me["varios estreses"].accuracy, 0)}. Ahí la etiqueta «estrés predominante» es ambigua, lo que sugiere tratar el problema como multietiqueta.`;
}
function GEN_TITLE(R) {
  const b = E(R, R.best, "BRACOL test"), raw = E(R, R.best, "RoCoLe raw"), mask = E(R, R.best, "RoCoLe mask");
  return `En campo la accuracy cae de ${pct(b.acc_5clases, 0)} a ${pct(raw.acc_5clases, 0)}; sin fondo vuelve a ${pct(mask.acc_5clases, 0)}`;
}
function GEN_NOTES(R) {
  const raw = E(R, R.best, "RoCoLe raw"), mask = E(R, R.best, "RoCoLe mask");
  const rawA = ["cnn_base", "effnet_b0", "vit_tiny"].map((r) => E(R, r, "RoCoLe raw").auroc);
  const maskA = ["cnn_base", "effnet_b0", "vit_tiny"].map((r) => E(R, r, "RoCoLe mask").auroc);
  return [
    `En raw, el ${pct(1 - raw.recall_sana, 0)} de las hojas sanas se predice «enferma»: el fondo de campo parece enfermedad.`,
    `Con mask: recall sana ${pct(mask.recall_sana, 0)}, recall enferma ${pct(mask.recall_enferma, 0)}, AUROC ${num(mask.auroc, 2)}.`,
    `Mismo patrón en los 3 modelos: AUROC raw ${num(Math.min(...rawA), 2)}–${num(Math.max(...rawA), 2)} → mask ${num(Math.min(...maskA), 2)}–${num(Math.max(...maskA), 2)}.`,
    `El resto de la brecha: variedad, luz y cámara.`,
  ];
}
function GEN_SPEECH(R) {
  const b = E(R, R.best, "BRACOL test"), raw = E(R, R.best, "RoCoLe raw"), crop = E(R, R.best, "RoCoLe crop"), mask = E(R, R.best, "RoCoLe mask");
  return `Esta es la pregunta central del proyecto. El mismo modelo que acierta el ${pct(b.acc_5clases, 0)} en el test interno solo acierta el ${pct(raw.acc_5clases, 0)} con fotos de campo: la generalización se degrada fuertemente. ¿Por qué? Casi todas las hojas sanas de campo, el ${pct(1 - raw.recall_sana, 0)}, se predicen como enfermas: el modelo nunca vio suelo, sombras ni otras hojas, y los interpreta como síntomas. Recortar la hoja ayuda poco, ${pct(crop.acc_5clases, 0)}, porque queda fondo en el recorte. Pero si ponemos el fondo en blanco, la accuracy sube a ${pct(mask.acc_5clases, 0)} y la AUROC sana-enferma pasa de ${num(raw.auroc, 2)} a ${num(mask.auroc, 2)}. Los tres modelos muestran el mismo patrón. Concluimos que la mayor parte de la caída se debe al fondo y la condición de captura; lo que no se recupera se atribuye a la variedad robusta, la iluminación natural y la cámara.`;
}
function GRADCAM_ROC_NOTES(R) {
  return [
    "Grad-CAM pondera los mapas de la última capa con el gradiente de la clase predicha (Selvaraju et al., 2017); en el ViT, tokens 14×14.",
    "raw: la activación se reparte entre la hoja y el fondo (suelo, ramas, otras hojas).",
    "crop: el borde del recorte y el relleno atraen la atención.",
    "mask: la activación se concentra sobre la lámina y la hoja sana se clasifica bien.",
    "→ parte del error es un atajo aprendido de la condición de captura.",
  ];
}
function GRADCAM_ROC_SPEECH(R) {
  return "Para entender el porqué usamos Grad-CAM, que muestra qué regiones empujaron la predicción. En la foto de campo original la activación se reparte entre la hoja y el fondo; en la tercera fila, una hoja sana se clasifica como roya mirando sobre todo otras hojas y el suelo. Al recortar, la atención se va a los bordes del recorte. Con el fondo en blanco, la activación vuelve a la hoja y la hoja sana se clasifica correctamente. Es evidencia directa de que el modelo usa el contexto de la imagen y no solo la lesión.";
}
function GRADCAM_BRA_NOTES(R) {
  return [
    "La CNN base marca con precisión las pústulas de roya, las galerías del minador y la lesión de phoma.",
    "EfficientNet-B0 activa regiones amplias de la hoja que incluyen las lesiones.",
    "ViT-Tiny, aunque es el mejor en validación, activa a menudo el fondo y los bordes: posible atajo (o menor fidelidad de Grad-CAM en Transformers).",
    "Una métrica alta no garantiza que el modelo mire lo correcto.",
  ];
}
function GRADCAM_BRA_SPEECH(R) {
  return "En el dominio de entrenamiento comparamos dónde mira cada arquitectura en hojas bien clasificadas. Sorprendentemente, la CNN desde cero es la que mejor localiza las lesiones: las pústulas de roya, las galerías del minador y la lesión de phoma. EfficientNet activa regiones amplias de la hoja. El ViT, que fue el mejor en validación, activa con frecuencia el fondo y los bordes. Puede ser un atajo, o una limitación de Grad-CAM en Transformers, donde es menos fiel. La lección es que una métrica alta no garantiza que el modelo use la evidencia correcta.";
}
function SEV_NOTE(R) {
  const lv = Object.fromEntries(R.por_nivel.map((r) => [r.rocole_class, r]));
  const k = "% predicho enfermo (mask)";
  return `Con mask: ${pct(lv.rust_level_1[k], 0)} de detección en roya nivel 1 y ≥ ${pct(Math.min(lv.rust_level_2[k], lv.rust_level_3[k], lv.rust_level_4[k]), 0)} en niveles 2–4; falsos positivos en sanas: ${pct(lv.healthy[k], 0)}. En raw todo se predice «enfermo».`;
}
function OOD_NOTE(R) {
  const o = R.ood.find((x) => x["versión"] === "raw");
  const order = ["sana", "minador", "roya", "phoma", "cercospora"].map((c) => [c, o[c]]).sort((a, b) => b[1] - a[1]);
  return `Sin opción «no sé», el softmax la asigna a clases conocidas: ${pct(order[0][1], 0)} ${order[0][0]}, ${pct(order[1][1], 0)} ${order[1][0]}. Su confianza (${num(o["confianza media OOD"], 2)}) es igual a la de sana/roya (${num(o["confianza media sana/roya"], 2)}): AUROC de detección = ${num(o["AUROC detección OOD (MSP)"], 2)} (azar).`;
}
function ABL_NOTE(R) {
  const t1 = R.test.effnet_b0.f1_macro, t0 = R.test.effnet_b0_sinaug.f1_macro;
  const m1 = E(R, "effnet_b0", "RoCoLe mask").auroc, m0 = E(R, "effnet_b0_sinaug", "RoCoLe mask").auroc;
  const r1 = E(R, "effnet_b0", "RoCoLe raw").auroc, r0 = E(R, "effnet_b0_sinaug", "RoCoLe raw").auroc;
  return `Sin aumento, el F1 en test casi no cambia (${pct(t0)} vs. ${pct(t1)}), pero la AUROC en RoCoLe se desploma: mask ${num(m0, 2)} vs. ${num(m1, 2)}; raw ${num(r0, 2)} vs. ${num(r1, 2)}. El aumento no mejora el test interno: decide la generalización.`;
}
function EXTRA_SPEECH(R) {
  const lv = Object.fromEntries(R.por_nivel.map((r) => [r.rocole_class, r]));
  const o = R.ood.find((x) => x["versión"] === "raw");
  return `Tres análisis complementarios. Primero, la severidad: con el fondo removido, la roya leve de nivel 1 se detecta en el ${pct(lv.rust_level_1["% predicho enfermo (mask)"], 0)} de los casos y los niveles 2 a 4 prácticamente siempre. Segundo, la arañita roja, una plaga que el modelo nunca vio: como un softmax no puede decir «no sé», la asigna sobre todo a roya, y con la misma confianza que a las clases conocidas, así que ni siquiera la confianza sirve para detectarla. Tercero, la ablación: sin aumento de datos el F1 en test casi no cambia, pero la generalización a campo colapsa. Este es quizá el hallazgo más útil: el aumento de datos no se nota en el test interno, pero es lo que permite generalizar.`;
}
function CONCLUSIONS(R) {
  const t = R.test, b = E(R, R.best, "BRACOL test"), raw = E(R, R.best, "RoCoLe raw"), mask = E(R, R.best, "RoCoLe mask");
  return [
    ["Objetivo 1 — pipeline sin fuga", "Partición verificada por huella, aumento solo en train, test usado una vez, RoCoLe solo para evaluar."],
    ["Objetivo 2 — preentrenar gana", `ViT-Tiny y EfficientNet-B0: 87–88 % de accuracy en test, frente a 84 % de la CNN desde cero.`],
    ["Objetivo 3 — la generalización se degrada", `${pct(b.acc_5clases, 0)} → ${pct(raw.acc_5clases, 0)} en campo; quitar el fondo la sube a ${pct(mask.acc_5clases, 0)} (Grad-CAM lo confirma).`],
    ["Un buen test no es un modelo listo para campo", "El aumento de datos y la segmentación de la hoja deciden la robustez; hay que validar con fotos de campo."],
  ];
}
function CONCL_SPEECH(R) {
  return "Cumplimos los tres objetivos. Construimos un pipeline reproducible y sin fuga de información. Los modelos preentrenados superan a la red desde cero y llegan a 87–88 por ciento en el test interno. Y medimos que en campo el rendimiento se desploma, con evidencia cuantitativa y visual de que el fondo es la causa principal. Nuestra conclusión central: una métrica alta en el test interno no garantiza un modelo útil en campo. Como limitaciones: una sola partición y semilla, sin intervalos de confianza; etiquetas ambiguas en hojas con varias enfermedades; RoCoLe solo evalúa sana y roya; y entrenamos en CPU con pocas épocas. Para mejorar, proponemos segmentar la hoja antes de clasificar, ya que mask lo demuestra; entrenar con fotos de campo; un enfoque multietiqueta; y rechazar entradas desconocidas. Gracias; quedamos atentos a sus preguntas.";
}

module.exports = { CURVE_NOTES, CURVE_SPEECH, TEST_TITLE, TEST_NOTES, TEST_SPEECH, CONF_NOTES, CONF_SPEECH, GEN_TITLE, GEN_NOTES,
  GEN_SPEECH, GRADCAM_ROC_NOTES, GRADCAM_ROC_SPEECH, GRADCAM_BRA_NOTES, GRADCAM_BRA_SPEECH, SEV_NOTE, OOD_NOTE, ABL_NOTE,
  EXTRA_SPEECH, CONCLUSIONS, CONCL_SPEECH };
