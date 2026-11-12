// Genera entregables/Presentacion_ProyectoFinal.pptx a partir de entregables/resultados.json y figures/.
// Uso: node tools/build_deck.js [resultados.json] [salida.pptx] [carpetaFiguras1;carpetaFiguras2]
const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");
const sharp = require("sharp");

const ROOT = path.resolve(__dirname, "..");
const DATA = process.argv[2] || path.join(ROOT, "entregables", "resultados.json");
const OUTF = process.argv[3] || path.join(ROOT, "entregables", "Presentacion_ProyectoFinal.pptx");
const FIGS = (process.argv[4] || path.join(ROOT, "figures")).split(";");
const R = JSON.parse(fs.readFileSync(DATA, "utf8"));
const SKILL = process.env.PPTX_SKILL;
const { CURVE_NOTES, CURVE_SPEECH, TEST_TITLE, TEST_NOTES, TEST_SPEECH, CONF_NOTES, CONF_SPEECH, GEN_TITLE, GEN_NOTES, GEN_SPEECH,
  GRADCAM_ROC_NOTES, GRADCAM_ROC_SPEECH, GRADCAM_BRA_NOTES, GRADCAM_BRA_SPEECH, SEV_NOTE, OOD_NOTE, ABL_NOTE, EXTRA_SPEECH,
  CONCLUSIONS, CONCL_SPEECH } = require("./deck_textos.js");

// ---------------------------------------------------------------- tema
const COL = {
  ink: "1E2A23", coffee: "3B2A20", leaf: "2E7D4F", rust: "C8553D", gold: "C9A227", slate: "5B6C8F",
  plum: "8E5572", sage: "7A8B6F", mist: "EEF2EC", line: "D5DBD3", white: "FFFFFF", mute: "5E6B63", cream: "F3EEE6",
};
const THEME = {
  name: "Cafe y Hoja", headFontFace: "Cambria", bodyFontFace: "Calibri",
  colors: { dk1: COL.ink, lt1: COL.white, dk2: COL.coffee, lt2: COL.mist, accent1: COL.leaf, accent2: COL.rust,
            accent3: COL.gold, accent4: COL.slate, accent5: COL.plum, accent6: COL.sage, hlink: COL.leaf, folHlink: COL.plum },
};
const H = "Cambria", B = "Calibri";

// ---------------------------------------------------------------- utilidades
const pct = (v, d = 1) => (100 * v).toFixed(d).replace(".", ",") + " %";
const num = (v, d = 3) => Number(v).toFixed(d).replace(".", ",");
const NOM = { cnn_base: "CNN base", effnet_b0: "EfficientNet-B0", vit_tiny: "ViT-Tiny", effnet_b0_sinaug: "EffNet-B0 sin aumento" };
const NOMDF = { cnn_base: "CNN base", effnet_b0: "EfficientNet-B0", vit_tiny: "ViT-Tiny", effnet_b0_sinaug: "EffNet-B0 sin aum." };
const BEST = R.best;
const ext = (run, dom) => R.externo.find((e) => e.modelo === NOMDF[run] && e.dominio === dom);
function fig(name) {
  for (const d of FIGS) { const p = path.join(d, name); if (fs.existsSync(p)) return p; }
  throw new Error("Figura no encontrada: " + name);
}
async function imgFit(slide, name, x, y, w, h, opts = {}) {
  const p = fig(name);
  const m = await sharp(p).metadata();
  const r = Math.min(w / m.width, h / m.height);
  const iw = m.width * r, ih = m.height * r;
  slide.addImage({ path: p, x: x + (w - iw) / 2, y: y + (h - ih) / 2, w: iw, h: ih, altText: opts.alt || name, objectName: opts.name || name });
}
async function cropFig(name, out, region) {     // región en fracciones [x0,y0,x1,y1]
  const p = fig(name), m = await sharp(p).metadata();
  const [x0, y0, x1, y1] = region;
  const dst = path.join(ROOT, "figures", "_deck", out);
  fs.mkdirSync(path.dirname(dst), { recursive: true });
  await sharp(p).extract({ left: Math.round(x0 * m.width), top: Math.round(y0 * m.height),
    width: Math.round((x1 - x0) * m.width), height: Math.round((y1 - y0) * m.height) }).toFile(dst);
  return dst;
}
function card(slide, x, y, w, h, fill = COL.mist, name = "tarjeta") {
  slide.addShape("roundRect", { x, y, w, h, fill: { color: fill }, line: { color: fill }, rectRadius: 0.08, objectName: name });
}
function badge(slide, x, y, label, color = COL.leaf, d = 0.42) {
  slide.addShape("ellipse", { x, y, w: d, h: d, fill: { color }, line: { color }, objectName: "insignia " + label });
  slide.addText(label, { x, y, w: d, h: d, align: "center", valign: "middle", fontFace: B, fontSize: 14, bold: true,
    color: COL.white, margin: 0, isTextBox: true });
}
function txt(slide, text, o) {
  slide.addText(text, { fontFace: B, fontSize: 14, color: COL.ink, valign: "top", margin: 0, isTextBox: true, ...o });
}
function bullets(slide, items, o) {
  const runs = items.map((t, i) => ({ text: t, options: { bullet: { indent: 14 }, breakLine: i < items.length - 1, paraSpaceAfter: 6 } }));
  slide.addText(runs, { fontFace: B, fontSize: 14, color: COL.ink, valign: "top", margin: 0, isTextBox: true, ...o });
}
function stat(slide, x, y, w, value, label, color = COL.leaf) {
  txt(slide, value, { x, y, w, h: 0.55, fontFace: H, fontSize: 30, bold: true, color });
  txt(slide, label, { x, y: y + 0.55, w, h: 0.5, fontSize: 11, color: COL.mute });
}
const chartBase = (title) => ({
  showTitle: true, title, titleFontFace: B, titleFontSize: 12, titleColor: COL.ink,
  catAxisLabelColor: COL.mute, valAxisLabelColor: COL.mute, catAxisLabelFontFace: B, valAxisLabelFontFace: B,
  catAxisLabelFontSize: 10, valAxisLabelFontSize: 9, dataLabelFontFace: B, dataLabelFontSize: 9, dataLabelColor: COL.ink,
  valGridLine: { color: "E3E7E2", size: 0.5 }, catGridLine: { style: "none" }, legendFontFace: B, legendFontSize: 10,
  legendColor: COL.ink,
});

(async () => {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_16x9";            // 10 × 5.625 in
  pres.title = "Enfermedades foliares del café: clasificación y generalización";
  pres.author = "Equipo del proyecto final";
  pres.theme = { headFontFace: H, bodyFontFace: B };

  const foot = { text: "Redes Neuronales y Aprendizaje Profundo · EAFIT 2026-2",
    options: { x: 0.5, y: 5.28, w: 6, h: 0.25, fontFace: B, fontSize: 9, color: "8A948D" } };
  pres.defineSlideMaster({
    title: "CONTENIDO", background: { color: COL.white },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.22, w: 9, h: 0.78, fontFace: H, fontSize: 22,
        bold: true, color: COL.coffee, valign: "middle", align: "left", margin: 0, fit: "shrink" }, text: "" } },
      { text: foot },
    ],
    slideNumber: { x: 9.0, y: 5.28, w: 0.5, h: 0.25, fontFace: B, fontSize: 9, color: "8A948D", align: "right" },
  });
  pres.defineSlideMaster({
    title: "OSCURA", background: { color: COL.coffee },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.22, w: 9, h: 0.78, fontFace: H, fontSize: 26,
        bold: true, color: COL.white, valign: "middle", align: "left", margin: 0 }, text: "" } },
    ],
    slideNumber: { x: 9.0, y: 5.28, w: 0.5, h: 0.25, fontFace: B, fontSize: 9, color: "C9BFB5", align: "right" },
  });
  const add = (master, section) => { const s = pres.addSlide({ masterName: master, sectionTitle: section }); s.addNotes = () => s; return s; };

  const D = R.datos, best = R.runs[BEST], bt = R.test[BEST];
  const raw = ext(BEST, "RoCoLe raw"), crop = ext(BEST, "RoCoLe crop"), mask = ext(BEST, "RoCoLe mask"), btest = ext(BEST, "BRACOL test");

  // ============================================================ 1 · Portada
  pres.addSection({ title: "Introducción" });
  let s = add("OSCURA", "Introducción");
  await imgFit(s, "portada_hoja.jpg", 5.6, 0, 4.4, 5.625, { alt: "Hoja de café con roya (BRACOL)", name: "foto portada" });
  txt(s, "PROYECTO FINAL · DEEP LEARNING", { x: 0.5, y: 0.9, w: 5, h: 0.3, fontSize: 12, bold: true, color: COL.gold, charSpacing: 2 });
  txt(s, "¿Funciona en campo un detector de enfermedades del café entrenado en laboratorio?",
    { x: 0.5, y: 1.35, w: 4.9, h: 1.9, fontFace: H, fontSize: 28, bold: true, color: COL.white });
  txt(s, "Clasificación de enfermedades foliares con CNN y Vision Transformers, y evaluación de su generalización ante cambio de dominio",
    { x: 0.5, y: 3.3, w: 4.9, h: 0.8, fontSize: 14, color: "E6DED5" });
  txt(s, "Cristobal Rodriguez Rojas\nJuan Esteban Villada\nCristian David Cabezas Jimenez\nRedes Neuronales y Aprendizaje Profundo · Universidad EAFIT · 2026",
    { x: 0.5, y: 4.2, w: 4.9, h: 0.95, fontSize: 11, color: "C9BFB5" });
  s.addNotes("[Cristobal] Buenos días. Nuestro proyecto responde una pregunta práctica: si entrenamos una red para reconocer enfermedades de la hoja de café con fotos de laboratorio, ¿sigue funcionando con fotos tomadas en campo? Vamos a mostrar el modelo, cómo lo entrenamos y, sobre todo, qué pasa cuando lo sacamos de su dominio.");

  // ============================================================ 2 · Problema
  s = add("CONTENIDO", "Introducción");
  s.addText("La roya puede detectarse por imagen, pero el campo no es un laboratorio", { placeholder: "title" });
  await imgFit(s, "01_bracol_vs_rocole_deck.png", 0.5, 1.1, 5.6, 3.1, { alt: "Comparación BRACOL vs RoCoLe" });
  card(s, 6.4, 1.15, 3.1, 3.95);
  txt(s, "Contexto", { x: 6.6, y: 1.3, w: 2.8, h: 0.3, fontFace: H, fontSize: 16, bold: true, color: COL.coffee });
  bullets(s, [
    "La roya (Hemileia vastatrix) es la enfermedad foliar más dañina del café; en Colombia causó grandes pérdidas en 2008–2011.",
    "Hoy el diagnóstico es visual y depende de un experto.",
    "Los datasets públicos se capturan en condiciones controladas: ¿el modelo aprende la enfermedad o la condición de captura?",
  ], { x: 6.6, y: 1.7, w: 2.75, h: 3.3, fontSize: 12 });
  txt(s, "Arriba: BRACOL (arábica, hoja desprendida, fondo blanco). Abajo: RoCoLe (robusta, hoja en la planta, campo).",
    { x: 0.5, y: 4.25, w: 5.6, h: 0.5, fontSize: 10, color: COL.mute, italic: true });
  s.addNotes("[Cristobal] La roya es la enfermedad foliar más importante del café, y Colombia la sufrió fuertemente entre 2008 y 2011. Detectarla a tiempo permite aplicar control focalizado. Deep Learning es pertinente porque los síntomas son patrones visuales de textura y color difíciles de describir con reglas. El problema es que los datasets públicos se toman en laboratorio: hojas sobre fondo blanco. En campo, la hoja está en la planta con suelo, otras hojas y sol. Esa diferencia es el centro del proyecto.");

  // ============================================================ 3 · Objetivos
  s = add("CONTENIDO", "Introducción");
  s.addText("Objetivos", { placeholder: "title" });
  card(s, 0.5, 1.1, 9, 1.0, COL.coffee);
  txt(s, "OBJETIVO GENERAL", { x: 0.75, y: 1.2, w: 3, h: 0.25, fontSize: 10, bold: true, color: COL.gold, charSpacing: 1 });
  txt(s, "Desarrollar y evaluar un modelo de Deep Learning que clasifique enfermedades y plagas en hojas de café, y medir cuantitativamente su capacidad de generalización a datos de otra región, variedad y condición de captura.",
    { x: 0.75, y: 1.45, w: 8.5, h: 0.6, fontSize: 13, color: COL.white });
  const obj = [
    ["Datos", "Construir un pipeline de preprocesamiento y aumento sobre BRACOL con partición estratificada 70/15/15 sin fuga de información."],
    ["Modelos", "Entrenar y comparar una CNN desde cero, EfficientNet-B0 y ViT-Tiny con accuracy, F1-macro, matriz de confusión y costo computacional."],
    ["Generalización", "Evaluar el mejor modelo en RoCoLe (raw, crop, mask), cuantificar la caída frente al test y explicarla con Grad-CAM."],
  ];
  obj.forEach(([t, d], i) => {
    const x = 0.5 + i * 3.07;
    card(s, x, 2.35, 2.86, 2.3);
    badge(s, x + 0.2, 2.52, String(i + 1), [COL.leaf, COL.slate, COL.rust][i]);
    txt(s, t, { x: x + 0.75, y: 2.56, w: 1.9, h: 0.35, fontFace: H, fontSize: 16, bold: true, color: COL.coffee });
    txt(s, d, { x: x + 0.2, y: 3.12, w: 2.5, h: 1.9, fontSize: 13 });
  });
  s.addNotes("[Cristobal] El objetivo general es desarrollar el clasificador y medir qué tanto generaliza. Lo dividimos en tres objetivos verificables: primero, preparar los datos sin fuga de información; segundo, comparar tres arquitecturas con métricas adecuadas al desbalance; y tercero, llevar el mejor modelo a un dataset externo, medir la caída de rendimiento y explicarla.");

  // ============================================================ 4 · Datos
  pres.addSection({ title: "Datos y método" });
  s = add("CONTENIDO", "Datos y método");
  s.addText("Dos datasets públicos: uno para aprender, otro para ponernos a prueba", { placeholder: "title" });
  const pc = D.por_clase;
  s.addChart(pres.charts.BAR, [{ name: "Imágenes", labels: Object.keys(pc), values: Object.values(pc) }], {
    x: 0.5, y: 1.1, w: 4.6, h: 3.0, barDir: "col", chartColors: [COL.leaf, COL.gold, COL.rust, COL.slate, COL.plum],
    varyColors: true, showValue: true, dataLabelPosition: "outEnd", showLegend: false, valAxisHidden: true,
    ...chartBase(`BRACOL: estrés predominante (n = ${D.bracol_usadas})`), objectName: "grafico clases BRACOL",
  });
  txt(s, `Se descartan ${D.bracol_indeterminadas} hojas «indeterminadas». Desbalance ${(Math.max(...Object.values(pc)) / Math.min(...Object.values(pc))).toFixed(1).replace(".", ",")}×. El ${String(D.multiestres_pct).replace(".", ",")} % de las hojas tiene varios estreses a la vez.`,
    { x: 0.5, y: 4.2, w: 4.6, h: 0.8, fontSize: 11, color: COL.mute });
  const dsCard = (x, title, color, lines) => {
    card(s, x, 1.1, 2.0, 3.95);
    s.addShape("ellipse", { x: x + 0.18, y: 1.27, w: 0.22, h: 0.22, fill: { color }, line: { color }, objectName: "punto " + title });
    txt(s, title, { x: x + 0.5, y: 1.22, w: 1.5, h: 0.32, fontFace: H, fontSize: 16, bold: true, color: COL.coffee });
    bullets(s, lines, { x: x + 0.18, y: 1.7, w: 1.8, h: 3.3, fontSize: 11 });
  };
  dsCard(5.3, "BRACOL", COL.leaf, ["Krohling et al., 2019", "Arábica · Espírito Santo, Brasil", "Hoja desprendida, fondo blanco, 2048×1024", "5 clases + severidad", "Rol: train / val / test"]);
  dsCard(7.5, "RoCoLe", COL.rust, ["Parraga-Alava et al., 2019", "Robusta · Ecuador", "Hoja en la planta, en campo, vertical", `${D.rocole_sana} sanas · ${D.rocole_roya} con roya (4 niveles)`, `${D.rocole_ood} con arañita roja → clase nunca vista`, "Rol: solo evaluación externa"]);
  s.addNotes(`[Cristobal] Entrenamos con BRACOL: ${D.bracol_total} hojas de café arábica de Brasil, fotografiadas desprendidas sobre fondo blanco y etiquetadas por expertos con su estrés predominante. Descartamos las ${D.bracol_indeterminadas} indeterminadas. Hay desbalance: la roya domina y cercospora es minoritaria. Además, casi una de cada cinco hojas tiene varias enfermedades a la vez, lo que vuelve ambigua la etiqueta. Para evaluar la generalización usamos RoCoLe, de Ecuador: café robusta fotografiado en campo, en la planta. Solo comparte con BRACOL las clases sana y roya, y además trae arañita roja, una plaga que nuestro modelo nunca vio.`);

  // ============================================================ 5 · Preprocesamiento
  s = add("CONTENIDO", "Datos y método");
  s.addText("Preprocesamiento: geometría intacta, sin fuga de información", { placeholder: "title" });
  const steps = [["Letterbox 256²", "Conserva la forma de las lesiones; relleno = media ImageNet (≈ 0 tras normalizar)"],
                 ["Aumento", "Recorte aleatorio 0,7–1 · volteos · rotación 90° · ColorJitter ±30 %. Solo en train"],
                 ["224 px + norm.", "Resize y normalización μ/σ de ImageNet (pesos preentrenados)"]];
  steps.forEach(([t, d], i) => {
    const x = 0.5 + i * 3.07;
    card(s, x, 1.1, 2.86, 1.45);
    badge(s, x + 0.18, 1.25, String(i + 1), COL.leaf, 0.36);
    txt(s, t, { x: x + 0.65, y: 1.27, w: 2.1, h: 0.32, fontFace: H, fontSize: 15, bold: true, color: COL.coffee });
    txt(s, d, { x: x + 0.18, y: 1.72, w: 2.55, h: 0.8, fontSize: 11 });
    if (i < 2) s.addShape("rightArrow", { x: x + 2.89, y: 1.68, w: 0.15, h: 0.3, fill: { color: COL.sage }, line: { color: COL.sage }, objectName: "flecha" });
  });
  await imgFit(s, "02_aumento_datos.png", 0.5, 2.7, 5.4, 2.35, { alt: "Ejemplos de letterbox y aumento" });
  const sp = D.split;
  const rows = [["Conjunto", "N.º", "Uso"].map((t) => ({ text: t, options: { bold: true, color: COL.white, fill: { color: COL.coffee } } })),
    ["Train", String(sp.train), "Ajustar pesos"], ["Validación", String(sp.val), "Selección del modelo"],
    ["Test", String(sp.test), "Evaluación final (1 vez)"], ["RoCoLe", String(D.rocole_total), "Solo evaluación externa"]];
  s.addTable(rows, { x: 6.1, y: 2.75, w: 3.4, colW: [0.9, 0.75, 1.75], fontFace: B, fontSize: 10, color: COL.ink,
    border: { type: "solid", pt: 0.5, color: COL.line }, rowH: 0.27, valign: "middle", objectName: "tabla particion" });
  txt(s, "Partición estratificada 70/15/15 (semilla 42, huella MD5 verificada). Pesos de clase calculados solo con train.",
    { x: 6.1, y: 4.3, w: 3.4, h: 0.7, fontSize: 10, color: COL.mute });
  s.addNotes("[Cristobal] Usamos letterbox: redimensionamos conservando la proporción y rellenamos con el color medio de ImageNet, que tras normalizar vale cero. Así no deformamos las manchas, cosa que pasaría al estirar una hoja 2:1. El aumento de datos se aplica solo en entrenamiento: rotaciones de 90 grados porque la orientación de la hoja es arbitraria, recortes y variaciones de color. La partición 70/15/15 es estratificada, con semilla fija; el test se usó una sola vez y RoCoLe nunca participó en el entrenamiento ni en la selección de hiperparámetros.");

  // ============================================================ 6 · Cambio de dominio
  s = add("CONTENIDO", "Datos y método");
  s.addText("El cambio de dominio se puede medir antes de entrenar", { placeholder: "title" });
  await imgFit(s, "01_cambio_de_dominio.png", 0.5, 1.15, 9, 2.45, { alt: "Histogramas de brillo, saturación y fondo blanco" });
  const dm = D.dominio;
  stat(s, 0.6, 3.85, 2.8, `${num(dm.BRACOL.brillo, 2)} → ${num(dm["RoCoLe-raw"].brillo, 2)}`, "brillo medio (BRACOL → RoCoLe)", COL.slate);
  stat(s, 3.6, 3.85, 2.8, `${pct(dm.BRACOL.frac_blanco, 0)} → ${pct(dm["RoCoLe-raw"].frac_blanco, 0)}`, "píxeles de fondo blanco", COL.rust);
  stat(s, 6.6, 3.85, 2.8, "2:1 → 9:16", "relación de aspecto (horizontal → vertical)", COL.leaf);
  s.addNotes("[Cristobal] Antes de entrenar ya podíamos anticipar el problema. Comparamos 400 imágenes de cada dominio: las fotos de campo son mucho más oscuras, casi no tienen fondo blanco y vienen en orientación vertical. Por eso además preparamos dos versiones de RoCoLe usando sus máscaras de segmentación: una recortada a la hoja y otra con el fondo reemplazado por blanco, que imita BRACOL. Con esas tres versiones aislaremos el efecto del fondo. Aquí le paso la palabra a Juan Esteban.");

  // ============================================================ 7 · Arquitecturas
  pres.addSection({ title: "Modelos y entrenamiento" });
  s = add("CONTENIDO", "Modelos y entrenamiento");
  s.addText("Tres arquitecturas: de una CNN propia a un Transformer", { placeholder: "title" });
  // diagrama EfficientNet-B0
  txt(s, `EfficientNet-B0 (Tan & Le, 2019) · ${R.runs.effnet_b0.params_M.toFixed(2).replace(".", ",")} M parámetros`,
    { x: 0.5, y: 1.1, w: 6, h: 0.3, fontFace: H, fontSize: 15, bold: true, color: COL.coffee });
  const blocks = [["Entrada", "3×224×224", COL.sage], ["Stem", "Conv3×3, 32", COL.slate], ["MBConv ×16", "7 etapas · 16→320 canales", COL.leaf],
                  ["Head", "Conv1×1, 1280", COL.slate], ["GAP + Dropout 0,3", "1280", COL.gold], ["FC", "5 clases", COL.rust]];
  blocks.forEach(([t, d, c], i) => {
    const x = 0.5 + i * 1.53;
    s.addShape("roundRect", { x, y: 1.5, w: 1.38, h: 0.85, fill: { color: c }, line: { color: c }, rectRadius: 0.06, objectName: "bloque " + t });
    txt(s, t, { x: x + 0.05, y: 1.56, w: 1.28, h: 0.35, fontSize: 11.5, bold: true, color: COL.white, align: "center" });
    txt(s, d, { x: x + 0.05, y: 1.9, w: 1.28, h: 0.4, fontSize: 9.5, color: COL.white, align: "center" });
    if (i < blocks.length - 1) s.addShape("rightArrow", { x: x + 1.39, y: 1.83, w: 0.13, h: 0.2, fill: { color: COL.mute }, line: { color: COL.mute }, objectName: "flecha" });
  });
  // detalle MBConv
  card(s, 0.5, 2.55, 5.5, 2.5);
  txt(s, "Bloque MBConv (inverted residual + SE)", { x: 0.7, y: 2.65, w: 5, h: 0.3, fontSize: 13, bold: true, color: COL.coffee });
  const mb = ["Conv 1×1\nexpansión ×6", "Depthwise\n3×3 / 5×5", "Squeeze-\nExcitation", "Conv 1×1\nproyección"];
  mb.forEach((t, i) => {
    const x = 0.7 + i * 1.3;
    s.addShape("roundRect", { x, y: 3.1, w: 1.15, h: 0.8, fill: { color: COL.white }, line: { color: COL.leaf, width: 1.25 }, rectRadius: 0.06, objectName: "mbconv " + i });
    txt(s, t, { x, y: 3.17, w: 1.15, h: 0.7, fontSize: 10.5, align: "center", valign: "middle", color: COL.ink });
    if (i < 3) s.addShape("rightArrow", { x: x + 1.16, y: 3.42, w: 0.13, h: 0.18, fill: { color: COL.mute }, line: { color: COL.mute }, objectName: "flecha" });
  });
  s.addShape("line", { x: 0.75, y: 4.08, w: 4.95, h: 0, line: { color: COL.rust, width: 1.25, dashType: "dash", beginArrowType: "none", endArrowType: "triangle" }, objectName: "residual" });
  txt(s, "conexión residual (+)", { x: 2.4, y: 4.12, w: 2, h: 0.25, fontSize: 10, color: COL.rust, align: "center" });
  txt(s, "Depthwise separable = pocos parámetros; SE = atención por canal; activación SiLU.",
    { x: 0.7, y: 4.45, w: 5.1, h: 0.5, fontSize: 10.5, color: COL.mute });
  // otras dos
  card(s, 6.2, 2.55, 3.3, 1.2);
  txt(s, `CNN base · ${R.runs.cnn_base.params_M.toFixed(2).replace(".", ",")} M`, { x: 6.38, y: 2.63, w: 3, h: 0.3, fontSize: 13, bold: true, color: COL.plum });
  txt(s, "4 bloques [Conv3×3-BN-ReLU]×2 + MaxPool (32→256), GAP, FC. Desde cero: referencia.", { x: 6.38, y: 2.95, w: 3, h: 0.75, fontSize: 10.5 });
  card(s, 6.2, 3.85, 3.3, 1.2);
  txt(s, `ViT-Tiny/16 · ${R.runs.vit_tiny.params_M.toFixed(2).replace(".", ",")} M`, { x: 6.38, y: 3.93, w: 3, h: 0.3, fontSize: 13, bold: true, color: COL.slate });
  txt(s, "196 parches 16×16 → 192-d + [CLS]; 12 bloques de self-attention (3 cabezas). Dosovitskiy et al., 2021.", { x: 6.38, y: 4.25, w: 3, h: 0.75, fontSize: 10.5 });
  s.addNotes("[Juan Esteban] Comparamos tres arquitecturas. La línea base es una CNN de cuatro bloques convolucionales entrenada desde cero, para medir cuánto aporta el preentrenamiento. La segunda es EfficientNet-B0: un stem convolucional, dieciséis bloques MBConv y una cabeza que reemplazamos por cinco salidas con dropout. Cada MBConv expande los canales con una convolución 1×1, aplica una convolución depthwise que filtra cada canal por separado, recalibra los canales con squeeze-and-excitation y vuelve a proyectar, con conexión residual; así logra alta exactitud con unos 4 millones de parámetros. La tercera, el Vision Transformer, resultó la mejor en validación y la explico en la siguiente diapositiva. En los tres casos la única modificación a la arquitectura original es la cabeza de clasificación.");

  // ============================================================ 7b · ViT
  s = add("CONTENIDO", "Modelos y entrenamiento");
  s.addText("ViT-Tiny/16: el modelo seleccionado por validación", { placeholder: "title" });
  const vb = [["Imagen", "3×224×224", COL.sage], ["Parches 16×16", "196 parches", COL.slate], ["Proyección lineal", "196 × 192", COL.slate],
              ["+ [CLS] + posición", "197 × 192", COL.gold], ["Encoder ×12", "self-attention", COL.leaf], ["[CLS] → FC", "5 clases", COL.rust]];
  vb.forEach(([t, d, c], i) => {
    const x = 0.5 + i * 1.53;
    s.addShape("roundRect", { x, y: 1.15, w: 1.38, h: 0.85, fill: { color: c }, line: { color: c }, rectRadius: 0.06, objectName: "vit " + t });
    txt(s, t, { x: x + 0.05, y: 1.21, w: 1.28, h: 0.38, fontSize: 11, bold: true, color: COL.white, align: "center" });
    txt(s, d, { x: x + 0.05, y: 1.58, w: 1.28, h: 0.35, fontSize: 9.5, color: COL.white, align: "center" });
    if (i < vb.length - 1) s.addShape("rightArrow", { x: x + 1.39, y: 1.48, w: 0.13, h: 0.2, fill: { color: COL.mute }, line: { color: COL.mute }, objectName: "flecha" });
  });
  card(s, 0.5, 2.2, 5.5, 2.85);
  txt(s, "Bloque del encoder (×12)", { x: 0.7, y: 2.3, w: 5, h: 0.3, fontSize: 13, bold: true, color: COL.coffee });
  const enc = ["LayerNorm", "Multi-head\nself-attention\n(3 cabezas)", "LayerNorm", "MLP 192→768→192\n(GELU)"];
  enc.forEach((t, i) => {
    const x = 0.7 + i * 1.3;
    s.addShape("roundRect", { x, y: 2.75, w: 1.15, h: 0.95, fill: { color: COL.white }, line: { color: COL.slate, width: 1.25 }, rectRadius: 0.06, objectName: "enc " + i });
    txt(s, t, { x, y: 2.8, w: 1.15, h: 0.85, fontSize: 10, align: "center", valign: "middle" });
    if (i < 3) s.addShape("rightArrow", { x: x + 1.16, y: 3.13, w: 0.13, h: 0.18, fill: { color: COL.mute }, line: { color: COL.mute }, objectName: "flecha" });
  });
  txt(s, "+ residual tras la atención y tras el MLP", { x: 0.7, y: 3.8, w: 5.1, h: 0.3, fontSize: 10.5, color: COL.rust });
  txt(s, "Atención: cada parche pondera a todos los demás, softmax(QKᵀ/√d)·V, así que el contexto es global desde la primera capa.",
    { x: 0.7, y: 4.15, w: 5.1, h: 0.8, fontSize: 10.5, color: COL.mute });
  card(s, 6.2, 2.2, 3.3, 2.85);
  txt(s, "¿Por qué funciona aquí?", { x: 6.38, y: 2.3, w: 3, h: 0.3, fontSize: 13, bold: true, color: COL.coffee });
  bullets(s, ["Sin sesgo de localidad: necesita mucho preentrenamiento (ImageNet-21k → 1k), que sí tiene.",
              "Las lesiones están dispersas por la hoja: la atención global las relaciona.",
              "Dosovitskiy et al. (2021). Cambio: cabeza 1000 → 5 con dropout 0,1."],
    { x: 6.38, y: 2.7, w: 3.0, h: 2.3, fontSize: 10.5 });
  s.addNotes("[Juan Esteban] El Vision Transformer trata la imagen como una secuencia. La divide en 196 parches de 16 por 16 píxeles, proyecta cada uno a un vector de 192 dimensiones, agrega un token especial de clasificación y una codificación de posición, y pasa todo por doce bloques de encoder. En cada bloque, la autoatención permite que cada parche pondere a todos los demás: el contexto es global desde la primera capa, a diferencia de una CNN, que ve vecindarios locales. Esa flexibilidad tiene un costo: sin los sesgos de localidad de una CNN necesita mucho preentrenamiento, por eso partimos de pesos de ImageNet-21k. Al final, el token de clasificación pasa a nuestra capa de cinco salidas.");

  // ============================================================ 8 · Configuración experimental
  s = add("CONTENIDO", "Modelos y entrenamiento");
  s.addText("Configuración experimental", { placeholder: "title" });
  const hp = [["Hiperparámetro", "Valor", "Justificación"],
    ["Optimizador", "AdamW", "Weight decay desacoplado; estándar en fine-tuning"],
    ["Learning rate", "1e-3 · 3e-4 · 1e-4", "CNN / EffNet / ViT: preentrenados con LR bajo"],
    ["Programación", "Warmup 2 ép. + coseno", "Estabiliza la cabeza aleatoria al inicio"],
    ["Batch · épocas", "32 · 40/25 máx.", "Early stopping por F1-macro de validación"],
    ["Pérdida", "CE ponderada + LS 0,1", "Desbalance 3,6×; menor sobreconfianza"],
    ["Weight decay", "1e-4 · 1e-2 · 5e-2", "Más regularización al ViT"],
    ["Selección", "Mejor F1-macro val.", "El test no se usa para decidir"]];
  s.addTable(hp.map((r, i) => r.map((t) => ({ text: t, options: i === 0 ? { bold: true, color: COL.white, fill: { color: COL.coffee } } : {} }))),
    { x: 0.5, y: 1.1, w: 6.1, colW: [1.35, 1.7, 3.05], fontFace: B, fontSize: 10.5, color: COL.ink, rowH: 0.42,
      border: { type: "solid", pt: 0.5, color: COL.line }, valign: "middle", objectName: "tabla hiperparametros" });
  card(s, 6.85, 1.1, 2.65, 3.95);
  txt(s, "Equipo de cómputo", { x: 7.05, y: 1.22, w: 2.3, h: 0.3, fontFace: H, fontSize: 15, bold: true, color: COL.coffee });
  const hw = R.hardware;
  const gpu = hw.gpu ? `${hw.gpu} (${hw.gpu_mem_gb} GB)` : "Sin GPU (Colab sin GPU disponible)";
  bullets(s, [hw.gpu ? `GPU: ${gpu}` : "CPU Intel Core i5-1155G7 (4 núcleos / 8 hilos)", `RAM: ${Math.round(hw.ram_gb)} GB`,
    `PyTorch ${String(hw.torch).split("+")[0]} · timm`, `Por época: ViT ${num(R.runs.vit_tiny.s_epoca / 60, 1)} min · EffNet ${num(R.runs.effnet_b0.s_epoca / 60, 1)} min`,
    `Total 4 modelos: ${num(Object.values(R.runs).reduce((a, r) => a + r.train_min, 0) / 60, 1)} h`, ...(hw.gpu ? [] : ["Colab no asignó GPU (cuota gratuita agotada)"])],
    { x: 7.05, y: 1.65, w: 2.3, h: 3.3, fontSize: 11.5 });
  s.addNotes("[Juan Esteban] Usamos AdamW con learning rate distinto por modelo: alto para la red desde cero, bajo para los preentrenados, porque un LR alto destruye los pesos aprendidos en ImageNet, y aún más bajo para el ViT, que es más sensible. Un warmup de dos épocas evita que la cabeza aleatoria dañe el modelo al inicio, y luego el coseno reduce el LR. La pérdida es entropía cruzada ponderada por el inverso de la frecuencia de cada clase, más label smoothing. Elegimos la mejor época por F1-macro de validación. Todo se entrenó en el equipo que ven a la derecha.");

  // ============================================================ 9 · Curvas
  s = add("CONTENIDO", "Modelos y entrenamiento");
  s.addText("Curvas de aprendizaje: convergencia y sobreajuste", { placeholder: "title" });
  await imgFit(s, "02_curvas_comparacion.png", 0.5, 1.1, 6.3, 2.6, { alt: "Curvas de pérdida y F1" });
  const rr = R.runs;
  const gap = (r) => rr[r].train_f1_at_best - rr[r].best_val_f1;
  const tbl = [["Modelo", "Mejor época", "F1 train", "F1 val", "Brecha"].map((t) => ({ text: t, options: { bold: true, color: COL.white, fill: { color: COL.coffee } } }))]
    .concat(["cnn_base", "effnet_b0", "vit_tiny"].map((r) => [NOM[r], `${rr[r].best_epoch}/${rr[r].epochs_run}`, num(rr[r].train_f1_at_best), num(rr[r].best_val_f1), num(gap(r))]));
  s.addTable(tbl, { x: 0.5, y: 3.85, w: 6.3, colW: [1.9, 1.1, 1.1, 1.1, 1.1], fontFace: B, fontSize: 10.5, color: COL.ink, rowH: 0.3,
    border: { type: "solid", pt: 0.5, color: COL.line }, valign: "middle", objectName: "tabla curvas" });
  card(s, 7.05, 1.1, 2.45, 3.95);
  txt(s, "Lectura", { x: 7.22, y: 1.2, w: 2.1, h: 0.3, fontFace: H, fontSize: 15, bold: true, color: COL.coffee });
  bullets(s, CURVE_NOTES(R), { x: 7.22, y: 1.6, w: 2.15, h: 3.4, fontSize: 10.5 });
  s.addNotes("[Juan Esteban] " + CURVE_SPEECH(R));

  // ============================================================ 10 · Resultados test
  pres.addSection({ title: "Resultados" });
  s = add("CONTENIDO", "Resultados");
  s.addText(TEST_TITLE(R), { placeholder: "title" });
  const runs4 = ["cnn_base", "effnet_b0", "vit_tiny", "effnet_b0_sinaug"];
  s.addChart(pres.charts.BAR, [
    { name: "Accuracy", labels: runs4.map((r) => NOM[r]), values: runs4.map((r) => R.test[r].accuracy) },
    { name: "F1-macro", labels: runs4.map((r) => NOM[r]), values: runs4.map((r) => R.test[r].f1_macro) }], {
    x: 0.5, y: 1.1, w: 5.6, h: 3.9, barDir: "col", barGapWidthPct: 60, chartColors: [COL.slate, COL.leaf], showValue: true,
    dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000", valAxisMinVal: 0, valAxisMaxVal: 1, valAxisLabelFormatCode: "0.0",
    showLegend: true, legendPos: "b", ...chartBase(`BRACOL test (n = ${R.n.test})`), objectName: "grafico test",
  });
  const lat = Object.fromEntries(R.latencia.map((l) => [l.modelo, l]));
  const t2 = [["Modelo", "Parám.", "ms/img CPU"].map((t) => ({ text: t, options: { bold: true, color: COL.white, fill: { color: COL.coffee } } }))]
    .concat(runs4.slice(0, 3).map((r) => [NOM[r], `${rr[r].params_M.toFixed(1).replace(".", ",")} M`, num(lat[NOMDF[r]].ms_img_cpu, 0)]));
  s.addTable(t2, { x: 6.35, y: 1.1, w: 3.15, colW: [1.35, 0.8, 1.0], fontFace: B, fontSize: 10.5, color: COL.ink, rowH: 0.3,
    border: { type: "solid", pt: 0.5, color: COL.line }, valign: "middle", objectName: "tabla costo" });
  card(s, 6.35, 2.45, 3.15, 2.6);
  bullets(s, TEST_NOTES(R), { x: 6.52, y: 2.58, w: 2.85, h: 2.4, fontSize: 11 });
  s.addNotes("[Juan Esteban] " + TEST_SPEECH(R));

  // ============================================================ 11 · Matriz de confusión
  s = add("CONTENIDO", "Resultados");
  s.addText("¿Dónde se equivoca? Las hojas con varias enfermedades", { placeholder: "title" });
  await imgFit(s, "03_matriz_confusion_test.png", 0.5, 1.05, 4.2, 4.0, { alt: "Matriz de confusión del mejor modelo" });
  await imgFit(s, "03_errores_test.png", 4.85, 1.05, 4.65, 2.15, { alt: "Ejemplos de errores" });
  card(s, 4.85, 3.3, 4.65, 1.75);
  bullets(s, CONF_NOTES(R), { x: 5.02, y: 3.42, w: 4.35, h: 1.6, fontSize: 11 });
  s.addNotes("[Juan Esteban] " + CONF_SPEECH(R) + " Ahora Cristian presenta la parte central: la generalización.");

  // ============================================================ 12 · Generalización
  pres.addSection({ title: "Generalización" });
  s = add("CONTENIDO", "Generalización");
  s.addText(GEN_TITLE(R), { placeholder: "title" });
  const doms = ["BRACOL test", "RoCoLe raw", "RoCoLe crop", "RoCoLe mask"];
  s.addChart(pres.charts.BAR, [
    { name: "Accuracy 5 clases", labels: doms, values: doms.map((d) => ext(BEST, d).acc_5clases) },
    { name: "AUROC sana vs. enferma", labels: doms, values: doms.map((d) => ext(BEST, d).auroc) }], {
    x: 0.5, y: 1.1, w: 5.9, h: 3.95, barDir: "col", barGapWidthPct: 55, chartColors: [COL.rust, COL.slate], showValue: true,
    dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00", valAxisMinVal: 0, valAxisMaxVal: 1, valAxisLabelFormatCode: "0.0",
    showLegend: true, legendPos: "b", ...chartBase(`${NOM[BEST]}: dentro vs. fuera del dominio (RoCoLe n = ${R.n.rocole_in})`),
    objectName: "grafico generalizacion",
  });
  card(s, 6.6, 1.1, 2.9, 3.95);
  txt(s, "Qué muestra", { x: 6.78, y: 1.2, w: 2.6, h: 0.3, fontFace: H, fontSize: 15, bold: true, color: COL.coffee });
  bullets(s, GEN_NOTES(R), { x: 6.78, y: 1.6, w: 2.6, h: 3.4, fontSize: 11 });
  s.addNotes("[Cristian] " + GEN_SPEECH(R));

  // ============================================================ 13 · Grad-CAM RoCoLe
  s = add("CONTENIDO", "Generalización");
  s.addText("Grad-CAM: ¿mira la hoja o el fondo?", { placeholder: "title" });
  await imgFit(s, "03_gradcam_rocole.png", 0.5, 1.05, 3.4, 4.1, { alt: "Grad-CAM en RoCoLe raw, crop y mask" });
  card(s, 4.1, 1.1, 5.4, 3.95);
  txt(s, "Lectura", { x: 4.3, y: 1.2, w: 5, h: 0.3, fontFace: H, fontSize: 15, bold: true, color: COL.coffee });
  bullets(s, GRADCAM_ROC_NOTES(R), { x: 4.3, y: 1.6, w: 5.0, h: 3.4, fontSize: 13 });
  s.addNotes("[Cristian] " + GRADCAM_ROC_SPEECH(R));

  // ============================================================ 14 · Grad-CAM BRACOL modelos
  s = add("CONTENIDO", "Generalización");
  s.addText("Grad-CAM en BRACOL: una métrica alta no garantiza mirar lo correcto", { placeholder: "title" });
  await imgFit(s, "03_gradcam_bracol.png", 0.5, 1.05, 6.4, 4.05, { alt: "Grad-CAM de los tres modelos en BRACOL" });
  card(s, 7.1, 1.1, 2.4, 3.95);
  bullets(s, GRADCAM_BRA_NOTES(R), { x: 7.27, y: 1.25, w: 2.1, h: 3.7, fontSize: 10.5 });
  s.addNotes("[Cristian] " + GRADCAM_BRA_SPEECH(R));

  // ============================================================ 15 · Severidad, OOD, ablación
  s = add("CONTENIDO", "Generalización");
  s.addText("Severidad, clases nunca vistas y efecto del aumento", { placeholder: "title" });
  const lvlOrder = ["healthy", "rust_level_1", "rust_level_2", "rust_level_3", "rust_level_4"];
  const lvlLab = ["Sana", "Roya 1", "Roya 2", "Roya 3", "Roya 4"];
  const lv = Object.fromEntries(R.por_nivel.map((r) => [r.rocole_class, r]));
  s.addChart(pres.charts.LINE, ["raw", "crop", "mask"].map((v) => ({ name: v, labels: lvlLab, values: lvlOrder.map((k) => lv[k][`% predicho enfermo (${v})`]) })), {
    x: 0.5, y: 1.1, w: 4.4, h: 3.0, chartColors: [COL.rust, COL.gold, COL.slate], lineSize: 2, lineDataSymbol: "circle", lineDataSymbolSize: 6,
    valAxisMinVal: 0, valAxisMaxVal: 1, valAxisLabelFormatCode: "0%", showLegend: true, legendPos: "b",
    ...chartBase("% predicho «enfermo» según nivel de roya"), objectName: "grafico severidad",
  });
  txt(s, SEV_NOTE(R), { x: 0.5, y: 4.2, w: 4.4, h: 0.85, fontSize: 10.5, color: COL.mute });
  card(s, 5.1, 1.1, 4.4, 1.85);
  txt(s, "Arañita roja (nunca vista)", { x: 5.28, y: 1.2, w: 4, h: 0.3, fontFace: H, fontSize: 14, bold: true, color: COL.coffee });
  txt(s, OOD_NOTE(R), { x: 5.28, y: 1.55, w: 4.05, h: 1.35, fontSize: 11 });
  card(s, 5.1, 3.1, 4.4, 1.95);
  txt(s, "Ablación: aumento de datos", { x: 5.28, y: 3.2, w: 4, h: 0.3, fontFace: H, fontSize: 14, bold: true, color: COL.coffee });
  txt(s, ABL_NOTE(R), { x: 5.28, y: 3.55, w: 4.05, h: 1.45, fontSize: 11 });
  s.addNotes("[Cristian] " + EXTRA_SPEECH(R));

  // ============================================================ 16 · Conclusiones
  pres.addSection({ title: "Cierre" });
  s = add("OSCURA", "Cierre");
  s.addText("Conclusiones", { placeholder: "title" });
  const concl = CONCLUSIONS(R);
  concl.forEach(([t, d], i) => {
    const y = 1.1 + i * 0.98;
    badge(s, 0.5, y + 0.05, ["✓", "✓", "✓", "!"][i] || "•", i < 3 ? COL.leaf : COL.rust, 0.4);
    txt(s, t, { x: 1.05, y, w: 4.6, h: 0.3, fontSize: 13.5, bold: true, color: COL.white });
    txt(s, d, { x: 1.05, y: y + 0.32, w: 4.6, h: 0.62, fontSize: 11, color: "E6DED5" });
  });
  s.addShape("roundRect", { x: 6.0, y: 1.1, w: 3.5, h: 3.95, fill: { color: "4A372B" }, line: { color: "4A372B" }, rectRadius: 0.08, objectName: "tarjeta limitaciones" });
  txt(s, "Limitaciones y mejoras", { x: 6.2, y: 1.22, w: 3.1, h: 0.3, fontFace: H, fontSize: 15, bold: true, color: COL.gold });
  bullets(s, [
    "Una sola partición y semilla: sin intervalos de confianza.",
    "Etiqueta «predominante» en hojas con varias enfermedades → probar multietiqueta.",
    "RoCoLe solo evalúa sana y roya; faltan fotos de fincas colombianas.",
    "Segmentar la hoja antes de clasificar, aumentos de fondo y adaptación de dominio.",
    "Rechazo de entradas fuera de distribución.",
    "Entrenado en CPU (sin GPU): pocas épocas y sin búsqueda de hiperparámetros.",
  ], { x: 6.2, y: 1.62, w: 3.15, h: 3.35, fontSize: 11, color: COL.white });
  s.addNotes("[Cristian] " + CONCL_SPEECH(R));

  // ============================================================ 17 · Referencias
  s = add("CONTENIDO", "Cierre");
  s.addText("Referencias", { placeholder: "title" });
  const refs = [
    "Krohling, R. A., Esgario, G. J. M., & Ventura, J. A. (2019). BRACOL – A Brazilian Arabica Coffee Leaf images dataset. Mendeley Data, v1. doi:10.17632/yy2k5y8mxg.1",
    "Parraga-Alava, J., Cusme, K., Loor, A., & Santander, E. (2019). RoCoLe: A robusta coffee leaf images dataset. Data in Brief, 25, 104414. doi:10.17632/c5yvn32dzg.2",
    "Esgario, J. G. M., Krohling, R. A., & Ventura, J. A. (2020). Deep learning for classification and severity estimation of coffee leaf biotic stress. Computers and Electronics in Agriculture, 169, 105162.",
    "Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. ICML.",
    "Dosovitskiy, A., et al. (2021). An image is worth 16×16 words: Transformers for image recognition at scale. ICLR.",
    "Selvaraju, R. R., et al. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. ICCV.",
    "Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization (AdamW). ICLR.",
    "Hendrycks, D., & Gimpel, K. (2017). A baseline for detecting misclassified and out-of-distribution examples in neural networks. ICLR.",
    "Wightman, R. (2019). PyTorch Image Models (timm). github.com/huggingface/pytorch-image-models",
  ];
  bullets(s, refs, { x: 0.5, y: 1.1, w: 9, h: 4.0, fontSize: 10.5 });
  s.addNotes("Referencias de los datasets, las arquitecturas y los métodos utilizados.");

  await pres.writeFile({ fileName: OUTF });
  if (SKILL) {
    const { applyTheme } = require(path.join(SKILL, "scripts", "apply_theme.js"));
    await applyTheme(OUTF, THEME);
  }
  console.log("ok", OUTF);
})().catch((e) => { console.error(e); process.exit(1); });

async function fitBox(p, x, y, w, h) {
  const m = await sharp(p).metadata();
  const r = Math.min(w / m.width, h / m.height);
  return { x: x + (w - m.width * r) / 2, y: y + (h - m.height * r) / 2, w: m.width * r, h: m.height * r };
}

