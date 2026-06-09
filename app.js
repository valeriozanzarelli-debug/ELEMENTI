// js/app.js (bootstrap) - safe loader

const CE_FILES = [
  "/custom-elements/ic-hero.ce.js",
  "/custom-elements/metodo.ce.js",
  "/custom-elements/preventivo.ce.js",
  "/custom-elements/galleria.ce.js",
  "/custom-elements/prenotazioni.ce.js",
  "/custom-elements/costa-calma.ce.js",
  "/custom-elements/domande.ce.js",
  "/custom-elements/contatti.ce.js",
];

await Promise.allSettled(CE_FILES.map((p) => import(p)));

// poi parte la home logic
await import("/js/main.js");