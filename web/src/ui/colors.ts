/** Run colours, in order of adding: chosen to stand apart on the viridis surface (trails also get a halo) and in
 *  both themes. Shared by the renderer (rgb, 0–1) and the UI (css). */
export const RUN_COLORS: { name: string; css: string; rgb: [number, number, number] }[] = [
  { name: "vermilion", css: "#d93d1a", rgb: [0.85, 0.24, 0.1] },
  { name: "blue", css: "#1f6bd9", rgb: [0.12, 0.42, 0.85] },
  { name: "amber", css: "#ed9e00", rgb: [0.93, 0.62, 0.0] },
  { name: "magenta", css: "#bf3399", rgb: [0.75, 0.2, 0.6] },
  { name: "teal", css: "#0d998c", rgb: [0.05, 0.6, 0.55] },
  { name: "violet", css: "#734dbf", rgb: [0.45, 0.3, 0.75] },
  { name: "brown", css: "#8c6b40", rgb: [0.55, 0.42, 0.25] },
  { name: "graphite", css: "#595961", rgb: [0.35, 0.35, 0.38] },
];
