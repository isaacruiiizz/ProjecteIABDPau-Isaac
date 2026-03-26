/**
 * Conversió RGB → HSV en convenció OpenCV:
 *   H ∈ [0..179]   (la meitat del cercle cromàtic per cabre en uint8)
 *   S ∈ [0..255]
 *   V ∈ [0..255]
 *
 * Implementació pura TS, sense dependències externes.
 * Compatible amb cv2.cvtColor(COLOR_RGB2HSV) usat a jersey_classifier.py.
 */
export function rgbToHsv(r: number, g: number, b: number): [number, number, number] {
  const rn = r / 255;
  const gn = g / 255;
  const bn = b / 255;

  const max = Math.max(rn, gn, bn);
  const min = Math.min(rn, gn, bn);
  const delta = max - min;

  // Value
  const v = Math.round(max * 255);

  // Saturation
  const s = max === 0 ? 0 : Math.round((delta / max) * 255);

  // Hue
  let h = 0;
  if (delta !== 0) {
    if (max === rn) {
      h = ((gn - bn) / delta) % 6;
    } else if (max === gn) {
      h = (bn - rn) / delta + 2;
    } else {
      h = (rn - gn) / delta + 4;
    }
    h = h * 60;
    if (h < 0) h += 360;
    // Escalar a [0..179] (convenció OpenCV)
    h = Math.round((h / 360) * 179);
  }

  return [h, s, v];
}

/** Retorna un string CSS rgb() a partir de valors [0..255]. */
export function rgbToCss(r: number, g: number, b: number): string {
  return `rgb(${r},${g},${b})`;
}
