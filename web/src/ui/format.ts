/** Numbers the way a paper writes them: 3.2·10⁻⁷, not 3.2e-7. */

const SUP: Record<string, string> = { "-": "⁻", "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹" };

export function sci(v: number, digits = 3): string {
  if (!Number.isFinite(v)) return Number.isNaN(v) ? "—" : v > 0 ? "∞" : "−∞";
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a >= 1e-3 && a < 1e5) return fixMinus(Number(v.toPrecision(digits + 1)).toString());
  const e = Math.floor(Math.log10(a));
  const m = v / 10 ** e;
  return `${fixMinus(m.toFixed(digits - 1))}·10${String(e).split("").map((c) => SUP[c]).join("")}`;
}

/** Thousands separated by thin spaces: 1 284 000 */
export function count(v: number): string {
  return Math.round(v).toString().replace(/\B(?=(\d{3})+(?!\d))/g, " ");
}

const fixMinus = (s: string) => s.replace("-", "−");
