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

const FROM_SUP: Record<string, string> = { "⁻": "-", "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9" };

/** `sci` as TeX: 3.20\cdot 10^{-7}. */
export function sciTex(v: number, digits = 3): string {
  const s = sci(v, digits);
  const i = s.indexOf("·10");
  if (i < 0) return s.replace("−", "-");
  const exp = [...s.slice(i + 3)].map((c) => FROM_SUP[c] ?? c).join("");
  return `${s.slice(0, i).replace("−", "-")}\\cdot 10^{${exp}}`;
}

/** `count` as TeX (thin spaces as \,). */
export function countTex(v: number): string {
  return Math.round(v).toString().replace(/\B(?=(\d{3})+(?!\d))/g, "\\,");
}
