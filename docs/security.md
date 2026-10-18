# Security

optiviz is a static web app: no accounts, no backend, no cookies, no storage of personal data. Everything runs in
the visitor's browser. The attack surface is what the page accepts from outside (share links) and what it
loads.

## Audit, 2026-10-19 (before making the repository public)

**Secrets and history.** All 21 commits scanned for credentials (GitHub/AWS/Google/Slack/OpenAI/Anthropic token
formats, private keys, password/secret/API-key assignments): none. Commits use a GitHub no-reply address. No
hostnames, IP addresses, tunnel IDs or home-directory paths in any tracked file or diff. Tracked files are source,
docs and lockfiles only (95 files, largest 52 KB).

**Dependencies.** `cargo audit` (RustSec, 1 338 advisories, 143 crates): no vulnerabilities. `pnpm audit`: no
known vulnerabilities. The only runtime npm dependency is KaTeX; Svelte compiles away.

**Untrusted input: share links.** The URL hash is parsed as JSON and only ever passed to the engine as data:
- problem and optimizer ids are looked up in fixed registries (unknown ids are ignored);
- coefficient values are clamped to their declared ranges, non-finite values replaced by the default
  (`problems::build`); parameter values likewise clamped, non-finite refused (`Run::set`, tested);
- runs are capped at 8, hill counts at 30, stress points at the frame capacity;
- nothing from the hash is rendered as HTML. Text the UI renders with `{@html}` is KaTeX output for TeX that
  comes from the compiled-in manifests (KaTeX's `trust` stays off), or prose whose non-TeX parts are escaped.

**Shader code.** The surface shaders splice in a problem's formula. The formula is built in Rust from fixed
templates and coefficients formatted as float literals (`lit`), so no input text can reach the shader. User-typed
functions (planned, DESIGN.md §6.6) must keep this property: parse to an expression graph and emit code from it,
never splice text.

**Headers.** The production preview (and any hosting) sends: COOP `same-origin` + COEP `require-corp` (needed for
SharedArrayBuffer), a Content-Security-Policy (`default-src 'self'`; scripts `'self' 'wasm-unsafe-eval'`;
styles `'self' 'unsafe-inline'` for KaTeX's inline style attributes; images and fonts `'self' data:`;
`object-src 'none'`, `base-uri 'none'`, `form-action 'none'`, `frame-ancestors 'none'`), CORP `same-origin`,
`X-Content-Type-Options: nosniff`, `Referrer-Policy: no-referrer`. The browser test suite runs against these
headers. The dev server only binds localhost.

**CI.** The workflow token is read-only (`permissions: contents: read`) and every third-party action is pinned
to a commit SHA.

## Reporting

Please open a GitHub issue, or for anything sensitive contact the owner through GitHub before publishing details.
