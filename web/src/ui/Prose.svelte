<script lang="ts">
  import katex from "katex";
  /** Prose with inline TeX between $ signs. */
  let { text }: { text: string } = $props();
  const html = $derived(
    text
      .split(/(\$[^$]+\$)/)
      .map((part) =>
        part.startsWith("$") && part.endsWith("$") && part.length > 1
          ? katex.renderToString(part.slice(1, -1), { throwOnError: false })
          : part.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" })[c]!),
      )
      .join(""),
  );
</script>

<span>{@html html}</span>
