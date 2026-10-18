<script lang="ts">
  import katex from "katex";
  let { tex, display = false }: { tex: string; display?: boolean } = $props();
  const html = $derived(katex.renderToString(tex, { throwOnError: false, displayMode: display }));
</script>

<span class="tex">{@html html}</span>
