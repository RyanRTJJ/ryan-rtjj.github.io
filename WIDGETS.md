# Embedding interactive widgets in posts

This site is **Jekyll + the Chirpy theme** (consumed as a gem, not vendored). Posts are
markdown in `_posts/`, rendered by **kramdown**, and deployed with:

```bash
bundle exec jekyll s                              # local dev w/ hot reload
JEKYLL_ENV=production bundle exec jekyll b         # production build
firebase deploy --only hosting                     # publish
```

kramdown passes **raw HTML straight through**, so interactive widgets (custom SVG
animations, JS/HTML interactions, data-viz with D3/Plotly/Chart.js, `<canvas>` games,
`<iframe>` embeds) work by dropping HTML/JS into a post — no extra build tooling, no
migration. Several existing posts already embed raw HTML this way.

> **For a future Claude instance:** when the user asks for a new widget, prefer the
> reusable-include pattern below, mind the three gotchas, and test with
> `bundle exec jekyll s` before declaring success.

---

## The three gotchas

### 1. Liquid eats `{{ }}` and `{% %}`
Jekyll runs **Liquid** templating over every file before kramdown. JS object literals,
template literals (`` `${x}` ``), and double braces collide with Liquid's `{{ }}` / `{% %}`
syntax and get mangled or throw build errors.

**Fix:** wrap any script containing those tokens in `{% raw %} ... {% endraw %}`.

```html
{% raw %}
<script>
  const msg = `score: ${score}`;   // safe inside raw
</script>
{% endraw %}
```

### 2. kramdown parses markdown *inside* block-level HTML
Text inside a `<div>` may be re-interpreted as markdown (e.g. `*x*` becomes italics,
indented lines become code blocks), mangling widget content.

**Fix:** add `markdown="0"` to the wrapping element.

```html
<div markdown="0">
  ...widget markup that should NOT be touched by markdown...
</div>
```

### 3. Loading JS libraries
Two options:

- **Per-post (simplest):** put `<script src="https://cdn...">` at the top of the post.
  The library loads only on pages that use it.
- **Site-wide (for reuse across many posts):** override Chirpy's `_includes/head.html`.
  Since the theme is a gem, copy the file out of the gem and edit the local copy —
  Jekyll prefers local `_includes`/`_layouts` over the gem's automatically:

  ```bash
  cp "$(bundle show jekyll-theme-chirpy)/_includes/head.html" _includes/head.html
  ```

---

## Recommended pattern for reusable widgets

For anything you might reuse or want versioned in one place, make it an **include**, not
inline post HTML.

1. Create `_includes/widgets/<name>.html` containing the markup + scoped CSS + JS.
2. Reference it from any post with:

   ```liquid
   {% include widgets/<name>.html %}
   ```

3. Pass parameters via include variables, read inside the widget as `{{ include.foo }}`:

   ```liquid
   {% include widgets/<name>.html size="400" theme="dark" %}
   ```

### Conventions for widget includes

- **Scope your CSS and JS.** Multiple widgets (or multiple instances of one widget) can
  appear on the same page. Namespace classes (e.g. `.ttt-cell`, not `.cell`) and avoid
  global JS variable names — wrap logic in an IIFE or a uniquely-named init function.
- **Support multiple instances** when reasonable: generate a unique id per include
  (e.g. from `{{ include.id }}` or an incrementing counter) so two embeds don't clash.
- **Wrap JS in `{% raw %}`** if it uses `{{ }}` / `${}` / `{% %}` (gotcha #1). If you also
  need a Liquid variable inside that script, close `raw` around just the dynamic bit.
- **Keep it dependency-light** for small widgets — vanilla JS + inline `<style>` avoids
  the library-loading question entirely.
- **One-offs** that won't be reused can stay inline in the post; the include pattern is
  for things worth keeping in one canonical place.

### Reference example

`_includes/widgets/tic-tac-toe.html` is a self-contained, dependency-free reference
widget (scoped CSS, namespaced JS, multi-instance safe). Embed with:

```liquid
{% include widgets/tic-tac-toe.html %}
```

A demo post lives at `_posts/2026-06-10-widget-demo.md` (`published: false`).
