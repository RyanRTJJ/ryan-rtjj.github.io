---
published: false
title: Taxonomy of Features
date: 2026-06-30 00:00:00 -500
categories: [mechanistic interpretability]
tags: [mechanistic interpretability]
math: true
---

In the mech interp world, what constitutes a "feature" has long eluded definition. Here are some problems that result from that:

### Auto-interp pipelines become untrustworthy and unscalable
Auto-interp pipelines refer to automatic methods to try and generate explanations for features (usually some form of Sparse Auto-Encoder (SAE) features). The way to do this is, once an SAE has been trained, the model (with the SAE attached) runs the forward pass on some amount of data to collect positive and negative examples, where positive examples are texts where a particular SAE feature activates highly, and negative examples are texts where that feature does not activate (or activates with insignificant magnitude). These positive and negative examples are then passed through an LLM, which is asked to generate possible meanings that this SAE feature could be probing for. The problem with this is that the SAE feature could truly activate on a concatenation of a large list of meanings, such as (real example):

<style>
  .ft-conj { --DARK_PAPER: #fff9f2; --BOOK_CLOTH: #cc785c; margin: 1.5rem 0; text-align: center; }
  .ft-conj svg { width: 100%; height: auto; max-width: 760px; }
  .ft-conj .ft-card { fill: var(--DARK_PAPER); stroke: var(--BOOK_CLOTH); stroke-width: 1.5; }
  .ft-conj .ft-label { fill: #000; font-size: 20px; font-weight: 400; font-family: "Colfax AI", "Source Sans Pro", sans-serif; }
  .ft-conj .ft-and { fill: var(--BOOK_CLOTH); font-size: 28px; font-weight: 700; font-family: "Colfax AI", "Source Sans Pro", sans-serif; }
</style>

<figure class="ft-conj" markdown="0">
  <svg viewBox="0 0 1040 140" role="img" aria-label="Two feature cards joined by a logical AND, then a trailing 'and so on': 'word that starts with S' and 'this word does not mean short', followed by '&& ...'.">
    <rect class="ft-card" x="20" y="40" width="320" height="50" rx="8" ry="8"/>
    <text class="ft-label" x="180" y="67" text-anchor="middle" dominant-baseline="middle">"word that starts with S"</text>

    <text class="ft-and" x="400" y="67" text-anchor="middle" dominant-baseline="middle">&amp;&amp;</text>

    <rect class="ft-card" x="460" y="40" width="400" height="50" rx="8" ry="8"/>
    <text class="ft-label" x="660" y="67" text-anchor="middle" dominant-baseline="middle">"this word does not mean 'short'"</text>

    <text class="ft-and" x="920" y="67" text-anchor="middle" dominant-baseline="middle">&amp;&amp;</text>

    <text class="ft-and" x="980" y="67" text-anchor="start" dominant-baseline="middle">...</text>
  </svg>
</figure>

If you simply did not sample a piece of text that contained the word "short", the generated explanation might just describe the feature as `"word that starts with S"`, which would be **incomplete**. Either you have to sample a laughably large number of texts to generate a reasonably well-covered explanation (which could well be a very unwieldy statement).


### Directions in weight space don't have to align with human-interpretable concepts
Because training dynamics favor statistically convenient representations, neurons (for LLMs and SAEs alike) don't have to have "clean" (read: concise / simple / terse / composable) meanings. For example, if a small model was trained on apartment listings to predict rent, a very strong predictor may have been not number of bathrooms or bedrooms, but specifically how close the ratio of bathrooms to bedroom was to 0.8 (arbitrary). 

<style>
  .ft-flow { --DARK_PAPER: #fff9f2; --MANILA: #ebdbbc; --KRAFT_LIGHT: #faeee1; --KRAFT: #d4a27f; --BOOK_CLOTH: #cc785c; margin: 1.5rem 0; text-align: center; }
  .ft-flow svg { width: 100%; height: auto; max-width: 700px; font-family: "Colfax AI", "Source Sans Pro", sans-serif; }
  .ft-flow .ft-icon { fill: var(--DARK_PAPER); stroke: var(--BOOK_CLOTH); stroke-width: 1.5; }
  .ft-flow .ft-pnl { fill: none; stroke: var(--BOOK_CLOTH); stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
  .ft-flow .ft-vecbox { stroke: none; }
  .ft-flow .ft-arrow { stroke: #b0b0b0; stroke-width: 2; fill: none; }
  .ft-flow .ft-arrow-light { stroke: #e6e6e6; stroke-width: 1.5; fill: none; }
  .ft-flow .ft-arrow-clay { stroke: var(--BOOK_CLOTH); stroke-width: 2; fill: none; }
  .ft-flow .ft-dots { font-size: 22px; font-weight: 700; fill: var(--text-color, #1f2328); }
  .ft-flow .ft-cap { font-size: 11px; fill: var(--text-color, #1f2328); }
  .ft-flow .ft-price { font-size: 15px; font-weight: 600; fill: var(--text-color, #1f2328); }
</style>

<figure class="ft-flow" markdown="0">
  <svg viewBox="-80 0 800 390" role="img" aria-label="Four-column flow chart: a vector icon fans out to four option-payoff icons (sold call, sold put, bought put, bought call), which lead to a butterfly-payoff icon, then an ellipsis. The bought put and bought call kinks line up with the butterfly's two outer inflection points.">
    <defs>
      <marker id="ft-arrowhead" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
        <path d="M0,0 L10,5 L0,10 z" fill="#b0b0b0"/>
      </marker>
      <marker id="ft-arrowhead-light" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
        <path d="M0,0 L10,5 L0,10 z" fill="#e6e6e6"/>
      </marker>
      <marker id="ft-arrowhead-clay" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
        <path d="M0,0 L10,5 L0,10 z" fill="#cc785c"/>
      </marker>
    </defs>

    <!-- Edges: vector boxes -> column 2 boxes (light grey, from vector boxes 1, 3, 4) -->
    <path class="ft-arrow-light" d="M67,163 C150,163 150,60 226,60" marker-end="url(#ft-arrowhead-light)"/>
    <path class="ft-arrow-light" d="M67,163 C150,163 150,150 226,150" marker-end="url(#ft-arrowhead-light)"/>
    <path class="ft-arrow-light" d="M67,163 C150,163 150,240 226,240" marker-end="url(#ft-arrowhead-light)"/>
    <path class="ft-arrow-light" d="M67,163 C150,163 150,330 226,330" marker-end="url(#ft-arrowhead-light)"/>
    <path class="ft-arrow-light" d="M67,195 C150,195 150,60 226,60" marker-end="url(#ft-arrowhead-light)"/>
    <path class="ft-arrow-light" d="M67,195 C150,195 150,150 226,150" marker-end="url(#ft-arrowhead-light)"/>
    <path class="ft-arrow-light" d="M67,195 C150,195 150,240 226,240" marker-end="url(#ft-arrowhead-light)"/>
    <path class="ft-arrow-light" d="M67,195 C150,195 150,330 226,330" marker-end="url(#ft-arrowhead-light)"/>
    <path class="ft-arrow-light" d="M67,211 C150,211 150,60 226,60" marker-end="url(#ft-arrowhead-light)"/>
    <path class="ft-arrow-light" d="M67,211 C150,211 150,150 226,150" marker-end="url(#ft-arrowhead-light)"/>
    <path class="ft-arrow-light" d="M67,211 C150,211 150,240 226,240" marker-end="url(#ft-arrowhead-light)"/>
    <path class="ft-arrow-light" d="M67,211 C150,211 150,330 226,330" marker-end="url(#ft-arrowhead-light)"/>

    <!-- Edges: col2 boxes -> col3 boxes 2 & 3 (light grey) -->
    <path class="ft-arrow-light" d="M292,60 C365,60 365,195 436,195" marker-end="url(#ft-arrowhead-light)"/>
    <path class="ft-arrow-light" d="M292,150 C365,150 365,195 436,195" marker-end="url(#ft-arrowhead-light)"/>
    <path class="ft-arrow-light" d="M292,240 C365,240 365,195 436,195" marker-end="url(#ft-arrowhead-light)"/>
    <path class="ft-arrow-light" d="M292,330 C365,330 365,195 436,195" marker-end="url(#ft-arrowhead-light)"/>
    <path class="ft-arrow-light" d="M292,60 C365,60 365,285 436,285" marker-end="url(#ft-arrowhead-light)"/>
    <path class="ft-arrow-light" d="M292,150 C365,150 365,285 436,285" marker-end="url(#ft-arrowhead-light)"/>
    <path class="ft-arrow-light" d="M292,240 C365,240 365,285 436,285" marker-end="url(#ft-arrowhead-light)"/>
    <path class="ft-arrow-light" d="M292,330 C365,330 365,285 436,285" marker-end="url(#ft-arrowhead-light)"/>

    <!-- Edges: col3 boxes 2 & 3 -> price (light grey) -->
    <path class="ft-arrow-light" d="M502,195 C565,195 565,194 632,194" marker-end="url(#ft-arrowhead-light)"/>
    <path class="ft-arrow-light" d="M502,285 C565,285 565,194 632,194" marker-end="url(#ft-arrowhead-light)"/>

    <!-- Column 1: vector icon (5 stacked boxes) -->
    <rect class="ft-vecbox" x="53" y="156" width="14" height="14" fill="#cc785c"/>
    <rect class="ft-vecbox" x="53" y="172" width="14" height="14" fill="#d4a27f"/>
    <rect class="ft-vecbox" x="53" y="188" width="14" height="14" fill="#ebdbbc"/>
    <rect class="ft-vecbox" x="53" y="204" width="14" height="14" fill="#faeee1"/>
    <rect class="ft-vecbox" x="53" y="220" width="14" height="14" fill="#d4a27f"/>

    <!-- Labels for vector boxes 2 & 5, right-justified flush to the boxes -->
    <text class="ft-cap" x="48" y="179" text-anchor="end" dominant-baseline="middle"># bedrooms</text>
    <text class="ft-cap" x="48" y="227" text-anchor="end" dominant-baseline="middle"># bathrooms</text>

    <!-- Column 2: four payoff icons (64x64 squares) -->
    <!-- Box 1: sold call (ReLU flipped vertically), 45 deg slope -->
    <text class="ft-cap" x="260" y="20" text-anchor="middle">penalizes &gt; 0.8 ratio</text>
    <rect class="ft-icon" x="228" y="28" width="64" height="64" rx="4" ry="4"/>
    <polyline class="ft-pnl" points="240,50 260,50 280,70"/>

    <!-- Box 2: sold put (ReLU flipped vertically + horizontally), 45 deg slope -->
    <text class="ft-cap" x="260" y="110" text-anchor="middle">penalizes &lt; 0.8 ratio</text>
    <rect class="ft-icon" x="228" y="118" width="64" height="64" rx="4" ry="4"/>
    <polyline class="ft-pnl" points="240,160 260,140 280,140"/>

    <!-- Box 3: bought put, kink shifted left to butterfly's left outer inflection, 45 deg slope -->
    <text class="ft-cap" x="260" y="200" text-anchor="middle">rewards &lt; 0.6 ratio</text>
    <rect class="ft-icon" x="228" y="208" width="64" height="64" rx="4" ry="4"/>
    <polyline class="ft-pnl" points="240,235 250,245 280,245"/>

    <!-- Box 4: bought call, kink shifted right to butterfly's right outer inflection, 45 deg slope -->
    <text class="ft-cap" x="260" y="290" text-anchor="middle">rewards &gt; 1.0 ratio</text>
    <rect class="ft-icon" x="228" y="298" width="64" height="64" rx="4" ry="4"/>
    <polyline class="ft-pnl" points="240,335 270,335 280,325"/>

    <!-- Column 3: three payoff icons (re-centered vertically around y=195) -->
    <!-- Box 1 (top): butterfly. Outer inflections at x=460 (rel 0.25) and x=480 (rel 0.75), peak at x=470 -->
    <text class="ft-cap" x="470" y="65" text-anchor="middle" font-weight="700">rewards 0.6 ~ 1.0 ratio</text>
    <rect class="ft-icon" x="438" y="73" width="64" height="64" rx="4" ry="4"/>
    <polyline class="ft-pnl" points="450,110 460,110 470,100 480,110 490,110"/>

    <!-- Box 2: straddle (V shape), 45 deg slopes -->
    <text class="ft-cap" x="470" y="155" text-anchor="middle">irrelevant feature</text>
    <rect class="ft-icon" x="438" y="163" width="64" height="64" rx="4" ry="4"/>
    <polyline class="ft-pnl" points="450,185 470,205 490,185"/>

    <!-- Box 3: flat line (dead neuron) -->
    <text class="ft-cap" x="470" y="245" text-anchor="middle">dead neuron</text>
    <rect class="ft-icon" x="438" y="253" width="64" height="64" rx="4" ry="4"/>
    <polyline class="ft-pnl" points="450,285 490,285"/>

    <!-- Column 4: price -->
    <text class="ft-price" x="655" y="200" text-anchor="middle">price</text>

    <!-- Edges: vector boxes 2 & 5 -> column 2 boxes (BOOK_CLOTH, drawn last to render on top) -->
    <path class="ft-arrow-clay" d="M67,179 C150,179 150,60 226,60" marker-end="url(#ft-arrowhead-clay)"/>
    <path class="ft-arrow-clay" d="M67,179 C150,179 150,150 226,150" marker-end="url(#ft-arrowhead-clay)"/>
    <path class="ft-arrow-clay" d="M67,179 C150,179 150,240 226,240" marker-end="url(#ft-arrowhead-clay)"/>
    <path class="ft-arrow-clay" d="M67,179 C150,179 150,330 226,330" marker-end="url(#ft-arrowhead-clay)"/>
    <path class="ft-arrow-clay" d="M67,227 C150,227 150,60 226,60" marker-end="url(#ft-arrowhead-clay)"/>
    <path class="ft-arrow-clay" d="M67,227 C150,227 150,150 226,150" marker-end="url(#ft-arrowhead-clay)"/>
    <path class="ft-arrow-clay" d="M67,227 C150,227 150,240 226,240" marker-end="url(#ft-arrowhead-clay)"/>
    <path class="ft-arrow-clay" d="M67,227 C150,227 150,330 226,330" marker-end="url(#ft-arrowhead-clay)"/>

    <!-- Edges: col2 boxes -> top col3 box (BOOK_CLOTH, drawn last to render on top) -->
    <path class="ft-arrow-clay" d="M292,60 C365,60 365,105 436,105" marker-end="url(#ft-arrowhead-clay)"/>
    <path class="ft-arrow-clay" d="M292,150 C365,150 365,105 436,105" marker-end="url(#ft-arrowhead-clay)"/>
    <path class="ft-arrow-clay" d="M292,240 C365,240 365,105 436,105" marker-end="url(#ft-arrowhead-clay)"/>
    <path class="ft-arrow-clay" d="M292,330 C365,330 365,105 436,105" marker-end="url(#ft-arrowhead-clay)"/>

    <!-- Edge: top col3 box -> price (BOOK_CLOTH, drawn last to render on top) -->
    <path class="ft-arrow-clay" d="M502,105 C565,105 565,194 632,194" marker-end="url(#ft-arrowhead-clay)"/>
  </svg>
</figure>

This mixture of apartment descriptors is still rather interpretable and simple, but you can imagine that in a slightly more complex situation (e.g. medical diagnoses), a mixture of any arbitrary number of medical descriptors / conditions could get complicated and hard to articulate very quickly. (The difficult to articulate is at once a linguistic problem, as well as a description-generation scaling problem.)

### Does not teach us 

Interpretability has been plagued with the problem that what constitutes an "interpretable feature" or even "feature" has been "hard to define." As a result, much research produced by even the top labs have to contend with problems stemming from the subjectivity of the definition of "interpretable" and "feature." These problems don't just stop at the conclusion that such definitions are subjective; they:
- Transmute into correctness and scaling issues in auto-interp pipelines, in that using an LLM to generate explanations for features suffers from a lack of completeness of feature descriptions, and whereby you have to sift through a large number of texts to by reasonably confident that you've seen most of the meanings that this feature could possibly encode.
- By trying to label features with human interpretable concepts, 
- as well as a lack of judgment on whether 2 concepts can be lumped into a parent concept or should remain as distinct "features."
- Transmute into scaling problems whereby you need to sample an exponentially (in terms of number of features the model has supposedly encoded) large number of texts to generate a description of a feature that captures a reasonably complete explanation for the texts that this feature has activated for.
- 