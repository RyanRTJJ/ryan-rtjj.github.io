---
published: false
title: Widget Demo — Hopping Agent
date: 2026-06-10 00:01:00 -0800
categories: [meta]
tags: [meta]
---

A second proof-of-concept widget (see `WIDGETS.md`), this time an *animation* rather
than an interaction. A duck 🦆 hops square-to-square across a 5×5 grid built in the same
style as the [tic-tac-toe demo]({% post_url 2026-06-10-widget-demo %}). Everything below
is plain HTML/CSS/JS — no external libraries.

{% include widgets/agent-hop.html %}

The agent starts at `grid[2][0]`, hops rightward along the middle row to `grid[2][4]`,
then upward along the right column to `grid[0][4]`, and loops. Each move is a single
**parabolic hop**: the duck traces an arc that peaks at a fixed vertical apex above the
straight line to the target square, with horizontal displacement equal to the column
change (zero for the upward leg). A ground shadow shrinks and fades as the duck rises to
sell the height. Use **Pause/Play** to freeze it and **Restart** to send it back to the
start.
