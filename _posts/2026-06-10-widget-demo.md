---
published: true
title: Widget Demo — Tic-Tac-Toe
date: 2026-06-10 00:00:00 -0800
categories: [meta]
tags: [meta]
---

A proof-of-concept showing an interactive widget embedded straight into a post via the
reusable-include pattern (see `WIDGETS.md`). Everything below is plain HTML/JS — no
external libraries.

{% include widgets/tic-tac-toe.html %}

That's it. The board is fully playable: click a square to place a mark, it alternates
X/O, detects wins and draws, highlights the winning line, and the Reset button starts
over.
