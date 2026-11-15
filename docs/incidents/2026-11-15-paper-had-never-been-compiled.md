# 2026-11-15: The paper had never been compiled

Cause: the development machine had no LaTeX and the weekly CI job that compiles the paper had not run yet, so `paper/main.tex` had only ever been read. The first compilation stopped on two errors: `\Hz_i` expands to `H_0_i` (double subscript), and the link colour `blue!50!black` needs the package xcolor. Three generated tables were also 70 to 85 pt wider than the page.

Fix: `H_{0,i}` written out, xcolor loaded; every generated table sits in an `adjustbox` that shrinks it only when it is wider than the text; slide figures fit both the height and the width of their column. Paper and slides now compile without an overfull box, locally with Tectonic and in the weekly job.

Test added: none in the test suite; the weekly CI job compiles both documents.
