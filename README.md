# Training Your Own Models on One 24 GB GPU

Vibe Authored by Dr.Puma

[Read the complete web edition](https://mrscripty.github.io/training-your-own-models/)

A project-first handbook that begins with a three-parameter classifier and develops practical understanding of language, vision, embeddings, decision models, speech recognition, diffusion, and sound generation.

## Inside

- The complete 40-chapter manuscript, with its introduction, 243-term glossary, works cited, and stable cross-reference links
- Three browser-only learning tools: gradient steps, persistent training-state memory arithmetic, and a causal attention mask
- Full-text section search, chapter navigation, responsive reading layouts, and an accessible no-JavaScript reading experience
- Unchanged PDF (247 pages including covers), Markdown, and 57-script companion downloads with SHA-256 checksums
- A 1280 × 640 social preview derived from the book cover, used in repository Settings and static Open Graph/Twitter metadata

## Evidence boundary

No GPU training or 24 GB peak-memory measurements were performed for this edition. The book separates documented capabilities, worked calculations, proposed starting configurations, and checks actually performed. The browser tools are educational calculations, not GPU training or benchmarks. A memory subtotal does not establish that a real run fits.

The companion README and chapter 40 explain execution and verification. Complete companion files, rather than snippets copied from a page, are the authoritative executable examples.

## Build

Requires Python 3, Pandoc, and BeautifulSoup4. The published website has no runtime dependencies, accounts, analytics, or external JavaScript.

```sh
python3 -m pip install -r requirements.txt
python3 build.py
python3 tests/validate.py
node --test tests/math.test.js
python3 -m http.server 8096 --directory docs
```

The checked-in source manuscript is `source/manuscript.md`. `build.py` renders every manuscript section, preserves explicit anchors, remaps cross-chapter links, builds search data, and emits metadata. Static CSS/JavaScript and original download files live in `docs/`. GitHub Pages serves `main` → `/docs`.

The original download bytes are intentionally preserved. Their hashes are published at `docs/downloads/checksums.json` and on the download page. Book figures and cover artwork are included under `docs/assets/`.

## Reuse

This repository does not grant a new blanket license for the book or third-party material. References retain their original attribution, licenses, and limitations. Consult the relevant sources before reusing external code or artwork.
