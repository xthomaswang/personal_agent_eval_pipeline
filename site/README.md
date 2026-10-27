# Benchmark website

The approved warm-cream website source, shared across four static pages. Python standard library only; no frontend build or package installation is required.

```sh
python3 -B scripts/build_docs.py
python3 -B scripts/build_docs.py --check
python3 -B -m unittest discover -s scripts/tests
python3 -m http.server 8765 --bind 127.0.0.1 --directory docs
```

- `template.html`: shared shell for `index.html`, `pipeline.html`, `methodology.html`, and `reproduce.html`.
- `app.js`: results, UI input/scoring controls, model details, harness comparison, and page navigation.
- `architecture.js`: selectable component architecture and suite-specific paths.
- `styles.css`: responsive layout and shared visual tokens.
- `assets/`: the generated chibi Akita avatar and its provenance.
- `../scripts/site_data.py`: public-evidence enrichment, validation, and dataset fingerprinting.

`docs/` contains generated pages and assets alongside the committed benchmark data. Edit this directory's source, then rebuild; do not edit generated HTML or assets. The check covers all generated pages, CSS, JavaScript, data, and the avatar. Asset URLs carry a content version so new data and styles refresh together.

Every primary table contains one machine and protocol. Available result categories, machines, protocols, dates, and counts come from the published rows. Unknown protocol metrics remain available through the protocol selector. UI modes use audited public counts, with screenshot, OCR, AX, and original-capture variants; strict and lenient read scores remain separate. The private UI task set, original photos, model responses, and local run files are not copied into this website.

Dataset fingerprints depend on public data, not the current Git commit or clock, so committing generated files does not make the next check stale.

The legacy preview can be refreshed with `python3 scripts/build_docs.py --output website-demo`. The preview folder and review screenshots are ignored by Git.
