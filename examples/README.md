# Offline example

From the repository root, after `python -m pip install -e .`:

```bash
python examples/offline_demo.py --output-dir /tmp/cityagent-demo
```

Use a new output directory on each run. The script generates a completely
fictional bridge report, invokes the real `bridge-evidence single` command in
offline mode, and writes a Word report plus JSON evidence. No API key, model,
competition SDK, real inspection report, or LibreOffice is needed for this DOCX
example. It does not evaluate model accuracy or competition performance.

The CLI can also process a report you are authorized to use:

```bash
bridge-evidence single --input /path/to/report.docx \
  --output-dir /path/to/results --json-dir /path/to/evidence --llm-mode off
```

Offline extraction uses deterministic rules and rejects reports that fail the
required-field checks; it is not a substitute for complete model inference.
Legacy binary DOC input needs a separately installed LibreOffice converter.
Rendering Chinese text in a document viewer also needs a suitable Chinese font.
