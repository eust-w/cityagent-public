# Third-party software notices

BridgeEvidence combines the following open-source projects. This file is an
attribution inventory, not a replacement for each project's license text.

| Project | Role | License | Source |
|---|---|---|---|
| LangGraph | ReAct state graph and conditional loop routing | MIT | https://github.com/langchain-ai/langgraph |
| Pydantic | Typed domain state and command validation | MIT | https://github.com/pydantic/pydantic |
| Instructor | Structured DeepSeek planner output | MIT | https://github.com/567-labs/instructor |
| OpenAI Python | OpenAI-compatible gateway client | Apache-2.0 | https://github.com/openai/openai-python |
| python-docx | DOCX parsing and report generation | MIT | https://github.com/python-openxml/python-docx |
| Pillow | Bounded DOCX image inspection and OCR candidate selection | HPND | https://github.com/python-pillow/Pillow |
| olefile | OLE2 and legacy Word preflight inspection | BSD-2-Clause | https://github.com/decalage2/olefile |
| pypdfium2 | PDF page rendering for conversion visual-difference audits | BSD-3-Clause / Apache-2.0 | https://github.com/pypdfium2-team/pypdfium2 |
| lxml | OOXML access used by python-docx | BSD-3-Clause | https://github.com/lxml/lxml |
| RapidFuzz | Text normalization and fuzzy similarity | MIT | https://github.com/rapidfuzz/RapidFuzz |
| SciPy | One-to-one optimal assignment | BSD-3-Clause | https://github.com/scipy/scipy |
| NetworkX | Disease graph modelling and diagnostics | BSD-3-Clause | https://github.com/networkx/networkx |
| structlog | Structured local event logs | Apache-2.0 / MIT | https://github.com/hynek/structlog |
| pytest | Test runner (development only) | MIT | https://github.com/pytest-dev/pytest |
| Hypothesis | Property testing (development only) | MPL-2.0 | https://github.com/HypothesisWorks/hypothesis |
| uv | Dependency resolution and lock file tooling | Apache-2.0 / MIT | https://github.com/astral-sh/uv |
| LibreOffice | Legacy `.doc` to `.docx` conversion; external executable | MPL-2.0 | https://github.com/LibreOffice/core |
| wvWare | Large legacy `.doc` semantic extraction fallback; external executable | GPL-2.0-or-later | https://sourceforge.net/projects/wvware/ |
| Pandoc | HTML-to-DOCX bridge for the wvWare fallback; external executable | GPL-2.0-or-later | https://github.com/jgm/pandoc |

No third-party model weights or source repositories are vendored in this
project. LibreOffice is invoked from the host environment and is not bundled.

TongAgents is an optional external SDK. Its code, wheel and course materials are
not bundled. Obtain it through its authorized distribution and review its terms
separately. Mock tests do not certify any SDK, model or hardware integration.

Additional runtime and developer dependencies:

| Project | Role | License | Source |
|---|---|---|---|
| jsonschema | Tool argument validation | MIT | https://github.com/python-jsonschema/jsonschema |
| HTTPX | HTTP transport | BSD-3-Clause | https://github.com/encode/httpx |
| Tokenizers | Optional CPU tokenizer | Apache-2.0 | https://github.com/huggingface/tokenizers |
| Jinja2 | Optional chat-template rendering | BSD-3-Clause | https://github.com/pallets/jinja |

The standard-library SQLite cache stores JSON only; this edition does not use
DiskCache or read its pickle-based cache files. Model weights and adapters retain
their own licenses and must be supplied separately. Optional training, hardware,
OCR and benchmark tools can require additional dependencies; installing the base
package does not install vendor inference engines or training frameworks.

`uv.lock` records dependency resolutions, including optional extras. It is not
proof that every optional framework was executed on every supported platform.
Use `uv sync --locked --extra dev` for the tested development dependency set.

## Presentation material

The selected defense slide renderings under `docs/assets/defense/` are project
presentation material, not Apache-2.0 source code. Conference logos, visual
backgrounds and other third-party marks retain their respective rights. See the
[asset notice](docs/assets/defense/README.md) before reusing those images. Their
presence does not imply endorsement by the event organizers.
