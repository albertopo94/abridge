# Junie Upstream

Add JetBrains Junie CLI as a fourth upstream provider in Abridge, exposing its models as `junie/*` through the existing gateway protocols, dashboard, and setup CLI.

## Specs

- **S1 (user intent)**: "quiero ahora quiero que agreguemos Junie CLI la documentacion es https://junie.jetbrains.com/docs/junie-cli.html te adelanto que ya lo tengo instalado y arranca al usar 'junie'"
- **S2 (routing decision)**: "Prefijo junie/*" — Junie models are served as `junie/<id>` (e.g. `junie/grok-4.5`), consistent with `cursor/*` and `opencode/*`. The client chooses when to use them.
- **S3 (scope decision)**: "Completo (upstream + dashboard + setup)" — junie_acp + junie_backend + server routing + /v1/models catalog + dashboard tab + `--provider junie` flag in setup CLI.
- **S4 (probe evidence)**: `junie --acp true` answers `initialize` over stdio JSON-RPC 2.0 with no `authenticate` step (matching OpenCode's handshake), `promptCapabilities.image: true` (multimodal supported), and `sessionCapabilities` including `fork/list/resume/delete/close` (standard ACP session model).
- **S5 (docs reference)**: Official doc https://junie.jetbrains.com/docs/junie-cli.html — ACP mode via `junie --acp true`, catalog via `junie list-models` (JSON `{"models":[{id, displayName, efforts, defaultEffort, default}]}`), headless via `junie --task "..."`.

## Tasks

- T1: `bridge/junie_acp.py` — persistent ACP client (`junie --acp true`), stdio JSON-RPC 2.0, warm process, handshake initialize → session/new, per-prompt model selection and streaming, mirroring `opencode_acp.py` (S2, S4, S5). Tests in `tests/test_junie_acp.py`.
- T2: `bridge/junie_backend.py` — model discovery via `junie list-models` (JSON) with static fallback, `is_junie_model`/`resolve_junie_model`, ACP execution with `junie --task` fallback, `[Junie]` display-name tags, `list_models()` for the catalog (S2, S5). Tests in `tests/test_junie_backend.py`.
- T3: `bridge/server.py` — route `junie/*` in `/v1/chat/completions` and `/v1/messages`, extend `/v1/models` catalog with junie backend, warmup on start (S2, S3). Tests in `tests/test_server.py`.
- T4: `bridge/dashboard.py` — Junie availability/catalog tab (no quota API exposed), consistent with the OpenCode tab (S3). Tests in `tests/test_dashboard.py`.
- T5: `bridge/setup/*` + `bridge/cli/setup.py` — add `junie` provider value to the `--provider` selection across client configurators (`agy|cursor|opencode|all` → `+junie`) (S3). Tests in `tests/test_setup.py`.
- T6: Documentation (README feature/notes), test count synchronization, SemVer MINOR bump to 0.37.0 synchronized in `bridge/__init__.py`, `pyproject.toml`, `install.sh`, and version-pinned tests (S1-S5).

## Log

- **L1** (2026-11-12, user): "quiero ahora quiero que agreguemos Junie CLI la documentacion es https://junie.jetbrains.com/docs/junie-cli.html te adelanto que ya lo tengo instalado y arranca al usar 'junie'. Sabes todo lo que implica este cambio o lo que implica agregar esta nueva fuente? Cualquier duda que tengas preguntame."
- **L2** (2026-11-12, user decision): routing = "Prefijo junie/*" (value: junie-prefix).
- **L3** (2026-11-12, user decision): scope = "Completo (upstream + dashboard + setup)".
- **L4** (2026-11-12, probe): `junie --acp true` initialize handshake confirmed — no authenticate step, image:true, standard ACP session capabilities.
- **L5** (2026-11-12, implemented): T1-T6 closed. All 1219 tests pass (exit 0). Version bumped 0.36.3 → 0.37.0. Feature branch `feature/junie-upstream`.
- **L6** (2026-11-12, commit): `d94560b` feat(junie): add JetBrains Junie CLI as fourth upstream — closed T1-T6. Slice 1928 lines (large cohesive feature, 1219 tests green). `junie-logo.svg` is a pre-existing untracked user asset, not part of this feature.
- **L7** (2026-11-12, delivery): Added Junie client card with its logo — commit `041c862` (assets/junie-logo.svg moved from root, JUNIE_ICON_SVG + guide-only card, card_ids now 10). Merged feature branch to main, pushed main, tagged v0.37.0 (both pushed). `abridge update` now delivers Junie.
