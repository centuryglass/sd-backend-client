# AGENTS.md

Rules for coding agents working in the sd-backend-client codebase. Human-facing setup and usage docs are in
[`README.md`](README.md) and [`tests/README.md`](tests/README.md). Open work lives in
[GitHub issues](https://github.com/centuryglass/sd-backend-client/issues); see "Tracking open work".

## How to use this file

- **This file is for facts that cross files, and most changes add nothing to it.** A bullet here earns its place by
  biting someone who is editing a *different* file than the one the fact lives in. Before adding one, ask:
  - Is it relevant only within one file?
  - Would opening that file to make the edit surface it anyway?
  - Would a reader be better served finding it there?

  A yes to any of these means the fact goes in that file's own comment (confirm it is there, or add it). Here it gets
  at most a one-line pointer.
- **Edit by replacing, not appending.** When a change makes a bullet wrong, rewrite that bullet; don't add a second one
  that corrects the first. When the code a bullet describes is gone, delete the bullet.
- **Code comments cite headings and bold lead phrases by file and name** (`AGENTS.md, "Tracking open work"`). Renaming
  one, or moving it to another file, breaks those pointers: grep the repo for the old phrase and fix every hit in the
  same change.
- **Area hazards move to `doc/agents/` once they outgrow this file.** When one area (the ComfyUI node graph, a
  backend client, ControlNet serialization) collects hazards that only matter to edits in that area, move them to
  `doc/agents/<area>.md` and route to it from a "Things that will bite you" table here, listing the paths each file
  covers.
- **`CLAUDE.md` imports this file** with `@AGENTS.md`, so agents that read either name get the same rules. Edit
  `AGENTS.md`.

## What this is

`sd_backend_client` is a standalone, headless Stable Diffusion backend client extracted from
[IntraPaint](https://github.com/centuryglass/IntraPaint)'s `src/api`. It talks to **ComfyUI**, **Forge/Automatic1111
WebUI**, and their **ControlNet** extensions, building the request bodies and node graphs each backend expects.
Everything lives under the `sd_backend_client` package and imports only from within it plus third-party libraries.
Python **3.11+** (see "Conventions").

Status: under active refactoring. Core txt2img / img2img / inpainting / upscaling (including tiled Ultimate SD
Upscale) / ControlNet paths work end-to-end on both backends.

## Who it serves

In priority order:

1. **Library users.** People who install `sd_backend_client` into their own projects. They see the package only through
   its public API, the README, the docs site and its errors. Keep the usage examples in the README and `docs/` working,
   mark any change that breaks the public API (see "Working with GitHub"), and make errors say what went wrong and what
   to do.
2. **The maintainer.** The package is the maintainer's backend client for IntraPaint and other projects. A change that
   makes it harder to drive either backend from those projects needs a strong reason.
3. **The portfolio.** The repo shows a library extracted from a larger hand-built project and brought up to
   professional standards with coding agents. The reader is a reviewer or hiring manager skimming the repo, looking
   for CI that gates merges, tests that would catch a real regression, honest docs and a clean issue history.

- When a change trades one audience against another, flag the tension to the maintainer. Don't quietly resolve it
  in either direction.
- A process or documentation gap that matters only for portfolio value gets its own GitHub issue, not bundled
  invisibly into unrelated work.

## Running

```bash
pip install -e .                    # the package and its runtime dependencies
pip install -r requirements-dev.txt # for development: adds pytest and pylint
pip install -r requirements-docs.txt # for the documentation site: `mkdocs serve` previews it

# Offline pure-logic unit tests: no server, no GPU, deterministic (use these by default):
pytest tests/unit/
pytest tests/unit/test_comfy_workflow_builder.py::TEST_NAME   # single test

# Live integration tests against whatever backend(s) are up; the rest skip themselves:
pytest tests/                     # fast read-only metadata/auth tests + unit tests
pytest tests/ --run-generation    # add slow real-diffusion round-trips (needs a loaded checkpoint)
RUN_SD_GENERATION=1 pytest tests/ # equivalent env-var form
```

- Point the integration tests at servers via env vars (defaults in parentheses): `SD_API_URL`
  (`http://127.0.0.1:7860`), `COMFYUI_API_URL` (`http://127.0.0.1:8188`), `SD_UNAME`/`SD_PASS` for A1111
  `--api-auth`. `tests/local.env` (gitignored) is a convenience `source`-able file for credentials. `tests/README.md`
  has the full matrix.
- Runtime dependencies are listed with lower bounds only, in both `pyproject.toml` and `requirements.txt`; keep the
  two lists in sync. CI's `test-minimum-deps` job runs the tests with every floor installed, so raise a floor when new
  code needs a newer release. Dev tools are pinned exactly in `requirements-dev.txt`, and Dependabot proposes their bumps
  (`.github/dependabot.yml`).

## Architecture

The package is a **functional core / imperative shell**: pure builders turn pydantic params into wire payloads, and
thin HTTP clients send them. The two backends' APIs are shaped very differently, which is the main thing to
understand.

**The public API is the package root's `__all__`** (`sd_backend_client/__init__.py`); every other module is internal.
A new class that `Backend`, `GenerationHandle` or an exported model uses gets added there, or
`tests/unit/test_public_api.py` fails. Each name in `__all__` also needs a directive on a `docs/reference/` page, or
the docs build fails (see "Documentation site").

### Shared, backend-agnostic layer: `api/shared_data/`

- `diffusion_params.py`: `DiffusionParams`, the pydantic base holding parameters common to both backends. Each backend
  **subclasses** it: `DiffusionRequestBody` (WebUI) and `ComfyUIDiffusionParams` add backend-specific fields.
- `api_datatypes.py`: `DiffusionUpscalingParams` and related shared models.
- `backend.py`: `Backend`, the interface both clients implement: `submit_*` for jobs, and `list_*` and
  `get_capabilities` for discovery (types in `backend_options.py`). Its contract is pinned by
  `tests/unit/test_backend_contract.py` and `tests/unit/test_discovery.py`, which drive both clients through it.
  `api/detect.py`'s `connect_to_backend` picks the client for a URL.
- `generation_handle.py`: the backend-agnostic async generation handle both clients implement.
- `controlnet/`: `ControlNetUnit` + `ControlNetModel` + `ControlNetPreprocessor`, all backend-agnostic. Units are
  attached to `diffusion_params.controlnet_units`; each backend serializes them into its own form. Serialization
  round-trips are pinned by `tests/unit/test_controlnet_serialization.py`.

### `api/webservice.py`: `WebService` base

Session/auth/GET/POST helper both clients extend. On a 401 it calls `_handle_auth_error()` (subclass-overridden), then
retries the request. `A1111Webservice` implements it via a `credentials_provider` callback; ComfyUI has no auth.

### WebUI client: `api/a1111_webservice.py` + `api/webui/` (synchronous)

`A1111Webservice.txt2img(body)` / `img2img(image, mask, body)` block until the image is ready and return
`{'images': list[PIL.Image], 'info': ...}`. `api/webui/` holds the request/response pydantic formats.
`DiffusionRequestBody.to_dict()` (`webui/diffusion_request_body.py`) strips `None`s and is where ControlNet units get
injected into `alwayson_scripts`.

### ComfyUI client: `api/comfyui_webservice.py` + `api/comfyui/` (asynchronous, node graphs)

ComfyUI is a queue: `ComfyUiWebservice.txt2img(params)` returns immediately with a `prompt_id`; callers poll
`check_queue_entry(prompt_id)` until FINISHED, then call `download_images(...)`. `submit_*` wraps this in a
`ComfyGenerationHandle`, which reads step progress and previews from the server websocket through the service's one
`ComfyProgressListener` (`comfyui/comfyui_progress_listener.py`).

The request body is a **node graph**, not JSON fields. This is the most involved part of the repo:

- `comfyui/diffusion_workflow_builder.py`: `DiffusionWorkflowBuilder`. Configure it by setting attributes, then
  `build_workflow()` returns a `ComfyNodeGraph`. It decides checkpoint/KSampler wiring, prompt encoding,
  txt2img-vs-img2img latent source, CLIP-skip insertion, batching, etc. Its graph structure is pinned by
  `tests/unit/test_comfy_workflow_builder.py`.
- `comfyui/nodes/`: one pydantic model per ComfyUI node type (`ComfyNode` subclasses), grouped into `input/`, `vae/`,
  `model_extensions/`, `controlnet/`. Fields are the node's inputs under their ComfyUI names, and `Output` attributes
  such as `KSamplerNode.latent_out` name its output slots. A connection input holds the upstream node's `NodeOutput`,
  so wiring a node is assigning to its field. `nodes/comfy_node_graph.py` (`ComfyNodeGraph`) collects every node
  reachable from the ones added, assigns integer string keys starting at 3, and emits the final
  `{key: {class_type, inputs}}` workflow dict with connections as `(node_key, output_slot)` tuples. Builders that keep
  nodes between builds must clear those nodes' connections after each build.
- Separate builders exist for upscaling and preprocessor-preview workflows (`basic_upscale_workflow_builder.py`,
  `latent_upscale_workflow_builder.py`, `preprocessor_preview_workflow_builder.py`).
  `LatentUpscaleWorkflowBuilder.build_workflow`'s docstring notes a limit on the Ultimate SD Upscale path: a target
  aspect ratio different from the source's isn't honored, since both sides scale by the larger width/height ratio.

### `util/`

`geometry.py` (`Size`), `api_defaults.py` (defaults shared by the parameter models), and `visual/image_utils.py`
(base64/PNG to and from `PIL.Image`, RGBA normalization).

## Conventions

- **No Qt / PySide6, no cv2 / numpy.** Images are plain `PIL.Image` (normalized to RGBA); sizes use the minimal `Size`
  class in `util/geometry.py`. The package carries no UI strings or translation helpers.
- **No config layer.** Generation parameters are passed explicitly as pydantic models, never read from a
  `Cache`/`AppConfig` singleton. A new parameter is threaded through the pydantic model.
- **Code runs on Python 3.11,** the `requires-python` minimum CI tests. Two 3.12-only forms break it:
  - the `type X = ...` statement. Write `X: TypeAlias = ...`.
  - `typing.TypedDict` in anything pydantic validates. Import `TypedDict` from `typing_extensions`.
- **Errors use the tree in `sd_backend_client/errors.py`.** A failure talking to a backend raises an `SDBackendError`
  subclass, invalid caller input raises `ValueError`, and `assert` only guards internal invariants, since `python -O`
  strips it.
- **Ask the maintainer before adding a dependency,** and keep them minimal. Every runtime dependency lands in each
  library user's environment.
- Match the surrounding code's style, naming, and structure.
- **A bug found during unrelated work gets fixed or filed, never just noticed.**
  - Trivial to fix (a wrong assertion, an off-by-one, a stale comment or pointer): fix it in the same pass.
  - Needs real investigation or design, or touches code you weren't already changing: open a GitHub issue (see
    "Tracking open work") with what was observed, how to reproduce it, and what is ruled out.
  - "Trivial" is about the fix, not the effort spent finding it. A fix that needs a manual check against a live
    backend, a new test harness, or more than one full test run to confirm belongs in an issue, unless the maintainer
    asked for that investigation.

## Comments and docs

**Comments are reference, not advocacy.** A comment tells the next reader what is true of the code as it stands,
quickly. It does not defend a design to a skeptic or argue against the version it replaced. These rules apply to
comments and docstrings you write or touch, to this file, and to new markdown docs. Existing human-written comments
don't need rewriting to conform, and nothing enforces these rules mechanically.

- **Lead with the rule.** The first line of a comment or docstring is a standalone summary; a reader who stops there
  must lose no invariant.
- **One fact, one home.** State a fact fully where the thing is defined. Elsewhere, point or stay silent. A pointer
  names a symbol or a section title, never a position ("see `DiffusionWorkflowBuilder.build_workflow`", not "see the
  comment above"), and it must resolve: check every "see X" before committing.
- **Pin to a declaration, not a region.** One comment describes one thing below it. Split a paragraph that describes
  several things and re-attach each piece, so each moves with its code.
- **Keep hazards, drop ghosts.**
  - A hazard warns that a change here breaks something there. Keep it, as the main clause.
  - A ghost is prose about a design the code doesn't have: an argument against an alternative ("rather than X") or a
    note about a prior state ("X used to live in Y"). Do that reasoning in your head, not the file.
  - Keep a history note only where a reader would otherwise trip: a redirect, a permanent alias, a link that still
    uses an old name. Where this package diverges from the IntraPaint code it was extracted from, saying so counts.
  - Before finishing, sweep the lines you touched for "used to", "instead of", "rather than", "no longer",
    "anymore", "previously", "now". Most hits are ghosts.
- **Length tracks risk.** A few lines is the default. More is earned only where deleting a clause would let a careful
  reader introduce a real bug. Never delete a hazard to look terse; condense or relocate it.
- **No color.** Leave out measurements, incident narrative and closed issue numbers unless the reader needs them to
  act. Cite an issue only when it is open and the reader should follow it.
- **Plain declaratives.** No shouting caps or conviction words ("exactly", "really", "deliberately", "on purpose").
  One clause per sentence, real lists for list-shaped content, and symbols rather than their current values.
- **Module docstrings say what the file is for** and which decision it embodies. Match the surrounding comment
  density, and don't narrate what the code does line by line.
- **ASCII hyphens, not em dashes,** in comments and markdown.

## Testing

- **`tests/unit/` is the safety net CI runs.** It is offline and deterministic and covers the request-shaping logic.
  A change that silently alters an emitted request must fail there, not only in a live generation, so a builder or
  serialization change comes with a unit test.
- **Integration tests live at the top of `tests/`** and talk to real backends. `tests/conftest.py`'s fixtures skip
  them when the backend is unreachable, auth or ControlNet is absent, or the `--run-generation` flag is off. CI has
  no backend, so it runs none of them: changes to the HTTP clients need a local run against a live server.
- The ControlNet generation integration tests contain a **silent-failure detector**: they generate the same
  seed+prompt with and without a control unit and fail unless the outputs differ. Keep that invariant in mind when
  touching ControlNet serialization.
- Generation tests save PNGs to `SD_TEST_OUTPUT_DIR` (default `tests/output/`, gitignored), where `tests/review.html`
  shows them. A test that saves a new or renamed image needs an entry in that page's `GROUPS`. Write any other test
  output to a temporary directory, never the working tree.
- **Prevent flaky tests instead of quarantining them.**
  - Unit tests don't touch the network, a backend, or wall-clock time.
  - No retry markers. Beyond the conftest skips above, the only allowed markers are `skip`, with a reason and an
    issue link, and `xfail(strict=True)`.
- CI (`.github/workflows/ci.yml`) runs on every pull request into `main` and every push to it: `pytest tests/` on
  Python 3.11-3.14 and once more on 3.11 with the oldest allowed runtime dependencies, plus the lint and type checks
  below. Its `ci` job is the single check to require for merging.
- **The test job fails when line-and-branch coverage drops below its `--cov-fail-under` floor** in `ci.yml`. New code
  comes with unit tests that keep it above the floor; raise the floor when coverage rises, and don't lower it to land
  a change.

## Type checking

`mypy` (no arguments, configured in `pyproject.toml`'s `[tool.mypy]`) is CI's type-check gate. It runs with mypy's
default, non-strict settings, and the package must pass it with zero errors.

- **Strictness is opt-in per module,** through `[[tool.mypy.overrides]]` sections. Don't turn on a strict flag
  globally: improving typing is welcome but not a priority, and work shouldn't block on it.

- **Loose data gets an honest type.** Where data is loose by design (parsed JSON, backend API responses, ComfyUI node
  inputs), type it as loosely as it is (`Any`, `object`, a partial `TypedDict`) until there is a real type to write.
  A strict type that fights the code's actual tolerance, a `cast` that asserts something unchecked, or a
  `# type: ignore` that hides a real mismatch is worse than a loose type.

## Documentation site

`mkdocs.yml` builds the site from `docs/`, and `.github/workflows/docs.yml` publishes it to GitHub Pages from `main`.
CI's `docs` job runs `mkdocs build --strict`, which fails on any warning.

- **Docstrings of public names are the API reference.** mkdocstrings renders them with `docstring_style: numpy`, so
  a `Parameters` / `Raises` section that breaks numpy layout fails the strict build or renders wrong.
- **The reference covers the root `__all__` and nothing else.** `scripts/mkdocs_hooks.py` fails the build when a
  `docs/reference/` page is missing a public name or documents one that isn't public.
- **Pydantic field limits render from `Field(...)`.** `docs/templates/python/material/attribute.html.jinja` shows
  `ge`, `gt`, `le`, `lt` and length limits above each field's docstring.

## Lint

`scripts/pylint.sh` shows the full report (uses `.pylintrc`). Keep new code clean against it.

`python scripts/pylint_check.py` is CI's lint gate. The code isn't pylint-clean yet, so it fails only when a file
gains messages beyond `scripts/pylint_baseline.json`, or when fixed messages leave the baseline too high. After fixing
messages, rerun it with `--update-baseline` and commit the baseline. Run it with Python 3.13: the baseline is only
valid for the version it was generated with.

## Git & releases

- **Hard rule:** `main` is the development and release branch. Changes only reach it via PR. Branch from `main` and
  target PRs at it.
- **release-please owns the version and the changelog.** `.github/workflows/release-please.yml` describes the release
  flow. Don't edit `pyproject.toml`'s `version`, `.release-please-manifest.json` or existing `CHANGELOG.md` entries
  by hand outside its release PR.

## Working with GitHub

- **Answer a question before changing anything.** A message asking whether, which or how gets its answer and then a
  stop. Reading code to form the answer is fine; edits, commits and PRs wait for a go-ahead. A message that both asks
  and directs gets the answer first, and the work proceeds only if the answer leaves the plan unchanged.
- **Open a PR against `main` once work is complete and checked, without waiting to be asked.** This overrides a coding
  agent's default of only opening a PR on explicit request. Run `pytest tests/`, `scripts/pylint_check.py` and `mypy`
  first. Agents may also commit and push to their working branch and create issues without asking.
- **PR titles use [Conventional Commits](https://www.conventionalcommits.org/) format:** `type: summary`, with a type
  such as `feat`, `fix`, `docs`, `refactor`, `perf`, `test`, `build`, `ci` or `chore`, and an optional scope
  (`fix(comfyui): ...`). Mark a change that breaks the public API with `!` (`feat!: ...`). PRs into `main` are
  squash-merged, so the title becomes the commit release-please turns into a changelog entry and version bump (`feat`,
  `fix` and `perf` appear in the changelog). Describe the change from a library user's or contributor's point of view.
- **A PR closing an issue says `Closes #NN` in its description.**
- **Don't ask whether to subscribe to a PR you just opened.** If the maintainer wants it watched, they'll say so.
- **An issue or comment an AI agent writes under the maintainer's account ends with a footer marking it as
  AI-generated**, e.g. `_Drafted with AI assistance._`, so it doesn't read as the maintainer arguing with themselves.
  Keep it tool-agnostic ("AI assistance", never a product name), since the maintainer uses more than one agent.
- **An issue links a repo document by permalink, not by branch path.** Use a blob url pinned to a commit sha, with the
  section's heading anchor (`.../blob/<sha>/tests/README.md#...`), so the link still shows what the issue was written
  against after the doc is edited, renamed or deleted. Code references by symbol name stay as they are.

## Tracking open work

- **Open work lives only in [GitHub issues](https://github.com/centuryglass/sd-backend-client/issues).** Nothing in the
  repo tracks tasks.
- **A found bug that isn't a same-pass fix opens an issue**: what was observed, how to reproduce it, and what is ruled
  out.
- **A fact worth knowing is not a task.** It belongs in the owning module's comment, or here.
- **Open issues are usually already in context.** The `SessionStart` hook (`.claude/hooks/session-start.sh`) runs
  `scripts/issues.py`, which writes `.claude/cache/issues/` (`index.md` plus one file per issue) and prints the index.
  The cache is generated and gitignored; never edit it or treat it as the source of truth. `scripts/issues.py`'s
  docstring covers the fetch paths and `SD_API_ISSUES_TOKEN`.
- **When the hook produced nothing** (rate-limited, no token, or an agent that doesn't run Claude Code hooks), build
  the cache by hand before concluding no issue covers something: fetch the issue list with whatever tool you have
  (the GitHub MCP `list_issues`, `gh issue list --json ...`), save it as JSON, run
  `python3 scripts/issues.py --from-json <path>`, then read `.claude/cache/issues/index.md`.

## Cloud sessions

`.claude/hooks/session-start.sh` refreshes the issue cache in every session (see "Tracking open work"). At the start
of a Claude Code on the web session it also creates `.venv` with Python 3.13, installs `requirements-dev.txt` and the
package in editable mode, and puts `.venv/bin` first on `PATH`, so `pytest`, `scripts/pylint_check.py` and `mypy` work
without further setup.
