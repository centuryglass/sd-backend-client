# ControlNet reference data

A sourced reference for SD1.5 and SDXL ControlNet control types, preprocessors and models. It feeds UI text (labels,
tooltips), artist guides and explanations of how each tool works. This directory sits outside the package. Anything
the package ships, such as #146's one-sentence descriptions, is derived from the `summary` of `verified` records here.

Open work is tracked in [#155](https://github.com/centuryglass/sd-backend-client/issues/155).

## Layout

```
data/controlnet/
  README.md              # this file
  sources.toml           # every cited source, by id
  control_types/<id>.toml
  preprocessors/<id>.toml
  models/<id>.toml
```

- **One TOML file per record.** A file's name is its record's `id`.
- **Ids are lowercase snake_case, stable and backend-neutral** (`canny`, `depth_anything_v2`,
  `xinsir_union_promax_sdxl`). Records reference each other by id only.
- **Backend names live in each record's `names` table.** `webui` holds sd-webui-controlnet (A1111) names. `forge`
  holds names from Forge, reForge and Forge Neo, which share one ControlNet extension. `comfyui` holds node class
  names.

## Schema

The pydantic models in [`scripts/controlnet_data.py`](../../scripts/controlnet_data.py) are the schema, with each
field documented there. `tests/unit/test_controlnet_data.py` runs it in CI. Run
`python scripts/controlnet_data.py` to load everything and print a status count.

Each model forbids unknown keys, so a misspelled field fails to load.

- **Every record has `label`, `summary`, `description` and `how_it_works`** (see "Text tiers"), plus `sources`,
  `research_status` (`stub`, `drafted` or `verified`) and `open_questions`.
- **A control type** adds `base_models`, `control_image` (what its models expect as input), `uses`, `names` and
  `choosing`, which holds a guide and ordered preprocessor and model recommendations.
- **A preprocessor** adds `control_types`, `models`, `effect`, `output`, `best_input`, `weak_input`, `names`,
  `name_parameters`, `hardware` and `parameters`.
- **A model** is one trained network. It adds `base_models`, `adapter`, `control_types`, `preprocessors`, `trainer`,
  `license`, `homepage`, `strengths`, `weaknesses`, `good_for` and `files`. Each pruned, fp16, renamed or
  LoRA-extracted copy is a `files` entry, with `variant` saying what sets it apart.
- **`name_parameters` tells apart preprocessors that share a backend name.** ComfyUI folds several WebUI
  preprocessors into one node with a switch. For example, `LineArtPreprocessor` with `coarse = "enable"` is
  `lineart_coarse`. Every record sharing a name sets it, except the one matching the node's defaults.
- **TOML puts every key after a `[table]` header inside that table.** Write a record's top-level keys, `sources`
  through `open_questions` included, before its first table.

### Model hashes

- **`sha256` is the hash of the whole file.** Hugging Face lists it per file.
- **`webui_hash` is the short hash in WebUI model names** (`control_v11p_sd15_canny [d14c016b]`). sd-webui-controlnet,
  Forge and reForge show it. It is the first 8 hex digits of the sha256 of the 64 KiB starting 1 MiB into the file
  (`model_hash` in stable-diffusion-webui's `modules/sd_models.py`). It never matches a prefix of `sha256`.
- **Forge Neo and ComfyUI list ControlNet models by file name only.**
- **A range request fetches `webui_hash` without downloading the file:**

  ```bash
  curl -sL -r 1048576-1114111 "$URL" | sha256sum | cut -c1-8
  ```

  Every `webui_hash` here was computed this way from the file's pinned `source_url`. Each one shown in the recorded
  fixtures matches.

## Text tiers

| Field | Audience | Form | Limit |
|---|---|---|---|
| `label` | UI | Title case, no trailing period | 32 chars |
| `summary` | tooltip | One plain sentence | 160 chars |
| `description` | artists | One or two paragraphs, plain text | ~120 words |
| `how_it_works` | curious readers | One paragraph, high level, no math | ~120 words |

- The `label` and `summary` limits are hard: the schema rejects longer text and newlines.
- The schema rejects `description` and `how_it_works` past 150 words, which leaves slack over the ~120-word target.
- A `stub` may leave all prose empty. A `drafted` or `verified` record needs `summary`, `description` and
  `how_it_works`.
- Longer, guide-shaped content goes in the structured fields (`uses`, `choosing.guide`, `parameters`), not in these.

## Writing rules

- Plain language. Name the visible effect before the mechanism.
- Every factual claim traces to an entry in the record's `sources`. A claim that can't be traced is left out or
  listed in `open_questions`.
- Mark uncertainty in `open_questions`, not in hedged prose.
- Sizes, hashes and defaults are recorded as found. Prose avoids unsourced quality rankings ("best", "most
  accurate") and says what a tool is better at instead.
- ASCII hyphens, not em or en dashes; the schema rejects both. English only (#149 covers translation).
- A `verified` record has at least one non-`tutorial` source and no `open_questions`.

## Sources

- **`local_code` sources name a project, its upstream commit and a path inside it,** never a local filesystem path.
  Each also carries a `url` pinned to that commit when the project is public.
- **`tutorial` sources may support usage advice only** (`uses`, `when_to_adjust`, `choosing.guide`), not claims about
  what a model was trained on or how a preprocessor works.
- **Model `license` values are the Hugging Face model card's `license` metadata.** Records without card metadata leave
  `license` empty.

## Research method

- Search the web first: upstream repos and wikis, Hugging Face model cards, paper abstracts and figures, then
  tutorials for usage advice.
- Local installs fill gaps: preprocessor parameter names, defaults and ranges, annotator download URLs, and backend
  display behavior. Cite them as `local_code` by upstream commit.
- Take hashes from Hugging Face's file listing (`/api/models/<repo>/tree/main?recursive=true` gives `size` and
  `lfs.oid`), from a range request (see "Model hashes"), or from hashing a local copy. Don't download large files
  just to hash them.
- **Two files are one record only when their weights match.** A file name or a similar size doesn't settle it.
  Range-request each safetensors header (an 8-byte length, then JSON listing every tensor's dtype, shape and byte
  offsets), then fetch a few small tensors from both files and compare their values. A dtype conversion leaves
  differences at fp16 or bf16 rounding; a separate training differs by the size of the weights themselves.

## Scope

- **Base models: SD1.5 and SDXL.** `base_models` takes more values later.
- **Control types: every type a backend presents,** plus `qr_code`, a grouping for QR and illusion models, which
  every backend files under "All".
- **Preprocessors: everything in sd-webui-controlnet, Forge, reForge, Forge Neo and comfyui_controlnet_aux,** plus
  the ComfyUI core nodes a guide would send an artist to (`Canny`, `ImageInvert`, `ImageBlur`). reForge's `*_variant`
  preprocessors are its copies of the A1111 legacy implementations. They are aliases in the base record's
  `names.forge`.
- **Models: the families a user is likely to have downloaded.** These are lllyasviel ControlNet 1.0 and 1.1, TencentARC
  T2I-Adapters (SD1.x and SDXL), diffusers SDXL ControlNets, every file in `lllyasviel/sd_control_collection`
  (SargeZT, kohya LLLite, Stability control-LoRAs, thibaud), xinsir (including union and ProMax), TTPlanet tile,
  monster-labs, DionTimmer, Nacholmo and latentcat/ioclab QR and brightness models, h94 IP-Adapter and FaceID,
  PuLID, InstantID, PhotoMaker, MistoLine, SparseCtrl, MediaPipe Face, the SDXL inpainting ControlNets in the
  recorded fixtures, and the three SD1.5 ControlNets that comfyui_controlnet_aux's table pairs with preprocessors
  (the Depth Anything ControlNet, HandRefiner's inpaint depth hand model and the animal pose ControlNet).

### Left out

- **Names that aren't preprocessors** are listed in `NOT_PREPROCESSORS` in `tests/unit/test_controlnet_data.py`.
  These are WebUI "none", comfyui_controlnet_aux helper nodes, pose-keypoint postprocessing, and optical flow, which
  serves video models.
- **TencentARC CoAdapters** (`coadapter-*-sd15v1`): no backend ships a control type or preprocessor for them.
- **bdsqlsz qinglong ControlNet-LLLite** (19 SDXL models): a large third-party family. Add it if users turn out
  to have it.
- **Other Civitai fine-tunes,** and SD2.1 releases of the same families.

## Status

| Kind | Records | Stub | Drafted | Verified |
|---|---|---|---|---|
| Control types | 23 | 14 | 9 | 0 |
| Preprocessors | 81 | 30 | 17 | 34 |
| Models | 127 | 58 | 11 | 58 |

`python scripts/controlnet_data.py` prints the current counts. Update this table when a session changes them.
