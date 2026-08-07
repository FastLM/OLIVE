# Teacher VLAs (git submodule links)

| Folder | Upstream | Model |
|--------|----------|--------|
| [`pi0.5/`](https://github.com/Physical-Intelligence/openpi) | [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) | **π₀.₅** — distillation teacher |
| [`pi0.6/`](https://github.com/Physical-Intelligence/openpi) | [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) | **π₀.₆** — same upstream; official weights not yet public ([model card](https://website.pi-asset.com/pi06star/PI06_model_card.pdf), [tracking issue](https://github.com/Physical-Intelligence/openpi/issues/791)) |

OLIVE freezes a compact **BaseController** distilled from these teachers
(see [`../distillation/`](../distillation/)).

## Initialise locally

```bash
# π0.5 only (recommended for distillation)
git submodule update --init --depth 1 teachers/pi0.5

# both pointers
git submodule update --init --depth 1
```

Shallow clone keeps the download small; omit `--depth 1` for a full history.
