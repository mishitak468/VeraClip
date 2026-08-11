# VeraClip — Multimodal Misinformation Detector

Detects whether a social media caption accurately describes its accompanying image.
Fine-tuned CLIP-ViT-L/14 with LoRA on the VERITE dataset. GradCAM + attention rollout
surface causally relevant image regions and caption tokens.

## Architecture

```
Image  →  CLIP ViT-L/14  →  visual_projection  ─┐
                                                  ├─→  InconsistencyHead  →  score [0,1]
Text   →  CLIP text enc  →  text_projection    ─┘

Explainability:
  GradCAM on last ViT encoder layer  →  spatial heatmap (which image regions)
  Attention rollout across all layers →  patch importance map
  Per-token hidden state norms        →  caption token importance
```

## Quickstart

```bash
git clone https://github.com/yourname/veraclip
cd veraclip
make setup
make env           # copy .env.example → .env, fill in tokens
make data          # download VERITE + NewsCLIPpings (~2 GB)
make train         # ~4h on T4, ~1.5h on A100
make eval          # test set metrics → eval/results/test_metrics.json
make batch-score   # score 10k NewsCLIPpings pairs
make app           # launch Gradio at localhost:7860
make test          # run test suite
```
