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

## Results

| Metric    | Value  |
|-----------|--------|
| AUC       | 0.83    |
| Accuracy  | 78%    |
| F1        | 0.76    |
| Precision | 0.79    |
| Recall    | 0.73    |


## File structure

```
veraclip/
├── data/
│   ├── download.py       ← pulls VERITE + NewsCLIPpings from HuggingFace
│   ├── preprocess.py     ← image transforms, binary label mapping, tensor cache
│   ├── dataset.py        ← PyTorch Dataset wrapping the HF cache
│   └── augment.py        ← caption corruption for hard negative generation
├── model/
│   ├── config.yaml       ← all hyperparameters in one place
│   ├── clip_base.py      ← CLIP-ViT-L/14 wrapper (encode_image, encode_text)
│   ├── lora_adapter.py   ← LoRA r=16 applied to projection + attention layers
│   ├── fusion_head.py    ← MLP: [cosine | img_emb | txt_emb] → score
│   ├── loss.py           ← BCE + contrastive margin loss
│   └── train.py          ← full training loop (fp16, cosine LR, WandB)
├── explain/
│   ├── gradcam.py        ← GradCAM on last ViT encoder layer → spatial heatmap
│   ├── attn_rollout.py   ← rollout across all layers → patch + token importance
│   └── visualize.py      ← heatmap rendering, overlay blending, report figures
├── eval/
│   ├── metrics.py        ← AUC, F1, precision, recall, threshold search
│   ├── evaluate.py       ← VERITE test set evaluation + score distribution plot
│   ├── batch_score.py    ← score 10k NewsCLIPpings pairs → CSV
│   └── report.py         ← GradCAM figures for worst false positives/negatives
├── app/
│   ├── inference.py      ← single-pair pipeline (preprocess → infer → explain)
│   └── app.py            ← Gradio web interface
└── tests/
    ├── test_model.py     ← backbone, head, loss unit tests
    ├── test_dataset.py   ← dataset loading with mock data
    ├── test_gradcam.py   ← attention rollout, visualize utilities
    └── test_inference.py ← end-to-end pipeline integration test
```

## Training on Google Colab (free T4)

```python
# In a Colab cell:
!git clone https://github.com/yourname/veraclip && cd veraclip
!pip install -r requirements.txt
!python -m data.download
!python -m data.preprocess
!python -m model.train
```

Connect Google Drive to persist checkpoints between sessions.

## Datasets

- **VERITE** (`Ftheodorakis/VERITE`): 3,600 labeled pairs, three classes
- **NewsCLIPpings** (`g-luo/newsclipper`): 100k+ pairs for large-scale evaluation
