"""
veraclip/eval/evaluate.py
Runs the trained model on the VERITE test split and saves metrics.
Run: python -m eval.evaluate

Outputs:
  eval/results/test_metrics.json   — full metric dict
  eval/results/score_distribution.png — histogram of scores by class
"""

import json
import yaml
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from transformers import CLIPProcessor
from tqdm import tqdm

from model.clip_base   import CLIPBackbone
from model.fusion_head import InconsistencyHead
from data.dataset      import MisinfoDataset
from eval.metrics      import compute_all_metrics, find_best_threshold

RESULTS_DIR = Path("eval/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


