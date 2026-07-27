"""
veraclip/eval/batch_score.py
Scores all 10,000 NewsCLIPpings pairs and writes a CSV.
This is what generates the resume stat:
  "Processed and scored 10,000+ image-caption pairs during evaluation"

Run: python -m eval.batch_score

Outputs:
  eval/results/newsclipper_scores.csv  — id, score, label, caption_snippet
  eval/results/newsclipper_summary.json — aggregate statistics
"""

import csv
import json
import yaml
import torch
from pathlib import Path
from tqdm import tqdm
from datasets import load_from_disk
