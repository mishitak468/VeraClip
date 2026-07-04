"""
veraclip/data/augment.py
Caption-level augmentation to create hard negatives during training.
Swaps location names and dates so the model learns fine-grained inconsistency.
Imported and called inside train.py's data loop.
"""

import re
import random

LOCATION_POOL = [
    "New York", "Mumbai", "Lagos", "São Paulo", "Jakarta",
    "Cairo", "Seoul", "Mexico City", "London", "Tokyo",
    "Paris", "Nairobi", "Dhaka", "Istanbul", "Beijing",
]

DATE_POOL = [
    "January 2019", "March 2020", "June 2021",
    "October 2022", "February 2023", "September 2024",
    "April 2018", "November 2017",
]

DATE_PATTERN = re.compile(
    r"\b(January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s+\d{4}\b"
)


def swap_location(caption: str) -> str:
    """Replace first detected location with a randomly chosen wrong one."""
    for loc in LOCATION_POOL:
        if loc.lower() in caption.lower():
            replacement = random.choice([l for l in LOCATION_POOL if l != loc])
            return re.sub(
                re.escape(loc), replacement,
                caption, count=1, flags=re.IGNORECASE
            )
    return caption


def swap_date(caption: str) -> str:
    """Replace first detected month+year with a wrong one."""
    match = DATE_PATTERN.search(caption)
    if match:
        replacement = random.choice(DATE_POOL)
        return caption[: match.start()] + replacement + caption[match.end() :]
    return caption


def create_hard_negative(caption: str) -> str:
    """
    Chain location + date swap.
    Used to augment the training set with synthetic inconsistent captions.
    """
    return swap_date(swap_location(caption))


def maybe_augment(caption: str, label: int, prob: float = 0.15) -> tuple[str, int]:
    """
    With probability `prob`, corrupt a consistent caption to create a hard negative.
    Only applies to label=0 examples (consistent pairs).
    Returns (caption, label).
    """
    if label == 0 and random.random() < prob:
        return create_hard_negative(caption), 1
    return caption, label
