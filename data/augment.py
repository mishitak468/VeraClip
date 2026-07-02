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


