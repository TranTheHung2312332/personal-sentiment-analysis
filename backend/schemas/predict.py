from pydantic import BaseModel
from typing import Optional

class PreprocessOptions(BaseModel):
    religion_norm: bool = True
    contraction: bool = True
    emoji_mapping: bool = True
    emoji_score: bool = True
    markdown: bool = True
    mention: bool = True
    url: bool = True
    time: bool = True
    date: bool = True
    hashtag: bool = True
    lowercase_norm: bool = True
    punctuation_norm: bool = True
    uppercase_ratio: bool = True
    ex_intensity: bool = True

    ex_intensity_cap: int = 5
    emoji_score_scale: float = 1.0


class PredictRequest(BaseModel):
    text: str
    preprocess: Optional[PreprocessOptions] = None
    neutral_threshold: Optional[float] = 1/3


class PredictResponse(BaseModel):
    clean_text: str
    ex_intensity: float
    emoji_score: float
    all_uppercase: bool
    uppercase_ratio: float
    probs: list[float]
    label: str