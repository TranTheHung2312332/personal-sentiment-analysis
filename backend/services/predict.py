from inference.model import Model
from inference.tokenizer import Tokenizer
from inference.pipeline import preprocessing, inference
from schemas.predict import PredictRequest, PredictResponse, PreprocessOptions

model = Model.create_model()
tokenizer = Tokenizer.create_tokenizer()

EXTRA_FEAT_ORDER = ['ex_intensity', 'emoji_score', 'all_uppercase', 'uppercase_ratio']

LABEL_MAP = {
    0: "Positive",
    1: "Neutral",
    2: "Negative"
}

def predict(req: PredictRequest) -> PredictResponse:
    options = req.preprocess or PreprocessOptions()
    neutral_threshold = req.neutral_threshold

    clean_text, extra_feats_dict = preprocessing(text=req.text, **options.model_dump())
    extra_feats = [extra_feats_dict[feat] for feat in EXTRA_FEAT_ORDER]

    result = inference(tokenizer, model, clean_text, extra_feats, neutral_threshold)

    idxmax = result["label"]
    probs = result["probs"]

    label = LABEL_MAP[int(idxmax)]

    return PredictResponse(
        clean_text=clean_text,
        ex_intensity=extra_feats_dict["ex_intensity"],
        emoji_score=extra_feats_dict["emoji_score"],
        all_uppercase=extra_feats_dict["all_uppercase"],
        uppercase_ratio=extra_feats_dict["uppercase_ratio"],
        label=label,
        probs=probs
    )