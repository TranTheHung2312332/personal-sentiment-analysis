from .model import Model
from .tokenizer import Tokenizer
from .preprocessing import *
import torch
import torch.nn.functional as F

def preprocessing(
        text,
        religion_norm = True,
        contraction = True,
        emoji_mapping = True,
        emoji_score = True,
        markdown = True,
        mention = True,
        url = True,
        time = True,
        date = True,
        hashtag = True,
        lowercase_norm = True,
        punctuation_norm = True,
        uppercase_ratio = True,
        ex_intensity = True,
        ex_intensity_cap = 5,
        emoji_score_scale = 1.0
):
    extra_feats = {}

    # Religion
    if religion_norm:
        text = normalize_religion(text)

     # Exclamation intensity
    if ex_intensity:
        text, ex_val = extract_exclamation_intensity(text, cap=ex_intensity_cap)
    else:
        ex_val = 0.0
    extra_feats['ex_intensity'] = ex_val

    # Markdown
    if markdown:
        text = extract_markdown(text)

    # Contraction
    if contraction:
        text = extend(text)

    # Emoji
    if emoji_mapping:
        text, emoji_score_val = extract_emoji(text, beta=emoji_score_scale)
    else:
        emoji_score_val = 0.0

    if emoji_score == False:
        emoji_score_val = 0.0

    extra_feats['emoji_score'] = emoji_score_val

    # Mention
    if mention:
        text = normalize_mention(text)

    # URL
    if url:
        text = normalize_url(text)

    # Time
    if time:
        text = normalize_time(text)

    # Date
    if date:
        text = normalize_date(text)

    # Hashtag
    if hashtag:
        text = normalize_hashtag(text)

    # Whitespace (always)
    text = normalize_whitespace(text)

    # Uppercase features
    text, is_all_upper = extract_is_all_uppercase(text)

    if uppercase_ratio:
        text, upper_ratio = extract_uppercase_ratio(text)
    else:
        upper_ratio = 0.0

    extra_feats['all_uppercase'] = is_all_upper
    extra_feats['uppercase_ratio'] = upper_ratio

    # Lowercase
    if lowercase_norm:
        text = lowercase(text)

    # Punctuation
    if punctuation_norm:
        text = normalize_punctuation(text)

    return text, extra_feats



def inference(tokenizer: Tokenizer, model: Model, text, extra_feats, neutral_threshold=1/3):
    model.eval()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    input_ids = tokenizer.encode(text).to(device)
    extra_feats = torch.tensor(extra_feats, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_ids, extra_feats)[0]
        probs = F.softmax(logits, dim=-1)
        idxmax = torch.argmax(probs)

        if idxmax == 1 and probs[idxmax] < neutral_threshold:
            idxmax = 0 if probs[0] >= probs[2] else 2

    return {
        "label": idxmax,
        "probs": probs
    }




