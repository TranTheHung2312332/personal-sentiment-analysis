import re
import pandas as pd
import emoji
import numpy as np
from .path import EMOJI_PATH, CONTRACTION_PATH, RELIGION_PATH


# === Religion preprocessing ===
with open(RELIGION_PATH, 'r', encoding='utf-8') as f:
    RELIGION_LIST = [line.strip() for line in f if line.strip()]

RELIGION_PATTERN = re.compile(
    r'\b(' + '|'.join(map(re.escape, RELIGION_LIST)) + r')\b',
    flags=re.IGNORECASE
)

def normalize_religion(text):
    return re.sub(RELIGION_PATTERN, '[RELIGION]', text)


# === Contraction ===
contraction_df = pd.read_csv(CONTRACTION_PATH, encoding='utf-8')
contraction_map = {key: value for key, value in zip(contraction_df.contraction, contraction_df.extension)}

contraction_pattern = re.compile(
    r"\b(" + "|".join(map(re.escape, contraction_map.keys())) + r")\b",
    flags=re.IGNORECASE
)

def replace_contraction(match):
    w = match.group(0).lower()
    if w in contraction_map:
        return contraction_map[match.group(0).lower()]
    else:
        return w

def extend(text):
    return re.sub(contraction_pattern, replace_contraction, text)



# === Emoji ===
emoji_df = pd.read_csv(EMOJI_PATH, encoding='utf-8')
emoji_df['Score'] = np.tanh(
    np.log((emoji_df.Positive + 1) / (emoji_df.Negative + 1))
)

GENDER_EMOJI_MAP = {
    "\u2640": "[EMO_FEMALE]",  # ♀
    "\u2642": "[EMO_MALE]",    # ♂
    "\u26A7": "[EMO_TRANS]"    # ⚧
}

def normalize_emoji(e):
    e = re.sub(r"\uFE0F", "", e)
    if e in GENDER_EMOJI_MAP:
        return GENDER_EMOJI_MAP[e]
    e = re.sub(r"\u200d", "", e)
    return e

emoji_df['Emoji_norm'] = emoji_df['Emoji'].apply(normalize_emoji)
emoji_df = emoji_df[emoji_df['Emoji_norm'].str.len() > 0]

emoji_list = sorted(
    emoji_df['Emoji_norm'].unique(),
    key=len,
    reverse=True
)

emoji_map = dict(
    zip(
        emoji_df['Emoji_norm'],
        zip(emoji_df['Unicode name'], emoji_df.Score)
    )
)

def extract_emoji(sentence, beta=1.0):
    emoji_scores = []
    new_text = sentence
    strongest = 0.0

    for e in emoji.emoji_list(sentence):
        em = e['emoji']
        norm_em = normalize_emoji(em)

        if not norm_em.startswith("[EMO_"):
            name, score = emoji_map.get(norm_em, ('[EMO]', 0.0))
            if score != 0.0:
                emoji_scores.append(score)
        else:
            name = norm_em

        new_text = new_text.replace(em, name)

    if emoji_scores:
        strongest = max(emoji_scores, key=lambda s: abs(s))

    return new_text, strongest * beta



# === Markdown reddit ===
def extract_markdown(text):
    # spoiler
    text = re.sub(r">!(.+?)!<", r" <spoiler> \1 </spoiler> ", text)

    # bold + italic
    text = re.sub(r"\*\*\*(.+?)\*\*\*", r" <bi> \1 </bi> ", text)

    # bold
    text = re.sub(r"\*\*(.+?)\*\*", r" <b> \1 </b> ", text)

    # italic *
    text = re.sub(r"\*(?!\*)(.+?)(?<!\*)\*", r" <i> \1 </i> ", text)

    # strike
    text = re.sub(r"~~(.+?)~~", r" <s> \1 </s> ", text)

    # quote (line-based)
    text = re.sub(r"^>(.+)", r" <q> \1 </q> ", text, flags=re.M)

    # inline code
    text = re.sub(r"`(.+?)`", r" <code> \1 </code> ", text)

    # triple double quotes
    text = re.sub(r'"""\s*(.+?)\s*"""', r' <quote> \1 </quote> ', text)

    return text



# === Mention ===
def normalize_mention(text):
    return re.sub(r"(?<!\w)@[A-Za-z_][A-Za-z0-9_]{1,30}", '[MENTION]', text)    



# === URL ===
def normalize_url(text):
    return re.sub(r"https?://\S+|www\.\S+", '[URL]', text)



# === Time ===
def normalize_time(text):
    return re.sub(r"\b(?:1[0-2]|0?[1-9]):[0-5][0-9]\s*(?i:am|pm)\b", '[TIME]', text)



# === Date ===
def normalize_date(text):
    # ISO 8601 datetime: 2026-01-07T10:30:00
    text = re.sub(r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\b", "[DATE]", text)
    # YYYY-MM-DD
    text = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", "[DATE]", text)
    # MM/DD/YYYY
    text = re.sub(r"\b\d{1,2}/\d{1,2}/\d{4}\b", "[DATE]", text)
    # DD-MM-YYYY
    text = re.sub(r"\b\d{1,2}-\d{1,2}-\d{4}\b", "[DATE]", text)
    # Month Day, Year (Jan 7, 2026 or January 7, 2026)
    text = re.sub(
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|"
        r"January|February|March|April|May|June|July|August|September|October|November|December)"
        r"\s+\d{1,2},\s*\d{4}\b", "[DATE]", text
    )
    # Day Month Year (7 Jan 2026 or 7 January 2026)
    text = re.sub(
        r"\b\d{1,2}\s+"
        r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|"
        r"January|February|March|April|May|June|July|August|September|October|November|December)"
        r"\s+\d{4}\b", "[DATE]", text
    )
    # Compact numeric YYYYMMDD
    text = re.sub(r"\b\d{8}\b", "[DATE]", text)
    
    return text



# === Hashtag ===
def normalize_hashtag(text):

    def repl(m):
        tag = m.group()[1:]
        tag = tag.lower()
        return f"[HASHTAG] {tag}"
    
    return re.sub(r'#\w+', repl, text)



# === Whitespace ===
def normalize_whitespace(text):
    return re.sub(r'\s+', ' ', text).strip()



# === Lowercase ===
def lowercase(text):
    token_pattern = r'(\[[A-Z_]+\])'

    parts = re.split(token_pattern, text)

    parts = [p.lower() if not re.fullmatch(token_pattern, p) else p for p in parts]

    return ''.join(parts)



# === Punctuation ===
def normalize_punctuation(text: str) -> str:
    text = re.sub(r'\.{3,}', '...', text)

    text = re.sub(r'!{3,}', '!!', text)
    text = re.sub(r'\?{3,}', '??', text)

    text = re.sub(r'(!\?|\?!){2,}', '!?', text)

    text = re.sub(r'\s+([!?.,])', r'\1', text)
    text = re.sub(r'([!?.,])\s+', r'\1 ', text)

    text = re.sub(r'([,;:]){2,}', r'\1', text)

    return text



# === All uppercase ===
def extract_is_all_uppercase(text):
    return text, int(text.isupper())



# === Uppercase ratio ===
def extract_uppercase_ratio(text):
    clean_text = re.sub(r'\[[A-Z_]+\]', '', text)
    
    alphas = [c for c in clean_text if c.isalpha()]
    
    if not alphas:
        return text,0.0
    
    return text, sum(c.isupper() for c in alphas) / len(alphas)



# === Exclaimination ===
def extract_exclamation_intensity(text, cap=5):
    max_run = 0
    cur = 0
    for c in text:
        if c == '!':
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 0

    return text, min(max_run, cap) / cap