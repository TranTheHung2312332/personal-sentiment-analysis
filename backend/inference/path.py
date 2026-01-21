import os

BASE_DIR = os.path.dirname(__file__)  # inference/

CONTRACTION_PATH = os.path.join(BASE_DIR, "mapping", "contraction_prod.csv")
EMOJI_PATH = os.path.join(BASE_DIR, "mapping", "emoji_prod.csv")
RELIGION_PATH = os.path.join(BASE_DIR, "mapping", "religion_list_prod.txt")

EMBEDDING_PATH = os.path.join(BASE_DIR, "model", "embedding_prod.pt")
CLF_PATH = os.path.join(BASE_DIR, "model", "clf_prod.pt")
