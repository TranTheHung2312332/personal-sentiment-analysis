import torch
import re
from .path import EMBEDDING_PATH


class Tokenizer:
    def __init__(self, word2idx, max_len=128):
        self.word2idx = word2idx
        self.max_len = max_len

    
    def tokenize(self, text):
        patterns = [
            r"\[[A-Z_]+\]",
            r"<\/?[\w_]+>",
            r"\w+",
            r"[?!]{2,}",
            r"\.{3,}",
            r"[^\w\s]"
        ]

        combined = re.compile("|".join(patterns), re.UNICODE)

        return combined.findall(text)



    def encode(self, text):
        tokens = self.tokenize(text)

        ids = [self.word2idx.get(token, self.word2idx['<unk>']) for token in tokens]
        ids = ids[:self.max_len]
        ids += [self.word2idx['<pad>']] * (self.max_len - len(ids))
        ids = torch.tensor(ids).unsqueeze(0)

        return ids


    def create_tokenizer():
        embedding_obj = torch.load(EMBEDDING_PATH, map_location='cpu')
        return Tokenizer(embedding_obj['word2idx'])