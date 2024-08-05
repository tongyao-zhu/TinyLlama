import os
import sys

from pathlib import Path
import sys
sys.path.append('/home/aiops/zhuty/tinyllama')

from lit_gpt.packed_dataset import CombinedDataset, PackedDataset
from lit_gpt import Tokenizer
# wd = Path(__file__).parent.parent.resolve()
# sys.path.append(str(wd))
# print(wd)
TOKENIZER_PATH = '/home/aiops/zhuty/tinyllama/models'

tokenizer = Tokenizer(Path(TOKENIZER_PATH))

print("Voab size:", tokenizer.vocab_size)

texts = ["""
tyzhu/tinyllama_common_tokenizer is not a local folder and is not a valid model identifier listed on 'https://huggingface.co/models'
If this is a private repository, make sure to pass a token having permission to this repo either by logging in with `huggingface-cli login` or by passing `token=<your_token>"""]
text = texts[0]

print("Text:", text)
original_ids = tokenizer.encode(text)
print("Original ids:", original_ids)
print("Length of original ids:", len(original_ids))
dropout_ids = tokenizer.encode(text, bpe_dropout=1)
print("Dropout ids:", dropout_ids)
print("Length of dropout ids:", len(dropout_ids))
assert len(dropout_ids) == len(text)
print("Dropout ids are the same length as the text")


