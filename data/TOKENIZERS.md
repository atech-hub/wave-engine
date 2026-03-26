# Tokenizers

BPE tokenizers trained on specific corpora. Use `--bpe --tokenizer <path>` to select.

## Available

| File | Vocab | Trained on | Use case |
|------|-------|-----------|----------|
| `tokenizer.json` | 50,257 | GPT-2 (pre-trained) | 768-dim production models |
| `tokenizer_512.json` | 512 | combined_10mb.txt (raw) | 168-dim with original 12MB corpus |
| `tokenizer_512_clean.json` | 512 | combined_clean.txt (prose) | 168-dim with prose-converted corpus |
| `tokenizer_512_gs.json` | 512 | grammar+Shakespeare (2.6MB) | 168-dim focused English training |
| `tokenizer_768.json` | 768 | combined_10mb.txt | Word coverage testing |
| `tokenizer_1k.json` | 1,024 | combined_10mb.txt | 256-dim BPE training |
| `tokenizer_1k_gs.json` | 1,024 | grammar+Shakespeare (2.6MB) | Richest harmonic structure (7 harmonics at 168-dim) |
| `tokenizer_2k.json` | 2,048 | combined_10mb.txt | Early BPE experiments |

## Choosing a tokenizer

The vocab size must match the model's embedding dimension:
- **168-dim (84 bands):** 512 or 1K vocab. 1K uses more harmonics.
- **256-dim (128 bands):** 512 to 2K vocab.
- **384-dim (192 bands):** 2K to 4K vocab.
- **768-dim (384 bands):** 50K vocab.

See `configs/` for proven settings per dimension.

## Corpus-specific vs general

Tokenizers trained on the SAME corpus you train on produce better merges.
`tokenizer_512_gs.json` is optimised for grammar+Shakespeare patterns.
`tokenizer_512.json` is optimised for the full 12MB mixed corpus.

## Training your own

```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
import json

tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
trainer = trainers.BpeTrainer(vocab_size=512, min_frequency=2, special_tokens=["<|endoftext|>"])
tokenizer.train(["your_corpus.txt"], trainer)
tokenizer.save("your_tokenizer.json")

# Fix merges format for wave-engine compatibility
with open("your_tokenizer.json", encoding="utf-8") as f:
    data = json.load(f)
if isinstance(data["model"]["merges"][0], list):
    data["model"]["merges"] = [" ".join(m) for m in data["model"]["merges"]]
with open("your_tokenizer.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False)
```

## Harmonic scaling finding

Vocab complexity drives harmonic usage in the wave architecture:

| Vocab | Harmonics used | Example pairs |
|-------|---------------|---------------|
| 65 (char) | 2 (n=1, n=9) | boy/ball: n=1 |
| 512 BPE | 4 (n=1,2,6,7) | cat/dog: n=6, noun/verb: n=2 |
| 1,024 BPE | 7 (n=1,4,5,6,7,9,11) | boy/ball: n=11, noun/verb: n=6, mat/rug: n=5 |

More tokens to distinguish → more harmonics activated. The wave basis scales naturally.
