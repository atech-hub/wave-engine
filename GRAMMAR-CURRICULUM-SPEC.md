# GRAMMAR CURRICULUM TRAINING SPEC
# Date: 2026-03-22
# For: Code (Claude Code)
# From: Desktop + Marco
# Status: Prepare alongside current wikitext run

---

## The Idea

Instead of throwing raw Wikipedia at the model and hoping it figures out
what English is, teach it HOW English works first using grammar textbooks.
These books contain thousands of example sentences, each demonstrating
a specific rule. The model learns structure AND correct English simultaneously.

Marco's insight: "The model wants to know how English works. Should be
able to talk without wikitext really."

## Data Sources (all Project Gutenberg, public domain)

Code needs to download these plain text files:

| Book | Gutenberg URL | Size (approx) | Level |
|------|--------------|----------------|-------|
| Plain English — Marian Wharton | gutenberg.org/files/40550/40550-0.txt | ~200KB | Beginner |
| Practical Grammar — Thomas Wood | gutenberg.org/files/22577/22577-0.txt | ~150KB | Intermediate |
| An English Grammar — Baskervill & Sewell | gutenberg.org/files/14006/14006-0.txt | ~400KB | Comprehensive |
| Advanced English Grammar — Kittredge & Farley | gutenberg.org/files/45814/45814-0.txt | ~250KB | Advanced |
| Word Study and English Grammar — Hamilton | gutenberg.org/files/30036/30036-0.txt | ~100KB | Supplementary |

Total: ~1.1MB of structured English grammar text.

### Download method

Code should download from Project Gutenberg's plain text URLs.
If gutenberg.org is blocked by network policy, alternatives:
- Use a Gutenberg mirror
- Marco can download manually and place in C:\claude\wave-engine\data\grammar\
- Desktop fetched the full HTML of "Plain English" — Code can strip HTML tags

### Text cleaning

Strip Gutenberg headers/footers (everything before "*** START OF" and after
"*** END OF"). Remove any non-ASCII artifacts. Keep all example sentences,
exercises, and spelling lessons — these are training data, not noise.

## Curriculum Structure

### Stage 1: Plain English (Wharton) — Lessons 1-10 (~500 iters)

This book starts from absolute basics:
- Lesson 1: What is language? What is a sentence? Subject and predicate.
  Example: "Men work. Flowers fade. Snow flies."
- Lesson 2: Kinds of sentences. Nouns and verbs.
  Example: "A noun is the name of something. A verb is a word that asserts."
- Lesson 3: All 8 parts of speech overview.
- Lesson 4: Classes of nouns (proper, common, collective, abstract).
- Lesson 5: Complete and incomplete verbs. Transitive, copulative.
- Lesson 6: Verb inflections. Regular and irregular verbs.
- Lesson 7: Time forms (tenses). Present, past, future, perfect forms.
- Lesson 8: Progressive verb phrases. Active and passive voice.
- Lesson 9: Participles and infinitives.
- Lesson 10: Helping verbs (shall, will, may, can, must, ought).

Plus interleaved SPELLING lessons covering:
- Vowels and consonants
- Diacritical marks
- Syllabification
- Accent
- Compound words
- Prefixes and suffixes

This is perfect foundational training. Every lesson has:
1. Explanation of the rule
2. Multiple example sentences demonstrating the rule
3. Exercises with more example sentences
4. Poetry/prose quotations showing rules in real English

### Stage 2: Practical Grammar (Wood) + Word Study (Hamilton) (~500 iters)

Reinforces Stage 1 with different explanations and examples.
Adds sentence construction patterns and common errors.

### Stage 3: English Grammar (Baskervill & Sewell) (~500 iters)

Comprehensive grammar with analysis of sentences.
Part I: Parts of Speech + Inflections
Part II: Analysis of Sentences
Part III: Syntax (uses of words)

This is the most thorough treatment — the model gets deep exposure
to sentence structure and how words function in context.

### Stage 4: Advanced Grammar (Kittredge & Farley) (~500 iters)

Complex syntax, sentence analysis, advanced constructions.
The model has learned the basics — now it sees sophisticated English.

### Stage 5 (optional): Wikitext (~2000+ iters)

If the model can already produce coherent English after Stages 1-4,
wikitext becomes supplementary vocabulary expansion, not structural
learning. The model already knows HOW English works — wikitext just
gives it more WHAT to talk about.

## Implementation

### Option A: Concatenated corpus with markers (simple)

Concatenate all grammar texts in order into one file:
```
data/grammar/grammar_corpus.txt
```

Train on this single file: `wave-engine data/grammar/grammar_corpus.txt --candle --bpe --iters 2000`

The natural order of the text (basics → advanced) provides implicit curriculum.
At 2000 iters × batch=8 × seq=256 = 4.1M tokens. If the corpus is ~200K tokens,
the model sees it ~20 times. That's enough for a small corpus of structured content.

### Option B: Staged loading (more control)

Train on each stage separately with checkpointing:
```bash
# Stage 1: Foundations
wave-engine data/grammar/01_plain_english.txt --candle --bpe --iters 500 --batch 8
# Save checkpoint

# Stage 2: Reinforcement
wave-engine data/grammar/02_combined.txt --candle --bpe --iters 500 --batch 8 --resume checkpoint

# Stage 3: Comprehensive
wave-engine data/grammar/03_baskervill.txt --candle --bpe --iters 500 --batch 8 --resume checkpoint

# Stage 4: Advanced
wave-engine data/grammar/04_kittredge.txt --candle --bpe --iters 500 --batch 8 --resume checkpoint
```

### Option C: Hybrid (recommended)

First 1000 iters on grammar corpus only (multiple passes over small corpus).
Then switch to wikitext for remaining iters (model applies learned structure
to diverse content).

```bash
# Phase 1: Learn English structure
wave-engine data/grammar/grammar_corpus.txt --candle --bpe --iters 1000 --batch 8
# Save checkpoint

# Phase 2: Apply to real English
wave-engine data/wikitext.txt --candle --bpe --iters 2000 --batch 8 --resume phase1_checkpoint
```

## Why This Could Work Better

1. **Dense signal**: Every sentence in a grammar textbook demonstrates a rule.
   Wikitext is mostly content noise with occasional structural patterns.
   Grammar text is pure structure signal.

2. **Repetition with variation**: The textbooks repeat the same patterns
   (subject-verb, noun-adjective, etc.) hundreds of times with different
   words. This is exactly how humans learn — same pattern, different content.

3. **Band alignment**: The wave-engine's low-frequency bands capture structure,
   high-frequency bands capture content. Grammar text is almost PURE structure.
   The low-frequency bands get clean training signal from the start.

4. **Small corpus, many passes**: 200K tokens seen 20 times > 4M tokens seen once.
   Repetition on structured content builds deeper representations than
   single-pass exposure to diverse content.

5. **Self-contained**: No external dependencies. No 500MB parquet files.
   The entire training corpus fits in 1MB. The model can run on any machine.

## Test Protocol

1. Train on grammar corpus only (2000 iters, ~6 hours at 9.2s/iter)
2. Serve the model through wave-server
3. Test with prompts like:
   - "The cat" → does it complete with a verb? (shows it learned sentence structure)
   - "A noun is" → does it complete with a definition? (shows it learned grammar)
   - "The workers" → does it produce subject-verb-object? (shows it learned English)
4. Compare against the wikitext-only baseline (currently training)

If the grammar model produces better structured English than the wikitext model
at the same number of iterations, the curriculum approach wins.

## What Code Should Do NOW

1. Create directory: `C:\claude\wave-engine\data\grammar\`
2. Download the 5 Gutenberg texts (plain text versions)
3. Clean: strip Gutenberg headers/footers
4. Concatenate in order into `grammar_corpus.txt`
5. Report file size and token count (run through BPE encoder)
6. Ready for training after current wikitext run finishes

The wikitext run continues as the baseline. The grammar corpus prepares
in parallel. When both finish, we compare.

## Files to create

```
C:\claude\wave-engine\data\grammar\
├── 01_plain_english.txt       ← Wharton (cleaned)
├── 02_practical_grammar.txt   ← Wood (cleaned)
├── 03_english_grammar.txt     ← Baskervill & Sewell (cleaned)
├── 04_advanced_grammar.txt    ← Kittredge & Farley (cleaned)
├── 05_word_study.txt          ← Hamilton (cleaned)
└── grammar_corpus.txt         ← all combined in order
```
