[![REUSE status](https://api.reuse.software/badge/github.com/SAP-samples/case)](https://api.reuse.software/info/github.com/SAP-samples/case)
[![HuggingFace - Model](https://img.shields.io/badge/HuggingFace-Model-orange?logo=huggingface)](https://huggingface.co/SAP/CASE)


# CASE: Context-Aware Semantic Embeddings
Inference code described in the paper ["Enhancing Tabular Learners with
Context-Aware Semantic Embeddings"](https://arxiv.org/abs/2608.03565).

## Citation

If you use CASE in your research or want to refer to our work, please cite:
```
@inproceedings{
    case,
    title={Enhancing Tabular Learners with Context-Aware Semantic Embeddings},
    author={Günther Schindler and Maximilian Schambach and Johannes Höhne},
    booktitle={Advances in Neural Information Processing Systems},
    year={2026},
    url={https://arxiv.org/abs/2608.03565}
}
```

## Abstract
While modern tabular learners excel at capturing statistical patterns, they frequently
operate in a semantic vacuum, treating textual features as discrete symbols, ignoring the rich semantics 
inherent in feature names or cell entries. We propose
CASE (Context-Aware Semantic Embeddings), a novel framework that bridges the
gap between the semantic understanding of Large Language Models (LLMs) and
the statistical capabilities of tabular learners. Unlike existing methods that embed
rows in isolation, CASE utilizes a contextualization strategy: we pre-fill the KV
cache of a custom-trained Gemma 3-based Tabular Language Model with a representative sample of rows to establish a 
persistent anchor of the dataset’s semantics.
This ensures that generated row embeddings are dynamically contextualized, resolving semantic ambiguities and 
anchoring representations in domain-specific context.
Our experiments across several benchmarks (CARTE, TextTab, and TabArena)
demonstrate that CASE significantly improve performance of tabular learners –
particularly in low-data regimes and on semantically rich datasets – setting a new
state of the art when combined with recent tabular in-context learners.

![logo](./case_arch.png)

## Installation
Install the base package via [uv](https://docs.astral.sh/uv/getting-started/installation/):
```bash
uv sync
```

### Optional dependencies
You may use CASE with Flash-Attention 2. Depending on your GPU hardware, it may not
be faster than the default SDPA backend. For example, on Hopper (H100) or Blackwell (B200, RTX-6000), the default
SDPA may already be faster than Flash-Attention 2. On Ampere (A100), Flash-Attention 2 may offer speedups over SDPA.

If you want to use the Flash-Attention 2 backend, install additionally (after the first successful sync)
```bash
uv sync --extra flash
```
This will build Flash-Attention 2 from scratch which takes quite a while.
If you want to use more cores and, e.g., only build for A100 for the build to accelerate, use
```bash
uv pip install ninja
FLASH_ATTN_CUDA_ARCHS=80 MAX_JOBS=20 uv sync --extra flash
```
tuning `MAX_JOBS` depending on your system's RAM and setting `FLASH_ATTN_CUDA_ARCHS` corresponding to your GPU.

## Model Checkpoints
To run CASE, you need access to our custom model checkpoints on HuggingFace: https://huggingface.co/SAP/CASE


## Usage Example

To be able to run the examples below, please install
```bash
uv sync --extra examples
```

Our `case` wrapper provides a familiar, `scikit-learn`-compatible transformer interface to extract context-aware semantic embeddings from tabular features, allowing downstream tabular learners to leverage LLM world knowledge.
In the following, we are combining it with TabICLv2 and showcase a regression example, but it will work with any tabular predictor of your choice, for both classification and regression problems:

```python
import numpy as np
import pandas as pd
from tabicl import TabICLRegressor
from case.embedding import CaseTransformer

# --- Training Set ---
# Contains a mix of messy text, semantic job hierarchies, and standard numerical features
X_train = pd.DataFrame({
    "age": [24.0, 30.0, 42.0, 30.0, 31.0, 45.0],
    "job": [
        "Junior Software Engenier", # Contains a typo
        "Senior Data Scientist",    # Base tech track
        "Engineering Manager",      # Management track
        "VP of Product & Strategy",  # Executive tier
        "Sr. Academic Researcher",  # Public/University tier
        "Product Executive",        # Corporate executive tier
    ],
    "years_experience": [1.5, 10.0, 12.0, 10.0, 5.0, 14.0],
    "company_size": ["Startup", "Enterprise", "Mid-market", "Enterprise", "University", "FAANG"],
})

# Target variable: Continuous income baseline
y_train = pd.Series([72000, 130000, 165000, 240000, 85000, 210000], name="income")


# --- Test Set (Out-of-Vocabulary Semantic Shift) ---
# Force identical numerical features, but introduce a clear semantic hierarchy
# completely unseen during training.
X_test = pd.DataFrame({
    "age": [30.0, 30.0, 30.0],
    "job": [
        "Associate Data Scientist",    # Unseen (implied lower salary band)
        "Data Scientist III",         # Unseen alternative naming (implied senior)
        "Principal Data Scientist",  # Unseen (implied leadership premium)
    ],
    "years_experience": [10.0, 10.0, 10.0],
    "company_size": ["Enterprise", "Enterprise", "Enterprise"],
})

# 1. Baseline: Fit/Predict without CASE embeddings
reg_baseline = TabICLRegressor()
reg_baseline.fit(X_train, y_train)
predictions_baseline = reg_baseline.predict(X_test)

# 2. Initialize CASE to generate context-aware semantic embeddings
case_embedder = CaseTransformer()
X_train_enhanced = case_embedder.fit_transform(X_train, y_train)
X_test_enhanced = case_embedder.transform(X_test)

reg_case = TabICLRegressor()
reg_case.fit(X_train_enhanced, y_train)
predictions_case = reg_case.predict(X_test_enhanced)

print("Predictions w/o CASE:", predictions_baseline)
print("Predictions w/  CASE:", predictions_case)

# Predictions depend on the model, context folds, GPU, and attention backend.
```

The example compares a predictor using the original features with one using
additional semantic features. CASE can connect unseen job titles to related
training examples, but this small illustrative dataset does not establish
prediction accuracy or a guaranteed improvement.

## Advanced Usage
If you want to use the CASE embeddings directly for another task, you may extract them as follows:
```python
...
case_embedder = CaseTransformer()
case_embedder.fit(X_train, y_train)
embeddings = case_embedder.embed(X_test)
```

## Further details
Cross-fitting is **optional and disabled by default** (`context_cv=1`). The
default builds context once and embeds each training row once.

For an ablation, enable it with `CaseTransformer(context_cv=5)` (or another value
greater than one). Each training row is then embedded against context sampled
from other folds, excluding its own label, and PCA fits those embeddings. The
fold count is capped at the number of training rows. CASE then builds the final
context cache from the full training set for inference. With `k` folds this
requires `k + 1` context prefills; each training row is still embedded once.
Its runtime and predictive benefit should be measured on your dataset.

When cross-fitting is enabled, use `fit_transform(X_train, y_train)` for training
and `transform(X_test)` for evaluation. Calling `transform(X_train)` or
`embed(X_train)` uses the final context and can expose training labels to their
own rows, so it differs from cross-fitted training features. Keep the entire
transformer inside each evaluation fold or scikit-learn pipeline.

`max_length` is a hard limit on the **complete context plus query**. The transformer
reserves up to `max_cell_tokens` tokens per query cell (default 50), separators,
and the checkpoint's final pooling token before packing context rows. This keeps
context fixed across batches and accommodates future cells up to the same cap.
If the header and reserved query cannot fit, increase `max_length` or reduce
`max_cell_tokens`. Context rows are dropped whole when the remaining budget is
full; oversized fixed sequences raise an error. The configured length must also
fit the model's context window.

Targets can be pandas Series, NumPy arrays, lists, or omitted. They must be
one-dimensional and match the number of training rows. With `y=None`, context
contains feature values with empty targets. With `num_context_rows=0`, the header
is still included. Text and pandas categorical targets keep their exact labels.
For numeric targets, `scientific_notation=True` retains the checkpoint's rounded
scientific representation; set it to `False` for integer class IDs or when exact
numeric spelling is needed. At least two training rows are required for PCA.

Feature names must be unique. Transform restores fitted column order and rejects
missing or extra columns. Output names are strings; non-string input names are
converted, and suffixes resolve collisions with generated `lm_pca_*` names.
`get_feature_names_out()` reports the resulting schema. PCA uses at most the
smaller of the requested component count, training-row count, and embedding
width. Empty inference tables return empty outputs with that schema.

Every fit rebuilds context and PCA. Identical model settings reuse loaded
weights; changing the model, dtype, or attention backend reloads them on the next
fit. Changing fitting parameters requires refitting. `batch_size` and
`append_original_features` may be changed for inference. CASE loads the backbone
without a vocabulary head and retains only final hidden states during inference.

CASE loads only the text backbone, including when the checkpoint stores text
weights inside a wrapper. No vision tower or image projector is constructed.
Use `device="cpu"` or `device="cuda"` to require a particular device; the default
`device="auto"` retains automatic placement. Changing the device requires refitting.
With `attn_implementation=None`, CPU placement uses SDPA. Automatic or CUDA
placement uses Flash Attention 2 when CUDA and `flash-attn` are available,
and SDPA otherwise.


## Tests
The default test suite uses lightweight text models and runs without downloading
the CASE checkpoint. Device-sensitive tests run on CPU and also on CUDA when
available; CUDA cases are skipped when CUDA is unavailable:

```bash
uv run pytest tests/
```

The end-to-end test with the actual CASE checkpoint is marked `checkpoint`
and skipped by default. To run it on CPU and also on CUDA when available:

```bash
uv run pytest --run-checkpoint -s -v tests/test_checkpoint.py
```

## How to obtain support
[Create an issue](https://github.com/SAP-samples/case/issues) in this repository if you find a bug or have questions about the content.

## Contributing
If you wish to contribute code, offer fixes or improvements, please send a pull request. Due to legal reasons, contributors will be asked to accept a DCO when they create the first pull request to this project. This happens in an automated fashion during the submission process. SAP uses [the standard DCO text of the Linux Foundation](https://developercertificate.org/).

## License
Copyright (c) 2026 SAP SE or an SAP affiliate company. All rights reserved. This project is licensed under the Apache Software License, version 2.0 except as noted otherwise in the [LICENSE](LICENSE) file.

For the checkpoint license, please refer to the corresponding [HuggingFace repository](https://huggingface.co/SAP/case).
