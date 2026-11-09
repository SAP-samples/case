"""End-to-end test using the actual SAP/CASE weights and tokenizer on CPU and CUDA."""

import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge

from case import CaseTransformer


def test_case_checkpoint_end_to_end(torch_device):
    """Fit CASE and a downstream regressor, then check real held-out embeddings."""
    use_cuda = torch_device == 'cuda'
    dtype = torch.bfloat16 if use_cuda and torch.cuda.is_bf16_supported() else torch.float32
    previous_threads = torch.get_num_threads()
    if not use_cuda:
        torch.set_num_threads(min(8, os.cpu_count() or 1))

    X_train = pd.DataFrame(
        {
            'job': ['Junior Engineer', 'Senior Engineer', 'Engineering Manager', 'Research Scientist'],
            'age': [24, 35, 42, 31],
        },
        index=['train_a', 'train_b', 'train_c', 'train_d'],
    )
    y_train = pd.Series([60_000.0, 110_000.0, 150_000.0, 95_000.0], index=X_train.index, name='income')
    X_test = pd.DataFrame(
        {'job': ['Engineer', 'Principal Research Scientist'], 'age': [30, 40]},
        index=['test_a', 'test_b'],
    )
    transformer = CaseTransformer(
        model_name=os.environ.get('CASE_CHECKPOINT_MODEL', 'SAP/CASE'),
        dtype=dtype,
        attn_implementation='sdpa',
        max_length=2048,
        num_context_rows=4,
        n_components=3,
        batch_size=2,
        context_cv=1,
        append_original_features=False,
        device=torch_device,
    )
    try:
        started = time.perf_counter()
        print(f'Loading and fitting SAP/CASE with {dtype}; CUDA={use_cuda}.', flush=True)
        train_features = transformer.fit_transform(X_train, y_train)
        fit_seconds = time.perf_counter() - started
        config = transformer._model.config
        hidden_size = config.get_text_config().hidden_size
        assert config.model_type == 'gemma3_text'
        assert all(parameter.device.type == torch_device for parameter in transformer._model.parameters())
        assert transformer._device.type == torch_device
        assert not any('vision' in name or 'multi_modal' in name for name, _ in transformer._model.named_modules())
        assert hidden_size == 3840
        assert train_features.shape == (len(X_train), 3)
        assert train_features.index.equals(X_train.index)
        assert np.isfinite(train_features.to_numpy()).all()
        cache_length = transformer._kv_cache.get_seq_length()
        assert cache_length + transformer._query_budget_ <= transformer.max_length
        print(f'Fit completed in {fit_seconds:.1f}s; context={cache_length} tokens.', flush=True)

        raw = transformer.embed(X_test)
        test_features = transformer.transform(X_test)
        assert raw.shape == (len(X_test), hidden_size)
        assert np.isfinite(raw).all()
        assert not np.allclose(raw[0], raw[1])
        assert test_features.shape == (len(X_test), 3)
        assert test_features.index.equals(X_test.index)
        assert test_features.columns.tolist() == transformer.get_feature_names_out().tolist()
        np.testing.assert_allclose(test_features.to_numpy(), transformer.pca_.transform(raw), rtol=1e-4, atol=1e-4)

        regressor = Ridge(alpha=1.0).fit(train_features, y_train)
        predictions = regressor.predict(test_features)
        assert predictions.shape == (len(X_test),)
        assert np.isfinite(predictions).all()

        transformer.set_params(batch_size=1)
        individual = transformer.embed(X_test)
        # BF16 kernels can round differently for different batch shapes.
        cosine = np.sum(raw * individual, axis=1) / (np.linalg.norm(raw, axis=1) * np.linalg.norm(individual, axis=1))
        assert np.all(cosine > 0.999), f'Batch/single-row cosine similarities: {cosine}'
        assert transformer._kv_cache.get_seq_length() == cache_length
        print(
            f'End-to-end passed in {time.perf_counter() - started:.1f}s; '
            f'batch cosine={cosine.tolist()}; predictions={predictions.tolist()}.',
            flush=True,
        )
    finally:
        torch.set_num_threads(previous_threads)
