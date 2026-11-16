"""Load real, tiny text checkpoints without their multimodal wrappers or output heads."""

import copy
import json
import pickle
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from accelerate.hooks import AlignDevicesHook, SequentialHook
from safetensors.torch import save_file
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.utils._pytree import tree_leaves
from transformers import (
    AutoModelForCausalLM,
    Gemma4TextConfig,
    GPT2Config,
    LlamaConfig,
    Qwen2Config,
    Qwen3_5TextConfig,
    Qwen3Config,
)

from case import CaseTransformer
from case.core import serialize_table


def _text_config(model_type):
    settings = dict(
        vocab_size=256,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=1,
    )
    if model_type.startswith('gemma4'):
        settings.update(
            head_dim=8,
            global_head_dim=8,
            sliding_window=8,
            layer_types=['sliding_attention', 'full_attention'],
            hidden_size_per_layer_input=4,
            vocab_size_per_layer_input=256,
            use_bidirectional_attention='vision',
        )
        if model_type == 'gemma4_shared':
            settings.update(
                num_hidden_layers=6,
                layer_types=['sliding_attention', 'full_attention'] * 3,
                num_kv_shared_layers=4,
                use_bidirectional_attention=None,
            )
        elif model_type == 'gemma4_moe':
            settings.update(
                attention_k_eq_v=True,
                global_head_dim=16,
                num_global_key_value_heads=1,
                hidden_size_per_layer_input=0,
                enable_moe_block=True,
                num_experts=4,
                top_k_experts=2,
                moe_intermediate_size=32,
            )
        config_class = Gemma4TextConfig
    else:
        config_class = {
            'gpt2': GPT2Config,
            'llama': LlamaConfig,
            'qwen2': Qwen2Config,
            'qwen3': Qwen3Config,
            'qwen3_5': Qwen3_5TextConfig,
        }[model_type]
        if model_type == 'gpt2':
            settings.update(n_positions=512)
        if model_type in ('qwen3', 'qwen3_5'):
            settings.update(head_dim=8)
        if model_type == 'qwen3_5':
            settings.update(
                linear_key_head_dim=8,
                linear_value_head_dim=8,
                linear_num_key_heads=2,
                linear_num_value_heads=2,
                layer_types=['linear_attention', 'full_attention'],
            )
    config = config_class(**settings)
    config._attn_implementation = 'sdpa'
    return config


def _save_checkpoint(path, model_type, prefix, wrapper):
    torch.manual_seed(42)
    source = AutoModelForCausalLM.from_config(_text_config(model_type)).eval()
    source.save_pretrained(path)
    if wrapper is not None:
        (path / 'config.json').write_text(json.dumps({'model_type': wrapper, 'text_config': source.config.to_dict()}))
        weights = {f'{prefix}{key}': value for key, value in source.base_model.state_dict().items()}
        # Auxiliary weights must not cause the image/audio towers to be constructed.
        weights['model.vision_tower.unused.weight'] = torch.zeros(2, 2)
        weights['model.audio_tower.unused.weight'] = torch.zeros(2, 2)
        save_file(weights, path / 'model.safetensors')
    return source


@pytest.mark.parametrize(
    'model_type, prefix, wrapper',
    [
        ('llama', '', None),
        ('gpt2', '', None),
        ('qwen2', '', None),
        ('qwen3', '', None),
        ('qwen3_5', '', None),
        ('qwen3_5', 'model.language_model.', 'qwen3_5'),
        ('gemma4', '', None),
        ('gemma4_shared', 'model.language_model.', 'gemma4'),
        ('gemma4_moe', 'model.language_model.', 'gemma4'),
        ('gemma4', 'model.language_model.', 'gemma4'),
        ('gemma4', 'language_model.model.', 'gemma4'),
        ('gemma4', 'language_model.', 'gemma4'),
        ('qwen2', 'model.language_model.', 'llava'),
        ('qwen2', 'model.text_model.', 'llava'),
    ],
)
def test_text_checkpoint_weights_and_cached_embeddings(
    tmp_path, monkeypatch, tokenizer, data, model_type, prefix, wrapper
):
    source = _save_checkpoint(tmp_path, model_type, prefix, wrapper)
    monkeypatch.setattr('case.embedding.AutoTokenizer.from_pretrained', lambda *args, **kwargs: tokenizer)
    transformer = CaseTransformer(
        model_name=str(tmp_path),
        dtype=torch.float32,
        max_length=512,
        max_cell_tokens=8,
        n_components=3,
        batch_size=1,
        device='cpu',
        attn_implementation='sdpa',
    )
    transformer._setup_model()
    loaded = transformer._model
    assert loaded.config.model_type == source.base_model.config.model_type
    assert not hasattr(loaded, 'lm_head')
    assert not any('vision' in name or 'audio' in name for name, _ in loaded.named_modules())
    assert loaded.state_dict().keys() == source.base_model.state_dict().keys()
    for key, expected in source.base_model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], expected, rtol=0, atol=0)

    prefixes = []

    def capture_context(model, args, kwargs):
        if kwargs['past_key_values'].get_seq_length() == 0:
            prefixes.append(kwargs['input_ids'].clone())

    handle = loaded.register_forward_pre_hook(capture_context, with_kwargs=True)
    X, y = data
    transformer.fit(X, y)
    handle.remove()
    raw = transformer.embed(X)
    context_length = transformer._kv_cache.get_seq_length()
    transformer.set_params(batch_size=3)
    np.testing.assert_allclose(raw, transformer.embed(X), atol=2e-6, rtol=2e-5)
    assert transformer._kv_cache.get_seq_length() == context_length
    query = serialize_table(
        y.name, tokenizer, row=X.iloc[0], config=transformer._serialization_config(with_header=False)
    )['input_ids']
    with torch.inference_mode():
        expected = source.base_model(torch.cat([prefixes[0], torch.tensor([query])], dim=1), use_cache=False)
    np.testing.assert_allclose(raw[0], expected.last_hidden_state[0, -1].numpy(), atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize('mode', [True, 'all', 'unknown'])
def test_bidirectional_text_rejected_before_loading_weights(monkeypatch, mode):
    if mode == 'all':
        config = Gemma4TextConfig.from_dict({**_text_config('gemma4').to_dict(), 'use_bidirectional_attention': mode})
    else:
        config = _text_config('llama')
        config.use_bidirectional_attention = mode
    monkeypatch.setattr('case.embedding.AutoConfig.from_pretrained', lambda *args, **kwargs: config)
    loader = Mock()
    monkeypatch.setattr('case.embedding.AutoModel.from_pretrained', loader)
    transformer = CaseTransformer(device='cpu')
    with pytest.raises(ValueError, match='causal text model'):
        transformer._setup_model()
    loader.assert_not_called()
    assert transformer._model is None


@pytest.mark.parametrize('flag', ['is_causal', 'is_encoder_decoder'])
def test_noncausal_configuration_rejected_before_loading_weights(monkeypatch, flag):
    config = _text_config('llama')
    setattr(config, flag, flag == 'is_encoder_decoder')
    monkeypatch.setattr('case.embedding.AutoConfig.from_pretrained', lambda *args, **kwargs: config)
    loader = Mock()
    monkeypatch.setattr('case.embedding.AutoModel.from_pretrained', loader)
    with pytest.raises(ValueError, match='causal text model'):
        CaseTransformer(device='cpu')._setup_model()
    loader.assert_not_called()


@pytest.mark.parametrize('problem', ['missing_weight', 'unknown_prefix'])
def test_incomplete_text_weights_rejected(tmp_path, monkeypatch, tokenizer, data, problem):
    source = _save_checkpoint(tmp_path, 'gemma4', 'model.language_model.', 'gemma4')
    prefix = 'unsupported_decoder.' if problem == 'unknown_prefix' else 'model.language_model.'
    weights = {f'{prefix}{key}': value for key, value in source.model.state_dict().items()}
    if problem == 'missing_weight':
        del weights[f'{prefix}norm.weight']
    save_file(weights, tmp_path / 'model.safetensors')
    monkeypatch.setattr('case.embedding.AutoTokenizer.from_pretrained', lambda *args, **kwargs: tokenizer)
    transformer = CaseTransformer(
        model_name=str(tmp_path), dtype=torch.float32, attn_implementation='sdpa', device='cpu'
    )
    with pytest.raises(ValueError, match='Text backbone weights were not loaded completely'):
        transformer.fit(*data)
    assert transformer._model is None
    assert transformer._kv_cache is None
    assert not transformer.__sklearn_is_fitted__()


@pytest.mark.parametrize('model_type', ['llama', 'qwen2', 'qwen3_5', 'gemma4', 'gemma4_shared', 'gemma4_moe'])
def test_long_queries_warn_and_keep_native_context(tmp_path, monkeypatch, tokenizer, data, caplog, model_type):
    wrapper = 'gemma4' if model_type.startswith('gemma4') else None
    source = _save_checkpoint(tmp_path, model_type, 'model.language_model.', wrapper)
    monkeypatch.setattr('case.embedding.AutoTokenizer.from_pretrained', lambda *args, **kwargs: tokenizer)
    transformer = CaseTransformer(
        model_name=str(tmp_path),
        dtype=torch.float32,
        max_length=143,
        max_cell_tokens=8,
        n_components=3,
        device='cpu',
        attn_implementation='sdpa',
    )
    transformer._setup_model()
    prefixes = []

    def capture_context(model, args, kwargs):
        if kwargs['past_key_values'].get_seq_length() == 0:
            prefixes.append(kwargs['input_ids'].clone())

    handle = transformer._model.register_forward_pre_hook(capture_context, with_kwargs=True)
    transformer.fit(*data)
    X, y = data
    future = X.iloc[:1].assign(job='x' * 8, age='y' * 8)
    saved_length = transformer._kv_cache.get_seq_length()
    query = serialize_table(
        y.name, tokenizer, row=future.iloc[0], config=transformer._serialization_config(with_header=False)
    )['input_ids']
    assert saved_length + len(query) > transformer.max_length
    raw = transformer.embed(future)
    handle.remove()
    assert len(prefixes) == 1
    assert 'exceeding max_length=143' in caplog.text
    inputs = torch.cat([prefixes[0], torch.tensor([query])], dim=1)
    assert inputs.shape[1] > transformer.max_length
    with torch.inference_mode():
        expected = source.base_model(inputs, use_cache=False)
    np.testing.assert_allclose(raw[0], expected.last_hidden_state[0, -1].numpy(), atol=2e-6, rtol=2e-5)
    assert transformer._kv_cache.get_seq_length() == saved_length


@pytest.mark.parametrize('model_type', ['gemma4_shared', 'qwen3_5'])
def test_hybrid_cache_state_isolated_and_restored(tmp_path, monkeypatch, tokenizer, data, model_type):
    wrapper = 'gemma4' if model_type == 'gemma4_shared' else 'qwen3_5'
    _save_checkpoint(tmp_path, model_type, 'model.language_model.', wrapper)
    monkeypatch.setattr('case.embedding.AutoTokenizer.from_pretrained', lambda *args, **kwargs: tokenizer)
    transformer = CaseTransformer(
        model_name=str(tmp_path),
        dtype=torch.float32,
        max_length=512,
        max_cell_tokens=8,
        n_components=3,
        batch_size=3,
        device='cpu',
        attn_implementation='sdpa',
    ).fit(*data)
    saved = copy.deepcopy(transformer._kv_cache)
    X, _ = data
    expected = transformer.embed(X)

    def tensors(cache):
        return [
            value
            for item in [cache, *cache.layers]
            for value in tree_leaves(vars(item))
            if isinstance(value, torch.Tensor)
        ]

    original_tensors, current_tensors = tensors(saved), tensors(transformer._kv_cache)
    assert original_tensors
    assert len(original_tensors) == len(current_tensors)
    for original, current in zip(original_tensors, current_tensors):
        torch.testing.assert_close(original, current, rtol=0, atol=0)

    restored = pickle.loads(pickle.dumps(transformer))
    assert restored._model is None
    np.testing.assert_allclose(restored.embed(X), expected, atol=2e-6, rtol=2e-5)

    # Check nested recurrent-state movement without requiring a CUDA device.
    moved = transformer._move_cache(copy.deepcopy(saved), 'meta')
    assert all(value.device.type == 'meta' for value in tensors(moved))
    for layer in moved.layers:
        if isinstance(getattr(layer, 'device', None), torch.device):
            assert layer.device.type == 'meta'


@pytest.mark.parametrize('model_type', ['gpt2', 'gemma4_shared', 'qwen3_5'])
@pytest.mark.parametrize('placement', ['parameters', 'offload_hook'])
def test_pickled_cache_restores_layer_execution_devices(tmp_path, monkeypatch, tokenizer, data, model_type, placement):
    source = _save_checkpoint(tmp_path, model_type, '', None)
    monkeypatch.setattr('case.embedding.AutoTokenizer.from_pretrained', lambda *args, **kwargs: tokenizer)
    # Start with automatic CPU placement before simulating the reloaded GPU layout.
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 0)
    transformer = CaseTransformer(
        model_name=str(tmp_path),
        dtype=torch.float32,
        max_length=512,
        max_cell_tokens=8,
        n_components=3,
        device='auto',
        attn_implementation='sdpa',
    ).fit(*data)
    restored = pickle.loads(pickle.dumps(transformer))
    assert restored._model is None
    model = source.base_model
    blocks = model.h if model_type == 'gpt2' else model.layers

    def reload_model():
        restored._model = model
        restored._device = torch.device('cuda:0')
        restored._tokenizer = tokenizer

    monkeypatch.setattr(restored, '_setup_model', reload_model)
    # Pickle real CPU state first: FakeTensorMode then exercises distinct GPU
    # placements and native cache updates without requiring CUDA hardware.
    with FakeTensorMode(allow_non_fake_inputs=True):
        for index, block in enumerate(blocks):
            device = torch.device('cuda', (index + 1) % 2)
            if placement == 'parameters':
                block.to(device)
            else:
                block.to('meta')
                block._hf_hook = SequentialHook(AlignDevicesHook(execution_device=device, offload=True))
        restored._ensure_runtime()
        cache = restored._kv_cache
        for index, layer in enumerate(cache.layers):
            device = torch.device('cuda', (index + 1) % 2)
            if getattr(layer, 'keys', None) is not None:
                states = torch.zeros((*layer.keys.shape[:2], 1, layer.keys.shape[-1]), device=device)
                keys, values = cache.update(states, states, index)
                assert keys.device == values.device == device
            else:
                recurrent = cache.get_recurrent_state(index)
                assert recurrent is not None
                assert recurrent.device == device
                cache.update_recurrent_state(torch.zeros_like(recurrent, device=device), index)
            tensors = [value for value in tree_leaves(vars(layer)) if isinstance(value, torch.Tensor)]
            assert tensors
            assert all(value.device == device for value in tensors)
