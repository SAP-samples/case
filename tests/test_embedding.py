import copy
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.utils import get_tags
from sklearn.utils.validation import check_is_fitted

from case import CaseTransformer
from case.core import serialize_table
from case.embedding import _resolve_attn_implementation


def test_native_batch_cache_and_full_sequence_equivalence(make_transformer, data):
    X, y = data
    tr = make_transformer(batch_size=1)
    tr._setup_model()
    prefixes = []

    def capture_prefix(model, args, kwargs):
        if kwargs['past_key_values'].get_seq_length() == 0:
            prefixes.append(kwargs['input_ids'].clone())

    handle = tr._model.register_forward_pre_hook(capture_prefix, with_kwargs=True)
    tr.fit(X, y)
    handle.remove()
    saved = copy.deepcopy(tr._kv_cache)
    single = tr.embed(X)
    tr.set_params(batch_size=3)
    batch = tr.embed(X)
    np.testing.assert_allclose(single, batch, atol=2e-6, rtol=2e-5)
    np.testing.assert_array_equal(batch, tr.embed(X))
    assert tr._kv_cache.get_seq_length() > tr._model.config.get_text_config().sliding_window
    for left, right in zip(saved.layers, tr._kv_cache.layers):
        assert torch.equal(left.keys, right.keys)
        assert torch.equal(left.values, right.values)
    query = serialize_table(
        tr.target_column_name_, tr._tokenizer, row=X.iloc[0], config=tr._serialization_config(with_header=False)
    )['input_ids']
    with torch.inference_mode():
        full = tr._model(torch.cat([prefixes[0], torch.tensor([query])], dim=1), use_cache=False)
    np.testing.assert_allclose(batch[0], full.last_hidden_state[0, -1].numpy(), atol=2e-6, rtol=2e-5)


def test_refit_matches_fresh_and_reuses_only_weights(make_transformer, data):
    X, y = data
    tr = make_transformer().fit(X, y)
    old_model, old_cache = tr._model, tr._kv_cache
    X2 = X.assign(job=['other', 'new', 'role', 'value'])
    tr.fit(X2, y + 500)
    assert tr._model is old_model
    assert tr._kv_cache is not old_cache
    assert make_transformer.loader.call_count == 1
    fresh = make_transformer().fit(X2, y + 500)
    np.testing.assert_allclose(tr.embed(X2), fresh.embed(X2), atol=1e-6)
    np.testing.assert_allclose(
        tr.transform(X2).filter(like='lm_pca'), fresh.transform(X2).filter(like='lm_pca'), atol=1e-6
    )


@pytest.mark.parametrize(
    'params',
    [
        {'model_name': 'different'},
        {'dtype': torch.float64},
        {'attn_implementation': 'eager'},
    ],
)
def test_model_parameter_changes_reload(make_transformer, data, params):
    X, y = data
    tr = make_transformer().fit(X, y)
    old_model = tr._model
    tr.set_params(**params)
    with pytest.raises(NotFittedError):
        tr.transform(X)
    tr.fit(X, y)
    assert make_transformer.loader.call_count == 2
    assert tr._model is not old_model
    assert next(tr._model.parameters()).dtype == tr.dtype


def test_disable_context_retains_header_only(make_transformer, data):
    X, y = data
    tr = make_transformer().fit(X, y)
    old_length = tr._kv_cache.get_seq_length()
    tr.set_params(num_context_rows=0).fit(X, y)
    header = serialize_table('salary', tr._tokenizer, retrieval_table=X.iloc[:0])['input_ids']
    assert tr._kv_cache.get_seq_length() == len(header) < old_length
    renamed = X.rename(columns={'job': 'disease', 'age': 'weight'})
    other = make_transformer(num_context_rows=0).fit(renamed, y)
    assert not np.allclose(tr.embed(X), other.embed(renamed))


def test_unlabeled_context_changes_embeddings(make_transformer, data):
    X, _ = data
    a = make_transformer().fit(X)
    b = make_transformer().fit(X.assign(job=['other', 'new', 'role', 'value']))
    assert not np.allclose(a.embed(X), b.embed(X))


def test_scientific_flag_reaches_actual_serialization(make_transformer, data):
    X, y = data
    a = make_transformer(scientific_notation=True).fit(X, y)
    b = make_transformer(scientific_notation=False).fit(X, y)
    assert not np.allclose(a.embed(X), b.embed(X))
    assert not torch.equal(a._kv_cache.layers[-1].keys, b._kv_cache.layers[-1].keys)


@pytest.mark.parametrize('convert', [lambda y: y.to_numpy(), lambda y: y.tolist(), lambda y: y.rename(None)])
def test_arraylike_targets_and_unnamed_target(make_transformer, data, convert):
    X, y = data
    tr = make_transformer().fit(X, convert(y))
    assert tr.target_column_name_ == 'target'
    assert tr.transform(X).shape == (4, 5)


@pytest.mark.parametrize('target', [[1, 2, 3], [1, 2, 3, 4, 5], np.ones((4, 1)), 3])
def test_bad_targets_rejected_before_loading(make_transformer, data, target):
    with pytest.raises(ValueError):
        make_transformer().fit(data[0], target)
    make_transformer.loader.assert_not_called()


@pytest.mark.parametrize(
    'params',
    [
        {'batch_size': -1},
        {'batch_size': 0},
        {'batch_size': 1.5},
        {'batch_size': True},
        {'num_context_rows': -1},
        {'n_components': 0},
        {'context_cv': 0},
        {'max_length': 0},
        {'max_cell_tokens': 0},
        {'random_state': -1},
        {'random_state': 2**32},
        {'dtype': torch.int64},
        {'normalize_embeddings': 'yes'},
    ],
)
def test_bad_parameters_rejected_before_loading(make_transformer, data, params):
    with pytest.raises(ValueError):
        make_transformer(**params).fit(*data)
    make_transformer.loader.assert_not_called()


def test_changed_batch_size_is_validated(make_transformer, data):
    tr = make_transformer().fit(*data)
    tr.set_params(batch_size=-1)
    with pytest.raises(ValueError, match='batch_size'):
        tr.embed(data[0])


def test_failed_refit_clears_learned_state(make_transformer, data):
    X, y = data
    tr = make_transformer().fit(X, y)
    with patch.object(tr, '_embed', side_effect=RuntimeError('inference failure')):
        with pytest.raises(RuntimeError, match='inference failure'):
            tr.fit(X, y)
    assert tr._kv_cache is None
    assert not hasattr(tr, 'pca_')
    with pytest.raises(NotFittedError):
        check_is_fitted(tr)
    tr.fit(X, y)
    check_is_fitted(tr)


@pytest.mark.parametrize('method', ['transform', 'embed', 'get_feature_names_out'])
def test_fitted_checks(method, data):
    with pytest.raises(NotFittedError):
        getattr(CaseTransformer(), method)(*(() if method == 'get_feature_names_out' else (data[0],)))


def test_schema_reordering_and_name_collisions(make_transformer, data):
    X, y = data
    X = X.rename(columns={'job': 'lm_pca_0'})
    tr = make_transformer().fit(X, y)
    expected = tr.transform(X)
    actual = tr.transform(X.iloc[:, ::-1])
    pd.testing.assert_frame_equal(expected, actual)
    assert expected.columns.is_unique
    assert expected.columns.tolist() == tr.get_feature_names_out().tolist()
    pd.testing.assert_frame_equal(expected.iloc[:, :2], X)
    assert tr.feature_names_in_.tolist() == X.columns.tolist()
    assert tr.n_features_in_ == 2
    for invalid in (X.assign(extra=1), X.iloc[:, :1], X.rename(columns={'age': 'different'})):
        with pytest.raises(ValueError, match='same feature names'):
            tr.transform(invalid)
    tr.set_params(append_original_features=False)
    reduced = tr.transform(X)
    assert reduced.shape == (4, 3)
    assert reduced.columns.tolist() == tr.get_feature_names_out().tolist()


def test_duplicate_columns_rejected_before_loading(make_transformer, data):
    X, y = data
    with pytest.raises(ValueError, match='unique'):
        make_transformer().fit(X.set_axis(['x', 'x'], axis=1), y)
    make_transformer.loader.assert_not_called()


def test_arrays_nonstring_names_and_empty_transform(make_transformer, data):
    X, y = data
    tr = make_transformer().fit(X.to_numpy(), y)
    output = tr.transform(X.to_numpy())
    assert all(isinstance(name, str) for name in output.columns)
    assert output.shape == (4, 5)
    assert tr.embed(X.to_numpy()[:0]).shape == (0, 32)
    assert tr.transform(X.to_numpy()[:0]).shape == (0, 5)
    # Literal stringification collisions must also be resolved.
    named = X.set_axis([0, '0'], axis=1)
    assert make_transformer().fit_transform(named, y).columns.is_unique


def test_total_token_budget_and_long_future_queries(make_transformer, data):
    X, y = data
    tr = make_transformer(max_length=100, max_cell_tokens=4).fit(X, y)
    observed = []

    def check_length(model, args, kwargs):
        length = kwargs['past_key_values'].get_seq_length() + kwargs['input_ids'].shape[1]
        observed.append(length)
        assert length <= 100

    h = tr._model.register_forward_pre_hook(check_length, with_kwargs=True)
    tr.transform(X.assign(job=['a' * 100] * len(X), age=['b' * 100] * len(X)))
    h.remove()
    assert observed
    assert tr._kv_cache.get_seq_length() + tr._query_budget_ <= tr.max_length
    with pytest.raises(ValueError, match='max_length'):
        make_transformer(max_length=20).fit(X, y)
    with pytest.raises(ValueError, match='model context limit'):
        make_transformer(max_length=20000).fit(X, y)


def test_only_backbone_output_and_compact_cache(make_transformer, data):
    tr = make_transformer()
    tr._setup_model()
    seen = []

    def check_outputs(model, args, kwargs, outputs):
        assert kwargs['output_hidden_states'] is False
        assert outputs.hidden_states is None
        assert outputs.attentions is None
        assert not hasattr(outputs, 'logits')
        assert outputs.last_hidden_state.shape[-1] == 32
        seen.append(True)

    h = tr._model.register_forward_hook(check_outputs, with_kwargs=True)
    tr.fit_transform(*data)
    tr.embed(data[0])
    h.remove()
    assert len(seen) >= 3
    for layer in tr._kv_cache.layers:
        for value in (layer.keys, layer.values):
            assert value.untyped_storage().nbytes() == value.numel() * value.element_size()


def test_cross_fitting_excludes_each_query_fold_and_embeds_once(make_transformer, data):
    X, y = data
    tr = make_transformer(context_cv=3)
    current_context = set()
    embedded = []
    prefill, embed = tr._prefill_kv, tr._embed

    def track_prefill(frame, targets=None):
        nonlocal current_context
        current_context = set(frame.index)
        # Verify that feature/label positional alignment survives arbitrary indices.
        np.testing.assert_array_equal(targets.to_numpy(), y.loc[frame.index].to_numpy())
        return prefill(frame, targets)

    def track_embed(frame):
        assert current_context.isdisjoint(frame.index)
        embedded.extend(frame.index)
        return embed(frame)

    with (
        patch.object(tr, '_prefill_kv', side_effect=track_prefill),
        patch.object(tr, '_embed', side_effect=track_embed),
    ):
        output = tr.fit_transform(X, y)
    assert sorted(embedded) == sorted(X.index)
    assert len(embedded) == len(X)
    assert current_context == set(X.index)  # Full context is rebuilt for later inference.
    assert output.shape == (4, 5)
    other = make_transformer(context_cv=3).fit_transform(X, y)
    pd.testing.assert_frame_equal(output, other)


def test_pca_normalization_and_fit_transform_without_cross_fitting(make_transformer, data):
    X, y = data
    tr = make_transformer(normalize_embeddings=True, n_components=100)
    assert tr.context_cv == 1
    with patch.object(tr, '_prefill_kv', wraps=tr._prefill_kv) as prefill:
        output = tr.fit_transform(X, y)
    prefill.assert_called_once()
    assert output.shape == (4, 6)
    assert tr.pca_.n_components_ == 4
    np.testing.assert_allclose(np.linalg.norm(tr.embed(X), axis=1), 1, atol=1e-6)
    pd.testing.assert_frame_equal(output, tr.transform(X), atol=2e-6, rtol=2e-5)


def test_sklearn_clone_tags_pipeline_and_set_output(make_transformer, data):
    X, y = data
    tr = make_transformer(context_cv=2, append_original_features=False)
    assert get_tags(tr).transformer_tags is not None
    cloned = clone(tr)
    assert cloned.get_params() == tr.get_params()
    pipeline = make_pipeline(cloned.set_output(transform='pandas'), Ridge())
    pipeline.fit(X, y.to_numpy())
    assert pipeline.predict(X).shape == (len(X),)


def test_cpu_attention_selection():
    with patch('torch.cuda.is_available', return_value=False), patch('importlib.util.find_spec', return_value=object()):
        assert _resolve_attn_implementation() == 'sdpa'


def test_loading_saved_language_model_as_backbone(tmp_path, monkeypatch, tokenizer, data):
    from transformers import (
        AutoModel,
        Gemma3Config,
        Gemma3ForConditionalGeneration,
        Gemma3TextConfig,
        SiglipVisionConfig,
    )

    torch.manual_seed(42)
    text = Gemma3TextConfig(
        vocab_size=256,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        sliding_window=8,
        max_position_embeddings=16384,
        query_pre_attn_scalar=8,
    )
    vision = SiglipVisionConfig(
        hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2, image_size=8, patch_size=4
    )
    config = Gemma3Config(
        text_config=text,
        vision_config=vision,
        mm_tokens_per_image=4,
        image_token_index=255,
        boi_token_index=253,
        eoi_token_index=254,
    )
    config._attn_implementation = 'sdpa'
    source = Gemma3ForConditionalGeneration(config).eval()
    source.save_pretrained(tmp_path)
    # Only tokenizer construction is replaced; actual AutoModel loading and weight mapping run.
    monkeypatch.setattr('case.embedding.AutoTokenizer.from_pretrained', lambda *args, **kwargs: tokenizer)
    tr = CaseTransformer(
        model_name=str(tmp_path),
        dtype=torch.float32,
        max_length=512,
        max_cell_tokens=8,
        context_cv=2,
        n_components=3,
        attn_implementation='sdpa',
    )
    tr.fit_transform(*data)
    assert isinstance(tr._model, AutoModel._model_mapping[type(config)])
    for key, weight in source.model.state_dict().items():
        torch.testing.assert_close(tr._model.state_dict()[key], weight, rtol=0, atol=0)
    assert not hasattr(tr._model, 'lm_head')


def test_direct_parameter_mutation_requires_refit(make_transformer, data):
    tr = make_transformer().fit(*data)
    tr.normalize_embeddings = True
    with pytest.raises(ValueError, match='Fitting parameters changed'):
        tr.embed(data[0])
    tr.fit(*data)
    np.testing.assert_allclose(np.linalg.norm(tr.embed(data[0]), axis=1), 1, atol=1e-6)


@pytest.mark.parametrize(
    'values, dtype',
    [
        ([1, None, 3, 4], 'Int64'),
        ([True, None, False, True], 'boolean'),
        (['cat', None, 'dog', 'cat'], 'category'),
    ],
)
def test_nullable_dataframes(make_transformer, data, values, dtype):
    frame = pd.DataFrame({'x': pd.Series(values, dtype=dtype)})
    transformed = make_transformer().fit_transform(frame, data[1].to_numpy())
    assert transformed.iloc[:, 1:].notna().all().all()
    pd.testing.assert_series_equal(transformed['x'], frame['x'])


def test_numeric_looking_class_labels_are_distinct(make_transformer, data):
    X, _ = data
    y = pd.Series(['123456', '123457', '123458', '123459'], name='class')
    tr = make_transformer().fit(X, y)
    original = tr._tokenizer.decode(
        serialize_table(
            'class',
            tr._tokenizer,
            retrieval_table=X,
            retrieval_targets=y,
            config=tr._serialization_config(with_header=True),
        )['input_ids']
    )
    for value in y:
        assert f'<bos>{value}~' in original
