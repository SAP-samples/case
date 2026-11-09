from dataclasses import replace

import pandas as pd
import pytest

from case.config import SerializationConfig
from case.core import _clean_and_tokenize_series, serialize_table


def test_format_and_pooling_token(tokenizer):
    X = pd.DataFrame({'job': ['doctor', 'artist']})
    y = pd.Series(['A', 'B'])
    result = serialize_table('label', tokenizer, row=X.iloc[0], retrieval_table=X, retrieval_targets=y)
    assert tokenizer.decode(result['input_ids']) == (
        'job    <bos>label\nartist    <bos>B~\ndoctor    <bos>A~\ndoctor    <bos>'
    )
    assert result['input_ids'][-1] == 2


def test_budget_drops_context_rows(tokenizer):
    X = pd.DataFrame({'job': ['doctor', 'artist']})
    y = pd.Series(['A', 'B'])
    config = SerializationConfig(max_length=1000)
    full = serialize_table('label', tokenizer, row=X.iloc[0], retrieval_table=X, retrieval_targets=y, config=config)
    one = serialize_table(
        'label', tokenizer, row=X.iloc[0], retrieval_table=X.iloc[:1], retrieval_targets=y.iloc[:1], config=config
    )
    budget = len(one['input_ids'])
    assert budget < len(full['input_ids'])
    truncated = serialize_table(
        'label',
        tokenizer,
        row=X.iloc[0],
        retrieval_table=X,
        retrieval_targets=y,
        config=replace(config, max_length=budget),
    )
    assert truncated == one
    assert len(truncated['input_ids']) <= budget


@pytest.mark.parametrize('with_header', [True, False])
def test_oversized_fixed_tokens_raise(tokenizer, with_header):
    with pytest.raises(ValueError, match='exceeding max_length'):
        serialize_table(
            'label',
            tokenizer,
            row=pd.Series({'feature': 'long text'}),
            config=SerializationConfig(max_length=3, with_header=with_header),
        )


def test_unlabeled_context_and_empty_schema_table(tokenizer):
    X = pd.DataFrame({'job': ['doctor', 'artist']})
    context = serialize_table('target', tokenizer, retrieval_table=X)
    header = serialize_table('target', tokenizer, retrieval_table=X.iloc[:0])
    assert tokenizer.decode(header['input_ids']) == 'job    <bos>target\n'
    assert 'artist    <bos>~\n' in tokenizer.decode(context['input_ids'])
    assert len(context['input_ids']) > len(header['input_ids'])


@pytest.mark.parametrize(
    'dtype, values',
    [
        ('Int64', [1, None]),
        ('boolean', [True, None]),
        ('category', ['cat', None]),
        ('string', ['cat', None]),
        ('Float64', [1.5, None]),
    ],
)
def test_nullable_cells(tokenizer, dtype, values):
    result = _clean_and_tokenize_series(pd.Series(values, dtype=dtype), tokenizer, SerializationConfig())
    assert tokenizer.decode(result) == f'{values[0]}        '


def test_scientific_notation_and_literal_header(tokenizer):
    X = pd.DataFrame({'id': ['x', 'y']})
    y = pd.Series(['123456', '123457'])
    normal = serialize_table(
        '123456',
        tokenizer,
        retrieval_table=X,
        retrieval_targets=y,
        config=SerializationConfig(scientific_notation=False),
    )
    scientific = serialize_table('123456', tokenizer, retrieval_table=X, retrieval_targets=y.astype(int))
    plain = tokenizer.decode(normal['input_ids'])
    assert '<bos>123456\n' in plain
    assert '<bos>123456~\n' in plain and '<bos>123457~\n' in plain
    assert '+1.2346e+05' in tokenizer.decode(scientific['input_ids'])
    assert normal != scientific
    assert normal == serialize_table('123456', tokenizer, retrieval_table=X, retrieval_targets=y)
    categorical = pd.Series(pd.Categorical([123456, 123457]))
    assert normal == serialize_table('123456', tokenizer, retrieval_table=X, retrieval_targets=categorical)


def test_schema_and_target_validation(tokenizer):
    X = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    with pytest.raises(ValueError, match='Mismatched shapes'):
        serialize_table('y', tokenizer, retrieval_table=X, retrieval_targets=pd.Series([1]))
    with pytest.raises(ValueError, match='requires a retrieval_table'):
        serialize_table('y', tokenizer, row=X.iloc[0], retrieval_targets=pd.Series([1]))
    with pytest.raises(ValueError, match='unique'):
        serialize_table('y', tokenizer, row=pd.Series([1, 2], index=['a', 'a']))
    with pytest.raises(ValueError, match='unique'):
        serialize_table('y', tokenizer, retrieval_table=X.set_axis(['a', 'a'], axis=1))
    with pytest.raises(ValueError, match='same feature names'):
        serialize_table('y', tokenizer, retrieval_table=X, row=pd.Series({'c': 1}))
    assert serialize_table('y', tokenizer, retrieval_table=X, row=X.iloc[0]) == serialize_table(
        'y',
        tokenizer,
        retrieval_table=X,
        row=X.iloc[0].iloc[::-1],
    )


@pytest.mark.parametrize(
    'kwargs',
    [
        {'max_length': 0},
        {'max_length': True},
        {'max_cell_tokens': -1},
        {'max_cell_tokens': 1.5},
        {'cell_separator': ''},
        {'with_header': 'yes'},
    ],
)
def test_config_validation(kwargs):
    with pytest.raises(ValueError):
        SerializationConfig(**kwargs)
