import copy
from unittest.mock import Mock

import pandas as pd
import pytest
import torch
from transformers import Gemma3TextConfig, Gemma3TextModel

from case import CaseTransformer


class CharacterTokenizer:
    """Offline character tokenizer with BOS, separators, and real tensor padding."""

    bos_token = '<bos>'
    eos_token = '~'
    pad_token = '<pad>'
    padding_side = 'right'

    def __call__(self, text, add_special_tokens=True, **kwargs):
        def encode(value):
            return ([2] if add_special_tokens else []) + [3 + ord(char) for char in str(value)]

        return {'input_ids': [encode(value) for value in text] if isinstance(text, list) else encode(text)}

    def decode(self, tokens):
        return ''.join('<bos>' if token == 2 else '<pad>' if token == 0 else chr(token - 3) for token in tokens)

    def pad(self, inputs, **kwargs):
        rows = inputs['input_ids']
        width = max(map(len, rows))
        return {
            'input_ids': torch.tensor([row + [0] * (width - len(row)) for row in rows]),
            'attention_mask': torch.tensor([[1] * len(row) + [0] * (width - len(row)) for row in rows]),
        }


@pytest.fixture
def tokenizer():
    return CharacterTokenizer()


@pytest.fixture
def data():
    X = pd.DataFrame({'job': ['a', 'bbb', 'cc', 'dddd'], 'age': [10, 20, 30, 40]}, index=[8, 2, 7, 4])
    y = pd.Series([100, 200, 300, 400], name='salary', index=X.index)
    return X, y


@pytest.fixture(scope='session', autouse=True)
def torch_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def make_transformer(monkeypatch):
    text_config = Gemma3TextConfig(
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
        attn_implementation='sdpa',
    )

    def load_model(name, **kwargs):
        # Build one text model only when loading is actually requested.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            model = Gemma3TextModel(copy.deepcopy(text_config)).eval()
        model.to(dtype=kwargs['dtype'])
        model.set_attn_implementation(kwargs['attn_implementation'])
        return model

    loader = Mock(side_effect=load_model)
    tokenizer_loader = Mock(side_effect=lambda *args, **kwargs: CharacterTokenizer())
    monkeypatch.setattr('case.embedding.AutoModel.from_pretrained', loader)
    monkeypatch.setattr('case.embedding.AutoTokenizer.from_pretrained', tokenizer_loader)

    def make(**kwargs):
        params = dict(
            model_name='tiny-text',
            dtype=torch.float32,
            attn_implementation='sdpa',
            max_length=512,
            max_cell_tokens=8,
            n_components=3,
        )
        params.update(kwargs)
        return CaseTransformer(**params)

    make.loader = loader
    make.tokenizer_loader = tokenizer_loader
    return make
