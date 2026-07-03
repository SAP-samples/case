from typing import Any

import pandas as pd
import pytest

from case.config import SerializationConfig
from case.core import serialize_table


# A minimal mock tokenizer to keep unit tests lightning fast and offline-friendly
class DummyTokenizer:
    def __init__(self):
        # Add the missing attributes expected by the serialization logic
        self.eos_token = '</s>'
        self.pad_token = '<pad>'
        self.bos_token = '<s>'

    def __call__(self, text: Any, add_special_tokens: bool = False, **kwargs):
        # Handle batch input (list of strings)
        if isinstance(text, list):
            batch_ids = []
            for item in text:
                words = str(item).split()
                tokens = [len(word) if word else 1 for word in words]
                batch_ids.append(tokens)
            return {'input_ids': batch_ids}

        # Handle single string input
        words = str(text).split()
        tokens = [len(word) if word else 1 for word in words]
        return {'input_ids': tokens}


@pytest.fixture
def dummy_tokenizer():
    return DummyTokenizer()


@pytest.fixture
def sample_data():
    row = pd.Series({'age': '30', 'score': '95.5'})
    retrieval_table = pd.DataFrame([{'age': '25', 'score': '88.0'}, {'age': '40', 'score': '100.0'}])
    retrieval_targets = pd.Series(['A', 'B'])
    return row, retrieval_table, retrieval_targets


def test_serialize_table_basic(dummy_tokenizer, sample_data):
    """Ensure table serialization runs and produces expected dictionary structure."""
    row, retrieval_table, retrieval_targets = sample_data
    config = SerializationConfig(max_length=100, with_header=True)

    result = serialize_table(
        target_column='score',
        tokenizer=dummy_tokenizer,
        row=row,
        retrieval_table=retrieval_table,
        retrieval_targets=retrieval_targets,
        config=config,
    )

    assert 'input_ids' in result
    assert isinstance(result['input_ids'], list)
    assert all(isinstance(i, int) for i in result['input_ids'])


def test_serialize_table_budget_truncation(dummy_tokenizer, sample_data):
    """Verify that retrieval rows are dropped if they exceed max_length budget."""
    row, retrieval_table, retrieval_targets = sample_data

    # Capture the unconstrained baseline length
    large_config = SerializationConfig(max_length=1000, with_header=True)
    full_result = serialize_table(
        target_column='score',
        tokenizer=dummy_tokenizer,
        row=row,
        retrieval_table=retrieval_table,
        retrieval_targets=retrieval_targets,
        config=large_config,
    )
    full_length = len(full_result['input_ids'])

    # Assign an isolated budget target that allows the primary row block through,
    # but strictly triggers context historical truncations.
    strict_budget = max(15, full_length - 5)
    strict_config = SerializationConfig(max_length=strict_budget, with_header=True)

    truncated_result = serialize_table(
        target_column='score',
        tokenizer=dummy_tokenizer,
        row=row,
        retrieval_table=retrieval_table,
        retrieval_targets=retrieval_targets,
        config=strict_config,
    )

    # Enforce safe upper bounds
    assert len(truncated_result['input_ids']) <= strict_budget


def test_serialize_table_mismatched_inputs(dummy_tokenizer, sample_data):
    """Enforce fast-failing input validations."""
    row, retrieval_table, _ = sample_data
    config = SerializationConfig(max_length=100)

    invalid_targets = pd.Series(['OnlyOneTarget'])  # Length mismatch with retrieval_table (2 rows)

    with pytest.raises(ValueError, match='Mismatched shapes'):
        serialize_table(
            target_column='score',
            tokenizer=dummy_tokenizer,
            row=row,
            retrieval_table=retrieval_table,
            retrieval_targets=invalid_targets,
            config=config,
        )
