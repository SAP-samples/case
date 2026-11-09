from typing import Any, Dict, List

import numpy as np
import pandas as pd

from case.config import SerializationConfig


def _clean_and_tokenize_series(series: pd.Series, tokenizer: Any, config: SerializationConfig) -> List[int]:
    """Cleans an entire pandas Series and serializes it into a flat list of token IDs."""
    # Vectorized cleaning across the Series
    cleaned_strings: List[str] = (
        series.astype(object).where(series.notna(), '').astype(str).str.replace(r'[\t\n\r]|    ', '', regex=True)
    ).tolist()

    if not cleaned_strings:
        return []

    # Vectorized Batch Tokenization
    batch_outputs = tokenizer(cleaned_strings, add_special_tokens=False)['input_ids']
    separator_tokens = tokenizer(config.cell_separator, add_special_tokens=False)['input_ids']

    # Efficient Flattening with Constraints
    token_ids: List[int] = []
    max_tokens = config.max_cell_tokens

    for cell_tokens in batch_outputs:
        token_ids.extend(cell_tokens[:max_tokens])
        token_ids.extend(separator_tokens)

    return token_ids


def _format_target_value(value: Any, config: SerializationConfig, eos_token: str | None = None) -> str:
    """Formats a target value into scientific notation or standard string format."""
    eos = eos_token or ''
    if pd.isna(value):
        return f'{eos}\n'

    # Text labels retain their identity even when float(value) would succeed.
    if config.scientific_notation and not isinstance(value, str):
        try:
            f_value = float(value)
            formatted = np.format_float_scientific(f_value, sign=True, precision=4, min_digits=4)
            return f'{formatted}{eos}\n'
        except (ValueError, TypeError):
            pass  # Fallback to standard string representation if float cast fails

    return f'{str(value)}{eos}\n'


def serialize_table(
    target_column: str,
    tokenizer: Any,
    row: pd.Series | None = None,
    retrieval_table: pd.DataFrame | None = None,
    retrieval_targets: pd.Series | None = None,
    config: SerializationConfig | None = None,
) -> Dict[str, List[int]]:
    """Serializes a primary row and historical retrieval rows into a single, flat
    sequence of token IDs, bounded tightly by a maximum token budget.
    """
    # Fallback to default configuration if none provided
    if config is None:
        config = SerializationConfig()

    # Validate the full schema before selecting cells by label.
    if retrieval_targets is not None and retrieval_table is None:
        raise ValueError('retrieval_targets requires a retrieval_table.')
    if retrieval_table is not None and not retrieval_table.columns.is_unique:
        raise ValueError('Feature names must be unique.')
    if row is not None and not row.index.is_unique:
        raise ValueError('Feature names must be unique.')
    if row is not None and retrieval_table is not None:
        columns = retrieval_table.columns
        if len(row.index) != len(columns) or not columns.isin(row.index).all():
            raise ValueError('row and retrieval_table must have the same feature names.')
        row = row.reindex(columns)
    if retrieval_table is not None and retrieval_targets is not None:
        if len(retrieval_table) != len(retrieval_targets):
            raise ValueError('Mismatched shapes: `retrieval_table` and `retrieval_targets` must have the same length.')

    # Determine base column/feature schema
    if retrieval_table is not None and len(retrieval_table.columns):
        # An empty table can still provide the schema for a header-only cache.
        columns = retrieval_table.columns
    elif row is not None:
        columns = row.index
    else:
        raise ValueError('You must provide either a valid `row` or a non-empty `retrieval_table`.')

    # Tokenize Header Row
    header_tokens: List[int] = []
    if config.with_header:
        header_features = pd.Series(columns, index=columns)
        header_tokens.extend(_clean_and_tokenize_series(header_features, tokenizer, config))

        # Column names are labels, including when they happen to look numeric.
        target_str = f'{target_column if target_column is not None else "target"}\n'
        header_tokens.extend(tokenizer(target_str)['input_ids'])

    # Tokenize Query Row (The primary row being processed)
    query_row_tokens: List[int] = []
    if row is not None:
        query_row_tokens.extend(_clean_and_tokenize_series(row, tokenizer, config))
        # Empty string target generation matching original behavior
        query_row_tokens.extend(tokenizer('')['input_ids'])

    fixed_token_budget = len(header_tokens) + len(query_row_tokens)
    if fixed_token_budget > config.max_length:
        raise ValueError(
            f'Header and query require {fixed_token_budget} tokens, exceeding max_length={config.max_length}. '
            'Increase max_length or reduce max_cell_tokens.'
        )

    if retrieval_table is None or retrieval_table.empty:
        return {'input_ids': header_tokens + query_row_tokens}

    # 6. Process Retrieval Context (Packing up to token budget limit)
    retrieval_sequences: List[List[int]] = []
    current_retrieval_length = 0

    # Note: Using .values or zipped arrays here avoids the huge overhead of creating pd.Series objects in a loop
    feature_rows = retrieval_table.values
    target_values = retrieval_targets.values if retrieval_targets is not None else [None] * len(retrieval_table)
    column_names = retrieval_table.columns

    for features, target in zip(feature_rows, target_values):
        # Construct temporary series swiftly without index overhead penalties
        feat_series = pd.Series(features, index=column_names)

        row_feat_tokens = _clean_and_tokenize_series(feat_series, tokenizer, config)
        if retrieval_targets is not None and isinstance(retrieval_targets.dtype, pd.CategoricalDtype):
            target = str(target) if pd.notna(target) else None
        target_str = _format_target_value(target, config, eos_token=tokenizer.eos_token)
        row_target_tokens = tokenizer(target_str)['input_ids']

        current_row_tokens = row_feat_tokens + row_target_tokens

        # Budget check: Stop adding historical rows if we cross max_length
        if fixed_token_budget + current_retrieval_length + len(current_row_tokens) > config.max_length:
            break

        retrieval_sequences.append(current_row_tokens)
        current_retrieval_length += len(current_row_tokens)

    # 7. Reverse historical context to keep closest matching context nearest to the main query row
    retrieval_sequences.reverse()
    flat_retrieval_tokens = [tok for seq in retrieval_sequences for tok in seq]

    return {'input_ids': header_tokens + flat_retrieval_tokens + query_row_tokens}
