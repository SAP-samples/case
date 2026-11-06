import copy
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd
import requests
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from case.config import SerializationConfig
from case.core import get_constant_columns, serialize_table


class CaseTransformer(BaseEstimator, TransformerMixin):
    """A scikit-learn transformer that generates context-aware semantic embeddings
    for tabular data by serializing tables and extracting hidden state representations.

    This estimator converts tabular inputs into structured token sequences, optionally
    builds in-context historical retrieval frames, and extracts embeddings using one of two backends:
        1. **'local'**: Direct Hugging Face model inference with manual KV cache pre-filling and batching.
        2. **'vllm'**: Multi-threaded HTTP requests to a remote vLLM + LMCache server endpoint.

    The final multi-dimensional embeddings can be returned in their raw form or reduced
    using an internal PCA projection layer to fit standard downstream tasks.

    Attributes:
        model_name (str): Identifier or local path of the Hugging Face transformer backbone.
        backend (str): Execution backend to use, either 'local' or 'vllm'. Defaults to 'local'.
        api_base (str): Base URL of the vLLM embedding server (used when backend='vllm').
            Defaults to 'http://localhost:8000/v1'.
        max_length (int): The absolute maximum token sequence length budget for context sequences.
        num_context_rows (int): The maximum number of context rows to ingest during in-context serialization.
        n_components (int): Target feature dimension size for the internal PCA reduction layer.
        scientific_notation (bool): If True, formats numerical target vectors into standardized
            scientific notation strings.
        normalize_embeddings (bool): If True, applies an L2 normalization step to the generated
            embeddings before PCA reduction.
        batch_size (int): Size of chunks to slice and feed into the local transformer model
            during local inference loops. Defaults to 16.
        max_workers (int): Maximum number of worker threads for concurrent row requests
            when using the vllm backend. Defaults to 8.
        dtype (torch.dtype): Target precision datatype utilized for local model weight allocations.
        attn_implementation (str): Attention optimization backend to load locally (e.g., "flash_attention_2", "sdpa").
        random_state (Optional[int]): Seed value utilized to guarantee deterministic context sampling and PCA decomposition.
        append_original_features (bool): If True, the original input features are horizontally concatenated
            side-by-side with the newly generated language model features in the final returned DataFrame. Defaults to True.
        select_cols (Optional[List[str]]): Specific subset of columns to filter and select from input tables
            before serialization and embedding. Defaults to None.
        return_raw (bool): If True, returns raw unreduced language model embedding vectors (lm_emb_x)
            instead of PCA components (lm_pca_x). Defaults to False.
    """

    def __init__(
        self,
        model_name: str,
        backend: str = 'local',
        api_base: str = 'http://localhost:8000/v1',
        max_length: int = 8_000,
        num_context_rows: int = 128,
        n_components: int = 32,
        scientific_notation: bool = True,
        normalize_embeddings: bool = False,
        batch_size: int = 16,  # Used for local backend batching
        max_workers: int = 64,  # Used for vllm multi-threaded requests
        dtype: torch.dtype = torch.bfloat16,
        attn_implementation: str = 'sdpa',
        random_state: Optional[int] = 42,
        append_original_features: bool = True,
        select_cols: Optional[List[str]] = None,
        return_raw: bool = False,
    ):
        self.model_name = model_name
        self.backend = backend.lower()
        self.api_base = api_base.rstrip('/')
        self.max_length = max_length
        self.num_context_rows = num_context_rows
        self.n_components = n_components
        self.scientific_notation = scientific_notation
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.dtype = dtype
        self.attn_implementation = attn_implementation
        self.random_state = random_state
        self.append_original_features = append_original_features
        self.select_cols = select_cols
        self.return_raw = return_raw

        # Private internal states
        self._model: Any = None
        self._tokenizer: Any = None
        self._kv_cache: Any = None
        self._context_ids: List[int] = []
        self._device: Optional[str] = None
        self.constant_columns: Optional[List[str]] = None
        self._hidden_dim: Optional[int] = None
        self._cols_to_embed_: Optional[pd.Index] = None
        self.target_column_name_: str = 'target'
        self.pca_: Optional[PCA] = None

        if self.backend not in ['local', 'vllm']:
            raise ValueError("Unsupported backend. Choose either 'local' or 'vllm'.")

    def _format_ids(self, ids: Any) -> List[int]:
        """Ensures extracted IDs are a clean, flat Python list of integers."""
        if hasattr(ids, 'tolist'):
            ids = ids.tolist()
        if isinstance(ids, list) and len(ids) > 0 and isinstance(ids[0], list):
            ids = ids[0]
        return list(ids)

    # ==========================================
    # SHARED SETUP & CONTEXT HELPERS
    # ==========================================

    def _get_serialized_context(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Optional[dict]:
        """Unified context sampling and serialization logic for both backends."""
        if self.num_context_rows <= 0:
            return None

        rng = random.Random(self.random_state)
        sampled_indices = rng.sample(range(len(X)), min(len(X), self.num_context_rows))
        X_context = X.iloc[sampled_indices]
        y_context = y.iloc[sampled_indices] if y is not None else None
        y_str = y_context.astype('string') if y_context is not None else None

        config = SerializationConfig(max_length=self.max_length, with_header=True)
        return serialize_table(
            config=config,
            row=None,
            retrieval_table=X_context,
            retrieval_targets=y_str,
            tokenizer=self._tokenizer,
            target_column=self.target_column_name_,
        )

    def _setup_model(self, X: pd.DataFrame, y: Optional[pd.Series]) -> None:
        """Unified initialization for tokenizer, model, and context cache/prefix."""
        self.target_column_name_ = y.name if y is not None else 'target'
        self._cols_to_embed_ = X.columns

        # 1. Unified Tokenizer Initialization
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            self._tokenizer.padding_side = 'right'

        # 2. Backend-specific Setup
        if self.backend == 'local':
            if self._model is None:
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map='auto',
                    torch_dtype=self.dtype,
                    attn_implementation=self.attn_implementation,
                )
                self._device = next(self._model.parameters()).device

            if self._kv_cache is None:
                inputs = self._get_serialized_context(X, y)
                if inputs:
                    self._kv_cache = DynamicCache(config=self._model.config)
                    input_ids = torch.tensor([inputs['input_ids']], dtype=torch.long).to(self._device)
                    with torch.no_grad():
                        outputs = self._model(input_ids, past_key_values=self._kv_cache)
                        self._kv_cache = outputs.past_key_values

        elif self.backend == 'vllm':
            if not self._context_ids:
                inputs = self._get_serialized_context(X, y)
                if inputs:
                    self._context_ids = self._format_ids(inputs['input_ids'])

    # ==========================================
    # EMBEDDING GENERATION METHODS
    # ==========================================

    def _embed(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> np.ndarray:
        n_rows = X.shape[0]
        text_config = getattr(self._model.config, 'text_config', self._model.config) if self._model else None
        hidden_dim = self._hidden_dim or (text_config.hidden_size if text_config else 3840)

        if n_rows == 0 or len(self._cols_to_embed_) == 0:
            return np.empty((0, 1, hidden_dim), dtype=np.float32)

        return self._embed_local(X) if self.backend == 'local' else self._embed_vllm(X)

    def _embed_local(self, X: pd.DataFrame) -> np.ndarray:
        """Batched GPU inference using local Hugging Face backbone and KV cache."""
        n_rows = X.shape[0]
        config = SerializationConfig(max_length=self.max_length, with_header=False)

        all_input_ids = [
            serialize_table(
                config=config,
                row=row[self._cols_to_embed_],
                retrieval_table=None,
                retrieval_targets=None,
                tokenizer=self._tokenizer,
                target_column=self.target_column_name_,
            )['input_ids']
            for _, row in X.iterrows()
        ]

        text_config = getattr(self._model.config, 'text_config', self._model.config)
        hidden_dim = text_config.hidden_size
        self._hidden_dim = hidden_dim
        final_embeddings = np.zeros((n_rows, 1, hidden_dim), dtype=np.float32)
        past_seq_len = self._kv_cache.get_seq_length() if self._kv_cache else 0

        for chunk_offset in tqdm(range(0, n_rows, self.batch_size), desc='Embedding (Local HF)'):
            batch_inputs = all_input_ids[chunk_offset : chunk_offset + self.batch_size]
            current_batch_size = len(batch_inputs)

            padded = self._tokenizer.pad({'input_ids': batch_inputs}, padding=True, return_tensors='pt')
            input_ids = padded['input_ids'].to(self._device)
            true_lengths = [len(x) for x in batch_inputs]

            past_key_values = None
            if past_seq_len > 0:
                past_key_values = copy.deepcopy(self._kv_cache)
                if current_batch_size > 1:
                    past_key_values.batch_repeat_interleave(current_batch_size)

            with torch.no_grad():
                outputs = self._model(
                    input_ids=input_ids,
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                )

            last_hidden_state = outputs.hidden_states[-1]
            for batch_idx in range(current_batch_size):
                original_idx = chunk_offset + batch_idx
                last_valid_token_idx = true_lengths[batch_idx] - 1
                embedding = last_hidden_state[batch_idx, last_valid_token_idx, :].unsqueeze(0)
                final_embeddings[original_idx] = embedding.cpu().float().numpy()

        if self.normalize_embeddings:
            reshaped_embs = final_embeddings.reshape(-1, hidden_dim)
            normalized_2d = normalize(reshaped_embs, norm='l2')
            final_embeddings = normalized_2d.reshape(final_embeddings.shape)

        return final_embeddings

    def _embed_vllm(self, X: pd.DataFrame) -> np.ndarray:
        """Pure multi-threaded inference dispatching row requests to remote vLLM server."""
        n_rows = X.shape[0]
        config = SerializationConfig(max_length=self.max_length, with_header=False)

        all_input_ids = [
            self._context_ids
            + self._format_ids(
                serialize_table(
                    config=config,
                    row=row[self._cols_to_embed_],
                    retrieval_table=None,
                    retrieval_targets=None,
                    tokenizer=self._tokenizer,
                    target_column=self.target_column_name_,
                )['input_ids']
            )
            for _, row in X.iterrows()
        ]

        def fetch_row_embedding(idx: int, input_ids: list) -> tuple:
            response = requests.post(
                f'{self.api_base}/embeddings',
                json={'model': self.model_name, 'input': input_ids},
                headers={'Content-Type': 'application/json'},
                timeout=180.0,
            )
            response.raise_for_status()
            res_json = response.json()
            emb = np.array(res_json['data'][0]['embedding'], dtype=np.float32).reshape(1, -1)
            return idx, emb

        results = {}
        max_workers = min(self.max_workers, n_rows) if n_rows > 0 else 1

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {executor.submit(fetch_row_embedding, i, ids): i for i, ids in enumerate(all_input_ids)}
            for future in tqdm(as_completed(future_to_idx), total=n_rows, desc='Embedding (vLLM Multi-threaded)'):
                idx = future_to_idx[future]
                try:
                    i, emb = future.result()
                    results[i] = emb
                except Exception as e:
                    print(f'[CASE Error] Request failed at row index {idx}: {e}')
                    raise e

        final_embeddings = [results[i] for i in range(n_rows)]
        final_embeddings_arr = np.array(final_embeddings, dtype=np.float32)

        if self._hidden_dim is None and len(final_embeddings_arr) > 0:
            self._hidden_dim = final_embeddings_arr.shape[-1]

        if self.normalize_embeddings:
            hidden_dim = final_embeddings_arr.shape[-1]
            reshaped_embs = final_embeddings_arr.reshape(-1, hidden_dim)
            normalized_2d = normalize(reshaped_embs, norm='l2')
            final_embeddings_arr = normalized_2d.reshape(final_embeddings_arr.shape)

        return final_embeddings_arr

    # ==========================================
    # SCIKIT-LEARN API METHODS
    # ==========================================

    def _preprocess_input(self, X: Union[pd.DataFrame, np.ndarray], fit: bool = False) -> pd.DataFrame:
        """Unified preprocessing for column filtering and constant column handling."""
        X_df = pd.DataFrame(X).copy()
        if self.select_cols:
            X_df = X_df[[c for c in self.select_cols if c in X_df.columns]]

        if fit:
            self.constant_columns = get_constant_columns(X_df)

        if self.constant_columns and len(self.constant_columns) > 0:
            drop_cols = [c for c in self.constant_columns if c in X_df.columns]
            if drop_cols:
                X_df = X_df.drop(columns=drop_cols)
        return X_df

    def _fit_pca(self, raw_embs: np.ndarray) -> None:
        """Shared dynamic PCA component fitting logic."""
        n_samples = raw_embs.shape[0]
        hidden_dim = raw_embs.shape[-1]
        effective_components = min(self.n_components, n_samples, hidden_dim)

        if effective_components < self.n_components:
            print(
                f'Warning: Lowering PCA components from {self.n_components} to {effective_components} due to sample size bounds.'
            )

        self.pca_ = PCA(n_components=effective_components, random_state=self.random_state)

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[pd.Series] = None) -> 'CaseTransformer':
        X_df = self._preprocess_input(X, fit=True)
        self._setup_model(X_df, y)
        raw_embs = self._embed(X=X_df, y=y)

        self._fit_pca(raw_embs)
        self.pca_.fit(np.squeeze(raw_embs, axis=1))
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        X_df = self._preprocess_input(X, fit=False)
        raw_embs = self._embed(X=X_df, y=None)
        squeezed_embs = np.squeeze(raw_embs, axis=1)

        if self.return_raw:
            raw_df = pd.DataFrame(
                squeezed_embs,
                columns=[f'lm_emb_{i}' for i in range(squeezed_embs.shape[1])],
                index=X_df.index,
            )
            return (
                pd.concat([pd.DataFrame(X, index=X_df.index), raw_df], axis=1)
                if self.append_original_features
                else raw_df
            )

        embs_pca = self.pca_.transform(squeezed_embs)
        pca_df = pd.DataFrame(
            embs_pca,
            columns=[f'lm_pca_{i}' for i in range(embs_pca.shape[1])],
            index=X_df.index,
        )

        return (
            pd.concat([pd.DataFrame(X, index=X_df.index), pca_df], axis=1) if self.append_original_features else pca_df
        )

    def fit_transform(
        self, X: Union[pd.DataFrame, np.ndarray], y: Optional[pd.Series] = None, **fit_params
    ) -> pd.DataFrame:
        X_df = self._preprocess_input(X, fit=True)
        self._setup_model(X_df, y)
        raw_embs = self._embed(X=X_df, y=y)

        self._fit_pca(raw_embs)
        squeezed_embs = np.squeeze(raw_embs, axis=1)

        with threadpool_limits(limits=1, user_api='blas'):
            embs_pca = self.pca_.fit_transform(squeezed_embs)

        if self.return_raw:
            raw_df = pd.DataFrame(
                squeezed_embs,
                columns=[f'lm_emb_{i}' for i in range(squeezed_embs.shape[1])],
                index=X_df.index,
            )
            return (
                pd.concat([pd.DataFrame(X, index=X_df.index), raw_df], axis=1)
                if self.append_original_features
                else raw_df
            )

        pca_df = pd.DataFrame(
            embs_pca,
            columns=[f'lm_pca_{i}' for i in range(embs_pca.shape[1])],
            index=X_df.index,
        )

        return (
            pd.concat([pd.DataFrame(X, index=X_df.index), pca_df], axis=1) if self.append_original_features else pca_df
        )
