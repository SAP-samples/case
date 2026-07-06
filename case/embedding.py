import copy
import random
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from case.config import SerializationConfig
from case.core import serialize_table


class CaseTransformer(BaseEstimator, TransformerMixin):
    """A scikit-learn transformer that generates context-aware semantic embeddings
    for tabular data by serializing tables for LLM processing.

    This estimator converts tabular inputs into structured token sequences,
    optionally builds in-context historical retrieval frames, and extracts hidden state
    representations from a transformer backbone. The final multi-dimensional embeddings
    are reduced using an internal PCA projection layer to fit standard downstream tasks.

    Attributes:
        model_name (str): Identifier or local path of the Hugging Face transformer backbone.
        max_length (int): The absolute maximum token sequence length budget for context sequences.
        num_context_rows (int): The maximum number of context rows to ingest during in-context serialization.
        n_components (int): Target feature dimension size for the internal PCA reduction layer.
        scientific_notation (bool): If True, formats numerical target vectors into standardized
            scientific notation strings.
        normalize_embeddings (bool): If True, applies an L2 normalization step to the generated
            embeddings before PCA reduction.
        batch_size (int): Size of chunks to slice and feed into the transformer model during inference loops.
        dtype (torch.dtype): Target precision datatype utilized for model weight allocations.
        attn_implementation (str): Attention optimization backend to load (e.g., "flash_attention_2", "sdpa").
        random_state (Optional[int]): Seed value utilized to guarantee deterministic PCA decomposition.
        append_original_features (bool): If True, the original input features (features passed into `X`) are
            horizontally concatenated side-by-side with the newly generated language model PCA features in the
            final returned DataFrame. If False, only the `lm_pca_x` components are returned. Defaults to True.
    """

    def __init__(
        self,
        model_name: str,
        max_length: int = 16_000,
        num_context_rows: int = 128,
        n_components: int = 32,
        scientific_notation: bool = True,
        normalize_embeddings: bool = False,
        batch_size: int = 16,
        dtype: torch.dtype = torch.bfloat16,
        attn_implementation: str = 'flash_attention_2',
        random_state: Optional[int] = 42,
        append_original_features: bool = True,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.num_context_rows = num_context_rows
        self.n_components = n_components
        self.scientific_notation = scientific_notation
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size
        self.dtype = dtype
        self.attn_implementation = attn_implementation
        self.random_state = random_state
        self.append_original_features = append_original_features

        # Private internal model states
        self._model: Any = None
        self._tokenizer: Any = None
        self._kv_cache: Any = None
        self._device: Optional[str] = None

    def _prefill_kv(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        """Prefills the KV Cache with context rows to guide the LLM."""
        self._kv_cache = DynamicCache(config=self._model.config)
        y_str = y.astype('string') if y is not None else None

        config = SerializationConfig(max_length=self.max_length, with_header=True)

        inputs = serialize_table(
            config=config,
            row=None,
            retrieval_table=X,
            retrieval_targets=y_str,
            tokenizer=self._tokenizer,
            target_column=self.target_column_name_,
        )

        input_ids = torch.tensor([inputs['input_ids']], dtype=torch.long).to(self._device)

        with torch.no_grad():
            outputs = self._model(input_ids, past_key_values=self._kv_cache)
            self._kv_cache = outputs.past_key_values

    def _prepare_batch_kv(self, batch_size: int) -> Optional[DynamicCache]:
        """Leverages native HF interleave utilities to replicate the cache perfectly
        across the batch dimension without breaking internal structural properties.
        """
        if not self._kv_cache:
            return None

        # Create a clean deepcopy of the single-row context cache
        batch_cache = copy.deepcopy(self._kv_cache)

        # Use the standard, built-in HF tool to cleanly scale the batch dimension.
        # This accurately handles keys, values, and layer-level token counters.
        if batch_size > 1:
            batch_cache.batch_repeat_interleave(batch_size)

        return batch_cache

    def _embed(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> np.ndarray:
        """Extracts text embeddings using native right-padding batching.

        Relies entirely on internal HF mechanisms to handle position IDs and
        cache tracking, ensuring optimal embedding accuracy across varying batch sizes.
        """
        n_rows = X.shape[0]
        if n_rows == 0 or len(self._cols_to_embed_) == 0:
            text_config = getattr(self._model.config, 'text_config', self._model.config)
            hidden_dim = text_config.hidden_size
            return np.empty((0, 1, hidden_dim), dtype=np.float32)

        # Enforce right-padding for clean causal alignment
        self._tokenizer.padding_side = 'right'

        config = SerializationConfig(max_length=self.max_length, with_header=False)

        all_input_ids = []
        for unique_row_id, row in X.iterrows():
            inputs = serialize_table(
                config=config,
                row=row[self._cols_to_embed_],
                retrieval_table=None,
                retrieval_targets=None,
                tokenizer=self._tokenizer,
                target_column=self.target_column_name_,
            )
            all_input_ids.append(inputs['input_ids'])

        text_config = getattr(self._model.config, 'text_config', self._model.config)
        hidden_dim = text_config.hidden_size
        final_embeddings = np.zeros((n_rows, 1, hidden_dim), dtype=np.float32)
        past_seq_len = self._kv_cache.get_seq_length() if self._kv_cache else 0

        progress_bar = tqdm(range(0, n_rows, self.batch_size), desc='Embedding batches')

        # Iterate through data batches
        for chunk_offset in progress_bar:  # range(0, n_rows, self.batch_size):
            batch_inputs = all_input_ids[chunk_offset : chunk_offset + self.batch_size]
            current_batch_size = len(batch_inputs)

            # Standardize right-padded batch preparation
            padded = self._tokenizer.pad({'input_ids': batch_inputs}, padding=True, return_tensors='pt')
            input_ids = padded['input_ids'].to(self._device)

            # Map exact text length boundaries to bypass trailing padding vectors later
            true_lengths = [len(x) for x in batch_inputs]

            if past_seq_len > 0:
                # Use your working batch_repeat_interleave replication routine
                past_key_values = self._prepare_batch_kv(batch_size=current_batch_size)
            else:
                past_key_values = None

            # Clean inference call completely absent of manual position overrides
            with torch.no_grad():
                outputs = self._model(
                    input_ids=input_ids,
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                )

            last_hidden_state = outputs.hidden_states[-1]

            # Target Extraction: Pull hidden states using true string boundary limits
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

    def _setup_model(self, X: pd.DataFrame, y: Optional[pd.Series]) -> None:
        """Lazy initialization of the LLM and tokenizer configuration."""
        self.target_column_name_ = y.name if y is not None else 'target'
        self._cols_to_embed_ = X.columns

        if self._model is None:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map='auto',
                dtype=self.dtype,
                attn_implementation=self.attn_implementation,
            )
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            self._tokenizer.padding_side = 'right'

            self._device = next(self._model.parameters()).device

        if self.num_context_rows > 0 and self._kv_cache is None:
            rng = random.Random(self.random_state)
            sampled_indices = rng.sample(range(len(X)), min(len(X), self.num_context_rows))
            X_context = X.iloc[sampled_indices]
            y_context = y.iloc[sampled_indices] if y is not None else None
            self._prefill_kv(X_context, y_context)

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[pd.Series] = None) -> 'CaseTransformer':
        """Fits the transformer by passing data through the LLM and training PCA."""
        X_df = pd.DataFrame(X).copy()
        self._setup_model(X_df, y)
        raw_embs = self._embed(X=X_df, y=y)
        n_samples = raw_embs.shape[0]
        hidden_dim = raw_embs.shape[-1]

        # Dynamic Component Guard: Prevent PCA crashes if n_samples is less than n_components
        effective_components = min(self.n_components, n_samples, hidden_dim)
        if effective_components < self.n_components:
            # Drop a warning or handle gracefully if your dataset is smaller than your component settings
            print(
                f'Warning: Lowering PCA components from {self.n_components} to {effective_components} due to sample size bounds.'
            )

        self.pca_ = PCA(n_components=effective_components, random_state=self.random_state)
        self.pca_.fit(np.squeeze(raw_embs, axis=1))
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """Transforms out-of-sample data into low-dimensional embeddings."""
        X_df = pd.DataFrame(X).copy()
        raw_embs = self._embed(X=X_df, y=None)
        embs_pca = self.pca_.transform(np.squeeze(raw_embs, axis=1))

        pca_df = pd.DataFrame(
            embs_pca,
            columns=[f'lm_pca_{i}' for i in range(embs_pca.shape[1])],
            index=X_df.index,
        )

        # Conditionally concatenate original features
        if self.append_original_features:
            return pd.concat([X_df, pca_df], axis=1)

        return pca_df

    def fit_transform(
        self, X: Union[pd.DataFrame, np.ndarray], y: Optional[pd.Series] = None, **fit_params
    ) -> pd.DataFrame:
        """Optimized fit_transform that runs LLM inference exactly once to avoid heavy recalculations."""
        X_df = pd.DataFrame(X).copy()
        self._setup_model(X_df, y)

        # Generate embeddings exactly once
        raw_embs = self._embed(X=X_df, y=y)
        n_samples = raw_embs.shape[0]
        hidden_dim = raw_embs.shape[-1]
        squeezed_embs = np.squeeze(raw_embs, axis=1)

        # Dynamic Component Guard
        effective_components = min(self.n_components, n_samples, hidden_dim)

        if effective_components < self.n_components:
            print(
                f'Warning: Lowering PCA components from {self.n_components} to {effective_components} due to sample size bounds.'
            )

        # Fit PCA and transform simultaneously
        self.pca_ = PCA(n_components=effective_components, random_state=self.random_state)
        embs_pca = self.pca_.fit_transform(squeezed_embs)

        pca_df = pd.DataFrame(
            embs_pca,
            columns=[f'lm_pca_{i}' for i in range(embs_pca.shape[1])],
            index=X_df.index,
        )

        # Conditionally concatenate original features
        if self.append_original_features:
            return pd.concat([X_df, pca_df], axis=1)

        return pca_df
