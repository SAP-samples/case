import copy
import importlib.util
import logging
import random

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.preprocessing import normalize
from sklearn.utils.validation import check_is_fitted
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, DynamicCache

from case.config import SerializationConfig
from case.core import serialize_table

logger = logging.getLogger(__name__)


def _resolve_attn_implementation() -> str:
    """Use Flash Attention only when both the package and a CUDA device exist."""
    if torch.cuda.is_available() and importlib.util.find_spec('flash_attn') is not None:
        return 'flash_attention_2'
    return 'sdpa'


def _unique_names(names):
    """Create stable string names without collisions."""
    reserved = set(map(str, names))
    used = set()
    result = []
    for value in names:
        name = str(value)
        candidate = name
        suffix = 1
        if candidate in used:
            while candidate in used or candidate in reserved:
                candidate = f'{name}_{suffix}'
                suffix += 1
        used.add(candidate)
        result.append(candidate)
    return result


class CaseTransformer(TransformerMixin, BaseEstimator):
    """Generate contextual row embeddings and append their PCA features.

    max_length bounds context plus query tokens. Space is reserved for a query
    containing max_cell_tokens tokens per feature, all separators, and the
    checkpoint's pooling token. A schema that cannot fit raises ValueError.

    context_cv=1 (the default) disables cross-fitting and builds context once.
    Set context_cv>1 to enable cross-fitting for supervised training: each row
    is embedded using context from other folds, and PCA fits those embeddings.
    A final context cache is then built using the full training set. With
    cross-fitting enabled, fit_transform(X, y) can differ from
    fit(X, y).transform(X); use fit_transform for downstream training.

    Targets may be one-dimensional array-like objects or None. Text and pandas
    categorical targets retain their spelling; scientific_notation controls
    numeric targets. Set it to False for integer class IDs. Feature names must
    be unique; transform accepts reordered
    columns and restores the fitted order, but rejects missing or extra columns.
    """

    def __init__(
        self,
        model_name: str = 'SAP/CASE',
        max_length: int = 16_000,
        num_context_rows: int = 128,
        n_components: int = 32,
        scientific_notation: bool = True,
        normalize_embeddings: bool = False,
        batch_size: int = 16,
        dtype: torch.dtype = torch.bfloat16,
        attn_implementation: str | None = None,
        random_state: int | None = 42,
        append_original_features: bool = True,
        max_cell_tokens: int = 50,
        context_cv: int = 1,
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
        self.max_cell_tokens = max_cell_tokens
        self.context_cv = context_cv
        self._model = None
        self._tokenizer = None
        self._device = None
        self._model_signature = None
        self._kv_cache = None
        self._is_fitted = False

    def __sklearn_is_fitted__(self):
        return self._is_fitted

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.input_tags.string = True
        tags.input_tags.categorical = True
        tags.transformer_tags.preserves_dtype = []
        return tags

    def _reset_fit(self):
        self._is_fitted = False
        self._kv_cache = None
        for name in (
            'pca_',
            'n_features_in_',
            'feature_names_in_',
            'target_column_name_',
            '_columns_',
            '_input_names_',
            'embedding_feature_names_',
            '_query_budget_',
            '_context_budget_',
            '_fit_parameters_',
        ):
            self.__dict__.pop(name, None)

    def set_params(self, **params):
        previous = self.get_params(deep=False)
        super().set_params(**params)
        if any(
            previous[name] != value
            for name, value in params.items()
            if name not in ('batch_size', 'append_original_features')
        ):
            self._reset_fit()
        return self

    def _validate_parameters(self):
        for name in ('max_length', 'max_cell_tokens', 'batch_size', 'n_components', 'context_cv'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f'{name} must be a positive integer.')
        value = self.num_context_rows
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError('num_context_rows must be a non-negative integer.')
        for name in ('scientific_notation', 'normalize_embeddings', 'append_original_features'):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f'{name} must be a boolean.')
        if self.random_state is not None and (
            isinstance(self.random_state, bool)
            or not isinstance(self.random_state, int)
            or not 0 <= self.random_state < 2**32
        ):
            raise ValueError('random_state must be None or an integer in [0, 2**32).')
        if not isinstance(self.model_name, str) or not self.model_name:
            raise ValueError('model_name must be a non-empty string.')
        if not isinstance(self.dtype, torch.dtype) or not self.dtype.is_floating_point:
            raise ValueError('dtype must be a floating-point torch.dtype.')
        if self.attn_implementation is not None and not isinstance(self.attn_implementation, str):
            raise ValueError('attn_implementation must be None or a string.')

    def _validate_X(self, X, *, fitting=False):
        if not isinstance(X, pd.DataFrame) and np.asarray(X).ndim != 2:
            raise ValueError('X must be a two-dimensional table.')
        frame = pd.DataFrame(X).copy()
        if frame.shape[1] == 0:
            raise ValueError('X must contain at least one feature.')
        if not frame.columns.is_unique:
            raise ValueError('Feature names must be unique.')
        if fitting:
            if len(frame) < 2:
                raise ValueError('X must contain at least two rows to fit PCA.')
        else:
            if not isinstance(X, pd.DataFrame):
                if frame.shape[1] != self.n_features_in_:
                    raise ValueError('X must have the same feature names and count as during fit.')
                frame.columns = self._columns_
            if len(frame.columns) != len(self._columns_) or not self._columns_.isin(frame.columns).all():
                raise ValueError('X must have the same feature names and count as during fit.')
            frame = frame.reindex(columns=self._columns_)
        return frame

    @staticmethod
    def _validate_y(y, n_rows):
        if y is None:
            return None
        if np.asarray(y).ndim != 1:
            raise ValueError('y must be one-dimensional.')
        if len(y) != n_rows:
            raise ValueError('X and y must have the same number of rows.')
        result = pd.Series(y).reset_index(drop=True)
        result.name = str(result.name) if result.name is not None else 'target'
        return result

    def _setup_model(self):
        """Load the backbone without its vocabulary head, reusing identical weights."""
        attention = self.attn_implementation or _resolve_attn_implementation()
        signature = (self.model_name, self.dtype, attention)
        if self._model is not None and self._model_signature == signature:
            return
        self._model = self._tokenizer = self._device = None
        self._model_signature = None
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # Load only the backbone: per-token vocabulary logits are unused and can consume several GiB.
        model = AutoModel.from_pretrained(
            self.model_name,
            device_map='auto',
            dtype=self.dtype,
            attn_implementation=attention,
        )
        model.eval()
        if getattr(model.config.get_text_config(), 'use_bidirectional_attention', False):
            raise ValueError('Context caching requires a causal model, not bidirectional attention.')
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is None:
                raise ValueError('The tokenizer must provide a padding or EOS token.')
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'right'
        device = model.get_input_embeddings().weight.device
        if device.type == 'meta':
            device = getattr(getattr(model.get_input_embeddings(), '_hf_hook', None), 'execution_device', None)
            if device is None:
                raise ValueError('Cannot determine the execution device for the offloaded input embeddings.')
        self._tokenizer, self._model, self._device = tokenizer, model, device
        self._model_signature = signature

    def _serialization_config(self, *, with_header, max_length=None):
        return SerializationConfig(
            max_length=self.max_length if max_length is None else max_length,
            max_cell_tokens=self.max_cell_tokens,
            scientific_notation=self.scientific_notation,
            with_header=with_header,
        )

    def _set_token_budgets(self, X):
        text_config = self._model.config.get_text_config()
        model_limit = getattr(text_config, 'max_position_embeddings', None)
        if model_limit is not None and self.max_length > model_limit:
            raise ValueError(f'max_length exceeds the model context limit of {model_limit} tokens.')
        config = self._serialization_config(with_header=True)
        header = serialize_table(
            self.target_column_name_,
            self._tokenizer,
            retrieval_table=X.iloc[:0],
            config=config,
        )['input_ids']
        separator_length = len(self._tokenizer(config.cell_separator, add_special_tokens=False)['input_ids'])
        pooling_length = len(self._tokenizer('')['input_ids'])
        self._query_budget_ = self.n_features_in_ * (self.max_cell_tokens + separator_length) + pooling_length
        self._context_budget_ = self.max_length - self._query_budget_
        if self._context_budget_ < len(header):
            raise ValueError(
                'max_length cannot fit the header and reserved query tokens. '
                'Increase max_length or reduce max_cell_tokens.'
            )

    def _prefill_kv(self, X, y=None):
        """Build a fresh context cache, including a header when no examples are used."""
        rng = random.Random(None if self.random_state is None else int(self.random_state))
        indices = rng.sample(range(len(X)), min(len(X), self.num_context_rows))
        context = X.iloc[indices]
        targets = y.iloc[indices] if y is not None else None
        inputs = serialize_table(
            target_column=self.target_column_name_,
            tokenizer=self._tokenizer,
            retrieval_table=context,
            retrieval_targets=targets,
            config=self._serialization_config(with_header=True, max_length=self._context_budget_),
        )
        self._kv_cache = None
        input_ids = torch.tensor([inputs['input_ids']], dtype=torch.long, device=self._device)
        cache = DynamicCache(config=self._model.config)
        with torch.no_grad():
            outputs = self._model(
                input_ids=input_ids,
                past_key_values=cache,
                use_cache=True,
                output_hidden_states=False,
                output_attentions=False,
                return_dict=True,
            )
        cache = outputs.past_key_values
        if cache is None:
            raise ValueError('The model must return a reusable causal KV cache.')
        # Sliding layers can retain views into full-prefill storage.
        for layer in cache.layers:
            if getattr(layer, 'is_sliding', False) and layer.keys is not None:
                layer.keys = layer.keys.clone(memory_format=torch.contiguous_format)
                layer.values = layer.values.clone(memory_format=torch.contiguous_format)
        self._kv_cache = cache

    def _prepare_batch_kv(self, batch_size):
        batch_cache = copy.deepcopy(self._kv_cache)
        if batch_size > 1:
            batch_cache.batch_repeat_interleave(batch_size)
        return batch_cache

    def _embed(self, X):
        hidden_dim = self._model.config.get_text_config().hidden_size
        embeddings = np.empty((len(X), hidden_dim), dtype=np.float32)
        config = self._serialization_config(with_header=False, max_length=self._query_budget_)
        context_length = self._kv_cache.get_seq_length()
        for start in tqdm(range(0, len(X), self.batch_size), desc='Embedding batches'):
            rows = X.iloc[start : start + self.batch_size]
            inputs = [
                serialize_table(self.target_column_name_, self._tokenizer, row=row, config=config)['input_ids']
                for _, row in rows.iterrows()
            ]
            if any(not ids for ids in inputs):
                raise ValueError('The tokenizer produced an empty query sequence.')
            padded = self._tokenizer.pad({'input_ids': inputs}, padding=True, return_tensors='pt')
            input_ids = padded['input_ids'].to(self._device)
            if context_length + input_ids.shape[1] > self.max_length:
                raise ValueError('Context plus query exceeds max_length.')
            attention_mask = torch.cat(
                [
                    torch.ones((len(rows), context_length), dtype=torch.long, device=self._device),
                    padded['attention_mask'].to(self._device),
                ],
                dim=1,
            )
            cache = self._prepare_batch_kv(len(rows))
            with torch.inference_mode():
                outputs = self._model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=cache,
                    use_cache=True,
                    output_hidden_states=False,
                    output_attentions=False,
                    return_dict=True,
                )
                hidden = outputs.last_hidden_state
                lengths = torch.tensor([len(ids) - 1 for ids in inputs], device=hidden.device)
                selected = hidden[torch.arange(len(rows), device=hidden.device), lengths]
                embeddings[start : start + len(rows)] = selected.float().cpu().numpy()
            # Release the old batch before allocating its successor.
            del outputs, hidden, selected, cache, input_ids, attention_mask
        if self.normalize_embeddings and len(embeddings):
            embeddings = normalize(embeddings, norm='l2')
        return embeddings

    def _fit(self, X, y):
        self._reset_fit()
        try:
            self._validate_parameters()
            X = self._validate_X(X, fitting=True)
            y = self._validate_y(y, len(X))
            self._setup_model()
            self._columns_ = X.columns.copy()
            self.n_features_in_ = X.shape[1]
            if all(isinstance(name, str) for name in X.columns):
                self.feature_names_in_ = X.columns.to_numpy(copy=True)
            self._input_names_ = _unique_names(X.columns.tolist())
            self.target_column_name_ = y.name if y is not None else 'target'
            self._set_token_budgets(X)
            if y is not None and self.num_context_rows > 0 and self.context_cv > 1:
                hidden_dim = self._model.config.get_text_config().hidden_size
                raw = np.empty((len(X), hidden_dim), dtype=np.float32)
                folds = KFold(
                    n_splits=min(self.context_cv, len(X)),
                    shuffle=True,
                    random_state=None if self.random_state is None else int(self.random_state),
                )
                for context_indices, query_indices in folds.split(X):
                    self._prefill_kv(X.iloc[context_indices], y.iloc[context_indices])
                    raw[query_indices] = self._embed(X.iloc[query_indices])
                self._prefill_kv(X, y)
            else:
                self._prefill_kv(X, y)
                raw = self._embed(X)
            effective_components = min(self.n_components, *raw.shape)
            if effective_components < self.n_components:
                logger.info('Lowering PCA components from %s to %s.', self.n_components, effective_components)
            self.pca_ = PCA(n_components=effective_components, random_state=self.random_state).fit(raw)
            all_names = _unique_names(self._input_names_ + [f'lm_pca_{i}' for i in range(effective_components)])
            self.embedding_feature_names_ = np.asarray(all_names[self.n_features_in_ :], dtype=object)
            self._fit_parameters_ = self._fitting_parameters()
            self._is_fitted = True
            return X, raw
        except BaseException:
            self._reset_fit()
            raise

    def _fitting_parameters(self):
        return {
            key: value
            for key, value in self.get_params(deep=False).items()
            if key not in ('batch_size', 'append_original_features')
        }

    def _check_fitted(self):
        check_is_fitted(self)
        self._validate_parameters()
        if self._fit_parameters_ != self._fitting_parameters():
            raise ValueError('Fitting parameters changed; call fit or fit_transform again.')

    def _output(self, X, projected):
        pca_frame = pd.DataFrame(projected, columns=self.embedding_feature_names_, index=X.index)
        if not self.append_original_features:
            return pca_frame
        original = X.copy()
        original.columns = self._input_names_
        return pd.concat([original, pca_frame], axis=1)

    def fit(self, X, y=None):
        """Fit PCA on training embeddings and retain context for future inference."""
        self._fit(X, y)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        """Return training features, optionally cross-fitted, in one embedding pass."""
        if fit_params:
            raise TypeError(f'Unsupported fit parameters: {", ".join(fit_params)}')
        frame, raw = self._fit(X, y)
        return self._output(frame, self.pca_.transform(raw))

    def embed(self, X):
        """Return raw (n_samples, hidden_dim) embeddings using the final fitted context.

        For supervised downstream training without a row seeing its own label,
        enable context_cv>1 and use fit_transform.
        """
        self._check_fitted()
        return self._embed(self._validate_X(X))

    def transform(self, X):
        """Append PCA features for new rows using the final fitted context."""
        self._check_fitted()
        frame = self._validate_X(X)
        raw = self._embed(frame)
        projected = self.pca_.transform(raw) if len(frame) else np.empty((0, self.pca_.n_components_))
        return self._output(frame, projected)

    def get_feature_names_out(self, input_features=None):
        """Return the stable output schema, including collision-free PCA names."""
        self._check_fitted()
        if input_features is not None and list(input_features) != list(self._columns_):
            raise ValueError('input_features must match the feature names seen during fit.')
        if self.append_original_features:
            return np.asarray(self._input_names_ + self.embedding_feature_names_.tolist(), dtype=object)
        return self.embedding_feature_names_.copy()
