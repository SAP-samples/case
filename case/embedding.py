import copy
import importlib.util
import logging
import random
import time

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.preprocessing import normalize
from sklearn.utils.validation import check_is_fitted
from torch.utils._pytree import tree_map
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer, DynamicCache, Gemma3TextModel

from case.config import SerializationConfig
from case.core import serialize_table

logger = logging.getLogger(__name__)


class _CaseGemma3TextModel(Gemma3TextModel):
    """Load CASE's text backbone, excluding unused image keys from the report."""

    _keys_to_ignore_on_load_unexpected = [
        *(Gemma3TextModel._keys_to_ignore_on_load_unexpected or []),
        r'^(?:model\.)?(?:vision_tower|multi_modal_projector)\.',
    ]


def _resolve_attn_implementation(device: str = 'auto') -> str:
    """Use Flash Attention when placement allows CUDA and its package is installed."""
    use_cuda = device == 'auto' or torch.device(device).type == 'cuda'
    if use_cuda and torch.cuda.is_available() and importlib.util.find_spec('flash_attn') is not None:
        return 'flash_attention_2'
    return 'sdpa'


def _validate_text_attention(config):
    """Require causal text attention while allowing vision-only bidirectional modes."""
    bidirectional = getattr(config, 'use_bidirectional_attention', False)
    if (
        bidirectional not in (None, False, 'vision')
        or getattr(config, 'is_causal', True) is False
        or getattr(config, 'is_encoder_decoder', False)
    ):
        raise ValueError(
            'Context caching requires a causal text model, not bidirectional or encoder-decoder attention.'
        )


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
    """A scikit-learn transformer that generates context-aware semantic embeddings
    for tabular data by serializing tables for LLM processing.

    This estimator converts tabular inputs into structured token sequences,
    optionally builds in-context historical retrieval frames, and extracts hidden state
    representations from a transformer backbone. The final multi-dimensional embeddings
    are reduced using an internal PCA projection layer to fit standard downstream tasks.

    Attributes:
        model_name (str): Identifier or local path of the Hugging Face transformer backbone.
            Defaults to "SAP/CASE". Causal text backbones can be loaded directly or from
            checkpoints with a nested text configuration and standard text-weight prefixes.
            No vision tower or vocabulary output head is loaded.
        max_length (int): Token budget for the serialized header and context rows.
            Must not exceed the backbone's context limit. Query rows are added afterward
            with the per-cell cap. If context plus query exceeds this budget, a warning
            is logged and inference is attempted using the fitted context. A header
            that cannot fit raises ValueError.
            Defaults to 16,000.
        num_context_rows (int): Maximum number of training rows sampled without replacement
            for in-context serialization. Fewer rows may be used to fit the token budget;
            context rows are included whole. Set to 0 to use only the header as context.
            Defaults to 128.
        n_components (int): Target feature dimension for the internal PCA reduction layer.
            The fitted dimension is capped at the smaller of the training row count and
            the backbone's hidden dimension. Defaults to 32.
        scientific_notation (bool): If True, formats numerical target values as rounded,
            standardized scientific notation strings. Text and pandas categorical targets
            retain their label identity. Set to False for integer class IDs that must retain
            their exact representation. Defaults to True.
        normalize_embeddings (bool): If True, applies L2 normalization to each generated
            embedding before PCA fitting and projection. Also applies to the raw embeddings
            returned by `embed`. Defaults to False.
        batch_size (int): Number of query rows fed into the transformer model per inference
            batch, during both fitting and transformation. Defaults to 16.
        dtype (torch.dtype): Floating-point precision used to load model weights.
            Defaults to torch.bfloat16. Raw embeddings are returned as NumPy float32 arrays.
        attn_implementation (str | None): Attention backend to load, for example
            "flash_attention_2" or "sdpa". If None, CPU placement selects "sdpa".
            Automatic or CUDA placement selects "flash_attention_2" when CUDA is available
            and the flash-attn package is installed; otherwise selects "sdpa". Defaults to None.
        random_state (int | None): Seed for context sampling, shuffled cross-fitting splits,
            and PCA decomposition. Accepts None or an integer in [0, 2**32). A fixed seed
            does not guarantee identical results across devices or attention backends.
            Defaults to 42.
        append_original_features (bool): If True, horizontally concatenates the original
            input features with the language model PCA features in the returned DataFrame,
            preserving original values and dtypes. If False, returns only the PCA features.
            PCA columns are named `lm_pca_0`, `lm_pca_1`, etc., with suffixes added to avoid
            collisions. All output column names are unique strings. Defaults to True.
        max_cell_tokens (int): Maximum tokens retained per feature value or feature-column
            name during serialization. Separators, target values, and the target-column name
            are outside this per-cell cap. All context tokens count toward `max_length`;
            query rows are added afterward. Defaults to 50.
        context_cv (int): Number of folds for optional supervised cross-fitting. The default,
            1, disables cross-fitting and builds context once. Values greater than 1 enable
            cross-fitting only when `y` is provided and `num_context_rows` is positive.
            The actual fold count is capped at the training row count. See Notes for
            training semantics and computational cost.
        device (str): Model placement: "auto", "cpu", "cuda", or "cuda:<index>".
            "auto" delegates placement to the model loader. Explicit CUDA placement raises
            ValueError if CUDA is unavailable. Changing this setting requires refitting.
            Defaults to "auto".
        model_revision (str | None): Checkpoint commit, tag, or branch used by the model
            loader. None uses the repository default. A resolved commit is retained for
            restoring saved fitted state. Local checkpoint paths must remain available.
        pca_ (PCA): Fitted PCA projection, available after fit or fit_transform.
        n_features_in_ (int): Number of input features seen during fitting.
        feature_names_in_ (numpy.ndarray): Input feature names in fitted order. Present
            only when all input column names are strings.
        target_column_name_ (str): Target name used in serialization; defaults to "target"
            when `y` is unnamed or absent.
        embedding_feature_names_ (numpy.ndarray): Collision-free names of the fitted PCA
            features. Use `get_feature_names_out()` for the complete output schema.

    Notes:
        Inputs must be two-dimensional tables with unique column names. Fitting requires
        at least two rows and one feature. Targets may be one-dimensional array-like
        objects of matching length or None; rows and targets are paired by position.
        Without targets, sampled context rows contain features with empty target values.
        At inference, DataFrame columns are restored to the fitted order, with missing
        or extra columns rejected. Array inputs follow the fitted positional schema.

        `fit_transform` and `transform` return pandas DataFrames, preserving the input
        index when given a DataFrame. `embed` returns a NumPy float32 array of shape
        (n_samples, hidden_dim), optionally normalized, without PCA or appended features.
        Both inference methods accept empty tables with the fitted schema.

        `fit_embed(X, y)` fits context and returns raw training embeddings with the
        same cross-fitting semantics as `fit_transform`, without fitting PCA. Consumers
        fit their own projection on those training embeddings and use `embed` for
        inference. `transform` requires a projection fitted by `fit` or `fit_transform`.

        With cross-fitting enabled, each training row is embedded using context drawn
        only from other folds, and PCA is fitted on those embeddings. A final inference
        context is then sampled from all training rows. For k folds, this requires k + 1
        context prefills; each training row is still embedded once as a query.
        `fit_transform(X, y)` can therefore differ from `fit(X, y).transform(X)`. Use
        `fit_transform` to train downstream models on cross-fitted features. `embed` and
        `transform` always use the final inference context, which may contain a queried
        training row's own target when applied to training data.

        Every call to `fit` or `fit_transform` rebuilds the context cache and PCA. Loaded
        model weights are reused when the model, precision, resolved attention backend,
        and device settings are unchanged. Changing any parameter except `batch_size`
        or `append_original_features` requires refitting before inference. A failed fit
        clears the fitted state.

        Pickling saves context, schema, and PCA while referencing the pretrained
        checkpoint. Backbone weights and the tokenizer reload lazily on first inference.

        The header and context rows are packed under max_length. Query rows are
        serialized batch by batch and added to the fitted context, with each feature
        cell capped at max_cell_tokens. If context plus a query exceeds max_length,
        a warning is logged and inference is attempted with the same context. The
        model may reject sequences beyond its supported capacity; those errors
        propagate to the caller.
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
        device: str = 'auto',
        model_revision: str | None = None,
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
        self.device = device
        self.model_revision = model_revision
        self._model = None
        self._tokenizer = None
        self._device = None
        self._model_signature = None
        self._kv_cache = None
        self._is_fitted = False

    def __sklearn_is_fitted__(self):
        return self._is_fitted

    def __getstate__(self):
        """Save fitted state without another copy of the pretrained backbone."""
        state = super().__getstate__().copy()
        state.update(_model=None, _tokenizer=None, _device=None, _model_signature=None)
        if self._kv_cache is not None:
            state['_kv_cache'] = self._move_cache(copy.deepcopy(self._kv_cache), 'cpu')
        return state

    @staticmethod
    def _move_cache(cache, device):
        def move(value):
            if isinstance(value, torch.Tensor):
                return value.to(device)
            if isinstance(value, torch.device):
                return torch.device(device)
            return value

        for item in [cache, *cache.layers]:
            for name, value in vars(item).items():
                # Hybrid attention caches also store tensors inside recurrent-state dictionaries.
                setattr(item, name, tree_map(move, value))
        return cache

    def _ensure_runtime(self):
        if self._model is None:
            self._setup_model()
            if self._kv_cache is not None:
                self._move_cache(self._kv_cache, self._device)

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
            '_fit_parameters_',
            'embedding_fit_time_',
            'pca_fit_time_',
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
        if self.model_revision is not None and (not isinstance(self.model_revision, str) or not self.model_revision):
            raise ValueError('model_revision must be None or a non-empty string.')
        if not isinstance(self.dtype, torch.dtype) or not self.dtype.is_floating_point:
            raise ValueError('dtype must be a floating-point torch.dtype.')
        if self.attn_implementation is not None and not isinstance(self.attn_implementation, str):
            raise ValueError('attn_implementation must be None or a string.')
        if self.device != 'auto':
            try:
                device = torch.device(self.device)
            except (TypeError, RuntimeError) as error:
                raise ValueError('device must be auto, cpu, cuda, or cuda:<index>.') from error
            if device.type not in ('cpu', 'cuda'):
                raise ValueError('device must be auto, cpu, cuda, or cuda:<index>.')
            if device.type == 'cuda' and not torch.cuda.is_available():
                raise ValueError('CUDA was requested but is not available.')

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
        attention = self.attn_implementation or _resolve_attn_implementation(self.device)
        signature = (self.model_name, self.model_revision, self.dtype, attention, self.device)
        if self._model is not None and self._model_signature == signature:
            return
        self._model = self._tokenizer = self._device = None
        self._model_signature = None
        source = (self.model_name, self.model_revision)
        revision = self.model_revision
        if getattr(self, '_checkpoint_source_', None) == source:
            revision = getattr(self, '_checkpoint_revision_', None) or revision
        revision_kwargs = {'revision': revision} if revision is not None else {}
        config = AutoConfig.from_pretrained(self.model_name, **revision_kwargs)
        text_config = config.get_text_config()
        _validate_text_attention(text_config)
        if getattr(config, 'is_encoder_decoder', False):
            raise ValueError('Context caching requires a causal text model, not an encoder-decoder wrapper.')
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, **revision_kwargs)
        load_kwargs = {}
        model_loader = AutoModel
        if text_config is not config:
            # Strip common wrapper paths when loading only the text backbone. Transformers
            # still handles the backbone's own base-model prefix and weight conversions.
            load_kwargs['key_mapping'] = {r'^(?:model\.)?(?:language_model|text_model)(?:\.model)?\.': ''}
        if config.model_type == 'gemma3':
            model_loader = _CaseGemma3TextModel
        # Load only the backbone: per-token vocabulary logits are unused and can consume several GiB.
        model, loading_info = model_loader.from_pretrained(
            self.model_name,
            config=text_config,
            device_map=self.device,
            dtype=self.dtype,
            attn_implementation=attention,
            output_loading_info=True,
            **revision_kwargs,
            **load_kwargs,
        )
        for issue in ('missing_keys', 'mismatched_keys', 'error_msgs'):
            if loading_info.get(issue):
                details = sorted(map(str, loading_info[issue]))[:5]
                raise ValueError(
                    f'Text backbone weights were not loaded completely ({issue}: {details}). '
                    'Check the checkpoint text configuration and weight prefixes.'
                )
        model.eval()
        _validate_text_attention(model.config.get_text_config())
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
        self._checkpoint_source_ = source
        self._checkpoint_revision_ = getattr(config, '_commit_hash', None) or revision

    def _serialization_config(self, *, with_header):
        return SerializationConfig(
            max_length=self.max_length,
            max_cell_tokens=self.max_cell_tokens,
            scientific_notation=self.scientific_notation,
            with_header=with_header,
        )

    def _prefill_kv(self, X, y=None):
        """Pack context under max_length and build its cache, including a header."""
        text_config = self._model.config.get_text_config()
        model_limit = getattr(text_config, 'max_position_embeddings', None)
        if model_limit is not None and self.max_length > model_limit:
            raise ValueError(f'max_length exceeds the model context limit of {model_limit} tokens.')
        rng = random.Random(None if self.random_state is None else int(self.random_state))
        indices = rng.sample(range(len(X)), min(len(X), self.num_context_rows))
        context = X.iloc[indices]
        targets = y.iloc[indices] if y is not None else None
        inputs = serialize_table(
            target_column=self.target_column_name_,
            tokenizer=self._tokenizer,
            retrieval_table=context,
            retrieval_targets=targets,
            config=self._serialization_config(with_header=True),
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
            # Reordering supports both KV layers and hybrid recurrent/linear-attention states.
            batch_cache.reorder_cache(torch.zeros(batch_size, dtype=torch.long))
        return batch_cache

    def _embed(self, X):
        self._ensure_runtime()
        hidden_dim = self._model.config.get_text_config().hidden_size
        embeddings = np.empty((len(X), hidden_dim), dtype=np.float32)
        context_length = self._kv_cache.get_seq_length()
        config = self._serialization_config(with_header=False)
        for start in tqdm(range(0, len(X), self.batch_size), desc='Embedding batches'):
            rows = X.iloc[start : start + self.batch_size]
            inputs = [
                serialize_table(
                    self.target_column_name_,
                    self._tokenizer,
                    row=row,
                    config=config,
                    enforce_max_length=False,
                )['input_ids']
                for _, row in rows.iterrows()
            ]
            if any(not ids for ids in inputs):
                raise ValueError('The tokenizer produced an empty query sequence.')
            padded = self._tokenizer.pad({'input_ids': inputs}, padding=True, return_tensors='pt')
            input_ids = padded['input_ids'].to(self._device)
            total_length = context_length + input_ids.shape[1]
            if total_length > self.max_length:
                logger.warning(
                    'Context plus query requires %d tokens, exceeding max_length=%d. '
                    'Attempting inference with the fitted context.',
                    total_length,
                    self.max_length,
                )
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

    def _fit(self, X, y, *, fit_pca=True):
        self._reset_fit()
        started = time.perf_counter()
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
            self.embedding_fit_time_ = time.perf_counter() - started
            pca_started = time.perf_counter()
            if fit_pca:
                effective_components = min(self.n_components, *raw.shape)
                if effective_components < self.n_components:
                    logger.info('Lowering PCA components from %s to %s.', self.n_components, effective_components)
                self.pca_ = PCA(n_components=effective_components, random_state=self.random_state).fit(raw)
                all_names = _unique_names(self._input_names_ + [f'lm_pca_{i}' for i in range(effective_components)])
                self.embedding_feature_names_ = np.asarray(all_names[self.n_features_in_ :], dtype=object)
            self.pca_fit_time_ = time.perf_counter() - pca_started if fit_pca else 0.0
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

    def fit_embed(self, X, y=None):
        """Fit context and return raw training embeddings, preserving cross-fitting.

        PCA is not fitted. Use embed for inference embeddings and let the consumer
        fit its projection on these training embeddings. Call fit or fit_transform
        before using transform or get_feature_names_out.
        """
        return self._fit(X, y, fit_pca=False)[1]

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
        check_is_fitted(self, attributes=['pca_'])
        frame = self._validate_X(X)
        raw = self._embed(frame)
        projected = self.pca_.transform(raw) if len(frame) else np.empty((0, self.pca_.n_components_))
        return self._output(frame, projected)

    def get_feature_names_out(self, input_features=None):
        """Return the stable output schema, including collision-free PCA names."""
        self._check_fitted()
        check_is_fitted(self, attributes=['pca_'])
        if input_features is not None and list(input_features) != list(self._columns_):
            raise ValueError('input_features must match the feature names seen during fit.')
        if self.append_original_features:
            return np.asarray(self._input_names_ + self.embedding_feature_names_.tolist(), dtype=object)
        return self.embedding_feature_names_.copy()
