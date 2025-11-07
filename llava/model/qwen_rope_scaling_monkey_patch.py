import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union
from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding, Qwen2Attention, apply_rotary_pos_emb, repeat_kv
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.utils import logging
import types

logger = logging.get_logger(__name__)

class Qwen2RotaryEmbeddingWithScaling(nn.Module):
    def __init__(
        self,
        dim=None,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        rope_type="default",
        config: Optional[Qwen2Config] = None,
    ):
        super().__init__()

        # Handle both config-based and direct parameter initialization
        self.rope_kwargs = {}
        if config is None:
            logger.warning_once(
                "`Qwen2RotaryEmbedding` can now be fully parameterized by passing the model config through the "
                "`config` argument. All other arguments will be removed in v4.46"
            )
            self.rope_kwargs = {
                "rope_type": rope_type,
                "factor": scaling_factor,
                "dim": dim,
                "base": base,
                "max_position_embeddings": max_position_embeddings,
            }
            self.rope_type = rope_type
            self.max_seq_len_cached = max_position_embeddings
            self.original_max_seq_len = max_position_embeddings
        else:
            # BC: "rope_type" was originally "type"
            if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
                self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
            else:
                self.rope_type = "default"
            print(f"Init Rope Embedding with rope type: {self.rope_type}")
            self.max_seq_len_cached = config.max_position_embeddings
            self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, **self.rope_kwargs)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _dynamic_frequency_update(self, seq_len, device, dtype):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len, **self.rope_kwargs
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)
            self.max_seq_len_cached = seq_len
            self._set_cos_sin_cache(seq_len=seq_len, device=device, dtype=dtype)

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len
            self._set_cos_sin_cache(seq_len=seq_len, device=device, dtype=dtype)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        # t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)
        #
        # freqs = torch.outer(t, self.inv_freq)
        # # Different from paper, but it uses a different permutation in order to obtain the same calculation
        # emb = torch.cat((freqs, freqs), dim=-1)
        # self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        # self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

        position_ids = torch.arange(seq_len, device=device, dtype=torch.int64).type_as(self.inv_freq)
        inv_freq_expanded = self.inv_freq[:, None].float()
        position_ids_expanded = position_ids[None, :].float()

        # Force float32 computation
        device_type = device if isinstance(device, str) and device != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(0, 1)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Apply scaling for advanced RoPE types
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        self.register_buffer("cos_cached", cos.to(dtype=dtype), persistent=False)
        self.register_buffer("sin_cached", sin.to(dtype=dtype), persistent=False)

    @torch.no_grad()
    def forward(self, x, seq_len=None):
        # Handle both the old interface (seq_len) and new interface (position_ids)
        if seq_len is None:
            seq_len = x.shape[-2]

        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(seq_len, device=x.device, dtype=x.dtype)

        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


def new_init(self, config: Qwen2Config, layer_idx: Optional[int] = None):
    super(Qwen2Attention, self).__init__()
    self.config = config
    self.layer_idx = layer_idx
    logger.info(f"Custom initialization for {self.__class__.__name__} at layer {layer_idx}")
    if layer_idx is None:
        logger.warning_once(
            f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
            "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
            "when creating this class."
        )

    self.hidden_size = config.hidden_size
    self.num_heads = config.num_attention_heads
    self.head_dim = self.hidden_size // self.num_heads
    self.num_key_value_heads = config.num_key_value_heads
    self.num_key_value_groups = self.num_heads // self.num_key_value_heads
    self.max_position_embeddings = config.max_position_embeddings
    self.rope_theta = config.rope_theta
    self.is_causal = True
    self.attention_dropout = config.attention_dropout

    if (self.head_dim * self.num_heads) != self.hidden_size:
        raise ValueError(
            f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
            f" and `num_heads`: {self.num_heads})."
        )
    self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
    self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
    self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
    self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    self.rotary_emb = Qwen2RotaryEmbeddingWithScaling(config=self.config)


def enable_rope_scaling_in_qwen2():
    """
    Monkey patch to enable mRoPE++ for Qwen2 models.
    Apply this patch before loading the model.
    """
    import logging

    logger = logging.getLogger(__name__)

    # Store original class for potential restoration
    original_rope_class = Qwen2RotaryEmbedding
    original_init = Qwen2Attention.__init__

    # Replace the original class with our modified version
    try:
        from transformers.models.qwen2 import modeling_qwen2

        modeling_qwen2.Qwen2RotaryEmbedding = Qwen2RotaryEmbeddingWithScaling
        Qwen2Attention.__init__ = new_init
        logger.info("Successfully enabled rope with scaling for Qwen2")
    except ImportError:
        logger.error("Failed to patch Qwen2: transformers library not found")
    except Exception as e:
        logger.error(f"Failed to patch Qwen2: {str(e)}")
        # Restore original class if patching fails
        modeling_qwen2.Qwen2RotaryEmbedding = original_rope_class
        Qwen2Attention.__init__ = original_init

