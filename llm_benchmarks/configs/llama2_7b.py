"""
Llama-2-7B configuration matching the official architecture

This matches Meta's Llama-2-7B model specification for accurate benchmarking.
Using standard config ensures results are comparable to published benchmarks.
"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class Llama2Config:
    """Configuration for Llama-2-7B model"""

    # Model architecture
    hidden_size: int = 4096              # Model dimension
    num_hidden_layers: int = 32          # Number of transformer layers
    num_attention_heads: int = 32        # Number of attention heads
    num_key_value_heads: int = 32        # KV heads (32 for Llama-2-7B, 8 for 70B GQA)
    intermediate_size: int = 11008       # FFN intermediate dimension

    # Attention configuration
    max_position_embeddings: int = 4096  # Maximum sequence length
    rope_theta: float = 10000.0          # RoPE base frequency
    attention_dropout: float = 0.0

    # Vocabulary
    vocab_size: int = 32000
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    # Compute
    use_cache: bool = True               # Enable KV caching for generation

    @property
    def head_dim(self) -> int:
        """Dimension per attention head"""
        return self.hidden_size // self.num_attention_heads

    @property
    def num_kv_heads(self) -> int:
        """Number of key-value heads (for GQA support)"""
        return self.num_key_value_heads

    def __post_init__(self):
        # Validate configuration
        assert self.hidden_size % self.num_attention_heads == 0, \
            f"hidden_size ({self.hidden_size}) must be divisible by num_attention_heads ({self.num_attention_heads})"

        assert self.num_attention_heads % self.num_key_value_heads == 0, \
            f"num_attention_heads ({self.num_attention_heads}) must be divisible by num_key_value_heads ({self.num_key_value_heads})"

# Pre-defined configs for common Llama variants
LLAMA2_7B = Llama2Config()

LLAMA2_13B = Llama2Config(
    hidden_size=5120,
    num_hidden_layers=40,
    num_attention_heads=40,
    num_key_value_heads=40,  # Llama-2-13B uses full attention (no GQA)
    intermediate_size=13824,
)

LLAMA2_70B = Llama2Config(
    hidden_size=8192,
    num_hidden_layers=80,
    num_attention_heads=64,
    num_key_value_heads=8,  # GQA: 64 query heads, 8 KV heads
    intermediate_size=28672,
)

def get_config(model_name: str = "7b") -> Llama2Config:
    """Get configuration by model size"""
    configs = {
        "7b": LLAMA2_7B,
        "13b": LLAMA2_13B,
        "70b": LLAMA2_70B,
    }
    return configs.get(model_name.lower(), LLAMA2_7B)
