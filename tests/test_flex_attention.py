import pytest
import torch

from progattn.config import ModelConfig
from progattn.model import GPT, Block
from progattn.programs import ProgramSpec, ProgramType

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="FlexAttention kernel test requires CUDA"
)


def flex_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=64,
        block_size=16,
        n_layer=1,
        n_head=4,
        n_embd=64,
        mlp_ratio=2,
        bias=True,
        dropout=0.0,
        tie_embeddings=True,
    )


def test_zero_strength_matches_flex_noop() -> None:
    device = torch.device("cuda")
    config = flex_config()
    programs = [ProgramSpec(0, 0, ProgramType.PREVIOUS_K, 1)]
    noop = GPT(config, condition="flex_noop", strict_flash=False).to(device)
    matched = GPT(
        config,
        condition="matched_program_prior",
        programs=programs,
        strict_flash=False,
    ).to(device)
    matched.load_state_dict(noop.state_dict())
    noop.prepare_attention(device)
    matched.prepare_attention(device)
    noop.set_prior_progress(0, 100)
    matched.set_prior_progress(0, 100)
    noop_compiled = torch.compile(noop, dynamic=False)
    matched_compiled = torch.compile(matched, dynamic=False)
    tokens = torch.randint(0, config.vocab_size, (2, config.block_size), device=device)

    noop_logits, noop_loss, _ = noop_compiled(tokens, tokens)
    matched_logits, matched_loss, _ = matched_compiled(tokens, tokens)

    assert noop_loss is not None and matched_loss is not None
    torch.testing.assert_close(matched_logits, noop_logits, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(matched_loss, noop_loss, atol=1e-6, rtol=1e-6)
    matched_loss.backward()
    block = matched.blocks[0]
    assert isinstance(block, Block)
    assert block.attn.raw_alpha.grad is not None
