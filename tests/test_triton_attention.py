import pytest
import torch

from progattn.config import ModelConfig
from progattn.model import GPT, Block
from progattn.programs import ProgramSpec, ProgramType

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton attention test requires CUDA"
)


def kernel_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=64,
        block_size=32,
        n_layer=2,
        n_head=4,
        n_embd=128,
        mlp_ratio=2,
        bias=True,
        dropout=0.0,
        tie_embeddings=True,
    )


def programs() -> list[ProgramSpec]:
    return [
        ProgramSpec(0, 0, ProgramType.LOCAL_WINDOW, 4),
        ProgramSpec(0, 2, ProgramType.PREVIOUS_K, 1),
        ProgramSpec(1, 1, ProgramType.LOCAL_WINDOW, 8),
    ]


def build(condition: str, kernel: str, device: torch.device) -> GPT:
    model = GPT(
        kernel_config(),
        condition=condition,
        programs=programs(),
        initial_alpha=4.0,
        strict_flash=False,
        kernel=kernel,
    )
    return model.to(device)


@pytest.mark.parametrize(
    "condition", ["matched_program_prior", "incorrect_program_prior", "flex_noop"]
)
def test_triton_matches_flex_path(condition: str) -> None:
    """The kernel must be a drop-in for FlexAttention, prior and all.

    Both paths are exact attention, so any disagreement beyond bf16/tf32 rounding
    would mean the fused bonus or its gradient is wrong somewhere the standalone
    kernel check did not reach - a per-layer program table, or the warmup scale.
    """
    device = torch.device("cuda")
    flex = build(condition, "flex", device)
    triton_model = build(condition, "triton", device)
    triton_model.load_state_dict(flex.state_dict())
    for model in (flex, triton_model):
        model.prepare_attention(device)
        model.set_prior_progress(50, 100)
    tokens = torch.randint(0, 64, (2, 32), device=device)

    flex_logits, flex_loss, _ = flex(tokens, tokens)
    triton_logits, triton_loss, _ = triton_model(tokens, tokens)
    assert flex_loss is not None and triton_loss is not None

    torch.testing.assert_close(triton_logits, flex_logits, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(triton_loss, flex_loss, atol=1e-4, rtol=1e-4)

    flex_loss.backward()
    triton_loss.backward()
    for (name, left), (_, right) in zip(
        flex.named_parameters(), triton_model.named_parameters(), strict=True
    ):
        if left.grad is None:
            assert right.grad is None, name
            continue
        assert right.grad is not None, name
        torch.testing.assert_close(right.grad, left.grad, atol=2e-3, rtol=2e-3)


def test_prior_strength_reaches_alpha_gradient() -> None:
    """A layer owning a selected head must accumulate a gradient on its own alpha."""
    device = torch.device("cuda")
    model = build("matched_program_prior", "triton", device)
    model.prepare_attention(device)
    model.set_prior_progress(50, 100)
    tokens = torch.randint(0, 64, (2, 32), device=device)
    _, loss, _ = model(tokens, tokens)
    assert loss is not None
    loss.backward()
    for index in (0, 1):
        block = model.blocks[index]
        assert isinstance(block, Block)
        grad = block.attn.raw_alpha.grad
        assert grad is not None
        assert torch.isfinite(grad).all()
        assert grad.abs().sum() > 0.0


def test_layer_without_programs_skips_the_prior() -> None:
    """A layer owning no selected head must be untouched by the prior."""
    device = torch.device("cuda")
    model = build("matched_program_prior", "triton", device)
    block = model.blocks[0]
    assert isinstance(block, Block)
    assert block.attn.applies_prior
    lonely = GPT(
        kernel_config(),
        condition="matched_program_prior",
        programs=[ProgramSpec(0, 0, ProgramType.LOCAL_WINDOW, 4)],
        initial_alpha=4.0,
        strict_flash=False,
        kernel="triton",
    ).to(device)
    second = lonely.blocks[1]
    assert isinstance(second, Block)
    assert not second.attn.applies_prior
