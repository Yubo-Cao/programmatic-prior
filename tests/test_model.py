import torch

from progattn.config import ModelConfig, load_config
from progattn.model import GPT, Block, CausalSelfAttention
from progattn.programs import ProgramSpec, ProgramType


def attention(model: GPT, layer: int) -> CausalSelfAttention:
    block = model.blocks[layer]
    assert isinstance(block, Block)
    return block.attn


def tiny_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=64,
        block_size=16,
        n_layer=2,
        n_head=4,
        n_embd=32,
        mlp_ratio=2,
        bias=True,
        dropout=0.0,
        tie_embeddings=True,
    )


def test_dense_forward_and_loss() -> None:
    model = GPT(tiny_config(), condition="flash_baseline", strict_flash=False)
    tokens = torch.randint(0, 64, (2, 16))

    logits, loss, attentions = model(
        tokens,
        tokens,
        force_dense=True,
        return_attentions=True,
    )

    assert logits.shape == (2, 16, 64)
    assert loss is not None and torch.isfinite(loss)
    assert attentions is not None and len(attentions) == 2
    assert attentions[0].shape == (2, 4, 16, 16)


def test_zero_warmup_matches_no_prior() -> None:
    config = tiny_config()
    program = ProgramSpec(
        layer=0,
        head=0,
        program_type=ProgramType.PREVIOUS_K,
        parameter=1,
    )
    baseline = GPT(config, condition="flash_baseline", strict_flash=False)
    matched = GPT(
        config,
        condition="matched_program_prior",
        programs=[program],
        strict_flash=False,
    )
    matched.load_state_dict(baseline.state_dict())
    matched.set_prior_progress(0, warmup_tokens=100)
    tokens = torch.randint(0, 64, (2, 16))

    baseline_logits, _, _ = baseline(tokens, force_dense=True)
    matched_logits, _, _ = matched(tokens, force_dense=True)

    torch.testing.assert_close(matched_logits, baseline_logits)


def test_prior_applies_only_to_layers_owning_a_selected_head() -> None:
    config = tiny_config()
    model = GPT(
        config,
        condition="matched_program_prior",
        programs=[ProgramSpec(0, 0, ProgramType.PREVIOUS_K, 1)],
        strict_flash=False,
    )

    assert attention(model, 0).applies_prior
    assert not attention(model, 1).applies_prior
    # Every layer still exposes a trainable raw_alpha, so the optimizer parameter set
    # and therefore checkpoint resume stay unchanged by the gate.
    assert all(attention(model, layer).raw_alpha.requires_grad for layer in (0, 1))


def test_skipping_program_free_layers_is_numerically_free() -> None:
    """The gate must not change any score: a program-free layer adds alpha * 0."""
    config = tiny_config()
    program = ProgramSpec(0, 0, ProgramType.PREVIOUS_K, 1)
    gated = GPT(
        config,
        condition="matched_program_prior",
        programs=[program],
        strict_flash=False,
    )
    forced = GPT(
        config,
        condition="matched_program_prior",
        programs=[program],
        strict_flash=False,
    )
    forced.load_state_dict(gated.state_dict())
    # Make the program-free layer claim a program so it builds the modifier anyway.
    attention(forced, 1).has_programs = True
    for model in (gated, forced):
        model.set_prior_progress(1000, warmup_tokens=100)
    tokens = torch.randint(0, 64, (2, 16))

    gated_logits, _, _ = gated(tokens, force_dense=True)
    forced_logits, _, _ = forced(tokens, force_dense=True)

    assert attention(forced, 1).applies_prior
    torch.testing.assert_close(gated_logits, forced_logits, atol=0.0, rtol=0.0)


def test_gpt2_small_parameter_count() -> None:
    config = load_config("configs/pilot_gpt2_small.yaml")
    model = GPT(config.model, condition="flash_baseline", strict_flash=False)

    assert model.parameter_count() == 124_046_736
