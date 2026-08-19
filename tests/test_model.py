import torch

from progattn.config import ModelConfig, load_config
from progattn.model import GPT
from progattn.programs import ProgramSpec, ProgramType


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


def test_gpt2_small_parameter_count() -> None:
    config = load_config("configs/pilot_gpt2_small.yaml")
    model = GPT(config.model, condition="flash_baseline", strict_flash=False)

    assert model.parameter_count() == 124_046_736
