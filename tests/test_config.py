from progattn.config import load_config


def test_pilot_shape() -> None:
    config = load_config("configs/pilot_gpt2_small.yaml")

    assert config.sequences_per_step == 256
    assert config.accumulation_steps == 8
    assert config.total_steps == 3815
