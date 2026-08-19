from progattn.config import load_config
from progattn.train import learning_rate


def test_learning_rate_schedule() -> None:
    config = load_config("configs/pilot_gpt2_small.yaml")

    assert learning_rate(config, 0) == 0.0
    assert learning_rate(config, config.training.warmup_tokens) == 0.0005
    assert learning_rate(config, config.training.train_tokens) == 0.00005
