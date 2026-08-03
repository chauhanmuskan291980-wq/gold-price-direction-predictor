"""

Responsibilities:
1.Read YAML using yaml.safe_load
2. check that required sections exist.
3.Reject invalid values.

"""

from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path(
    "config/signal_validation_gate.yaml"
)


def load_gate_config(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Gate configuration not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            "Gate configuration must be a YAML mapping."
        )

    validate_gate_config(config)
    return config


def validate_gate_config(
    config: dict[str, Any],
) -> None:
    required_sections = {
        "gate",
        "sample",
        "walk_forward",
        "permutation",
        "bootstrap",
        "concentration",
        "verdict",
    }

    missing = required_sections - set(config)

    if missing:
        raise ValueError(
            f"Missing configuration sections: "
            f"{sorted(missing)}"
        )

    if config["sample"]["minimum_trades"] <= 0:
        raise ValueError(
            "minimum_trades must be positive."
        )

    if config["permutation"]["repetitions"] <= 0:
        raise ValueError(
            "Permutation repetitions must be positive."
        )

    if config["bootstrap"]["repetitions"] <= 0:
        raise ValueError(
            "Bootstrap repetitions must be positive."
        )


"""
The gate has one trusted configuration source.
""" 