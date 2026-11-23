"""Install the RoboTTT adapter and register its Marvin simulation dimensions."""

import argparse
import json
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile


ROBOT = "marvin_dual_gripper"
DIMENSIONS = {"arm_dim": [7, 7], "ee_dim": [1, 1]}
SIM = "marvin_isaac"


def install(checkout):
    checkout = Path(checkout).expanduser().resolve()
    training_registry = checkout / "utils/robot/_robot_info.json"
    if (checkout.name != "XPolicyLab" or not training_registry.is_file()
            or not (checkout / "setup_policy_server.py").is_file()):
        raise ValueError("Provide a standard XPolicyLab checkout named XPolicyLab")
    env_cfg = checkout.parent / "env_cfg"
    runtime_registry = env_cfg / "robot/_robot_info.json"
    updates = []
    for path in (training_registry, runtime_registry):
        registry = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        if not isinstance(registry, dict):
            raise ValueError(f"Expected a robot registry object in {path}")
        if registry.get(ROBOT, DIMENSIONS) != DIMENSIONS:
            raise ValueError(f"{path} has an incompatible definition for {ROBOT}")
        if ROBOT not in registry:
            registry[ROBOT] = DIMENSIONS
            updates.append((path, json.dumps(registry, indent=4, ensure_ascii=False) + "\n"))
    for path, data in (
        (env_cfg / f"{ROBOT}.yml", f"config:\n  robot: {ROBOT}\n  sim: {SIM}\n"),
        (env_cfg / "sim" / f"{SIM}.yml", "scene:\n  num_envs: 1\n"),
    ):
        if not path.exists():
            updates.append((path, data))

    for path, data in updates:
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=path.parent, mode="w", encoding="utf-8", delete=False) as stream:
            staged = Path(stream.name)
            stream.write(data)
        try:
            staged.chmod(path.stat().st_mode & 0o777 if path.exists() else 0o644)
            staged.replace(path)
        finally:
            staged.unlink(missing_ok=True)

    source = Path(__file__).resolve().parent
    destination = checkout / "policy/RoboTTT"
    if source != destination:
        destination.mkdir(parents=True, exist_ok=True)
        for path in source.iterdir():
            if path.is_file():
                shutil.copy2(path, destination / path.name)
    print(f"Installed RoboTTT at {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path)
    options = parser.parse_args()
    try:
        install(options.checkout)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")
