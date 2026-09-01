"""PPO training entry point for the odor search RL policy.

Prerequisites (run inside the OdorSim venv after `source setup/activate.sh`):
    pip install gymnasium stable-baselines3

Usage:
    python scripts/train_ppo.py --seed 42 --total-timesteps 200000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train odor search policy with PPO.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--room-half", type=float, default=4.0)
    parser.add_argument("--max-episode-steps", type=int, default=1000)
    parser.add_argument(
        "--scenario",
        type=str,
        default=str(PROJECT_ROOT / "odorsim_scenarios" / "warehouse_8x8" / "environment_configurations" / "config1"),
    )
    parser.add_argument(
        "--env-config",
        type=str,
        default=str(PROJECT_ROOT / "config" / "warehouse_empty_8x8.yaml"),
    )
    parser.add_argument("--save-dir", type=str, default=str(PROJECT_ROOT / "outputs" / "rl"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        import gymnasium as gym
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import CheckpointCallback
        from stable_baselines3.common.monitor import Monitor
    except ImportError as exc:
        print("[ERROR] Missing dependency:", exc)
        print("Please run: pip install gymnasium stable-baselines3")
        return 1

    from odor_sim.bridge.gaden_bridge import GadenBridge
    from odor_sim.runtime.gaden_server import GadenServerManager
    from src.rl.env_wrapper import OdorSearchRLEnv
    from src.utils import load_yaml

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PPO Training for Odor Search (GADEN backend)")
    print("=" * 60)

    server: GadenServerManager | None = None
    bridge: GadenBridge | None = None

    try:
        print("[1/3] Starting GADEN server...")
        server = GadenServerManager(
            scenario_path=args.scenario,
            scene_id="scene1",
            step_on_timer=False,
            publish_markers=False,
            log_dir=save_dir / "gaden_logs",
        )
        server.start(timeout=120.0, kill_stale=True)

        print("[2/3] Connecting GadenBridge...")
        bridge = GadenBridge(node_name="rl_bridge")
        if not bridge.wait_for_server(timeout=60.0):
            raise RuntimeError("GADEN service not available")

        print("[3/3] Building RL env...")
        env_cfg = load_yaml(args.env_config)
        env = OdorSearchRLEnv(
            scenario_path=args.scenario,
            env_config=env_cfg,
            bridge=bridge,
            max_steps=args.max_episode_steps,
            room_half=args.room_half,
            seed=args.seed,
        )
        env = Monitor(env, filename=str(save_dir / "monitor.csv"))

        policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
        model = PPO(
            "MlpPolicy",
            env,
            device="cpu",
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=1,
            seed=args.seed,
            policy_kwargs=policy_kwargs,
            tensorboard_log=str(save_dir / "tensorboard"),
        )

        checkpoint_cb = CheckpointCallback(
            save_freq=50_000,
            save_path=str(save_dir / "checkpoints"),
            name_prefix="ppo_odor_search",
        )

        print("Starting training...")
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=checkpoint_cb,
            progress_bar=True,
        )

        final_path = save_dir / "ppo_odor_search_final"
        model.save(final_path)
        print(f"Model saved to {final_path}")
        return 0

    except KeyboardInterrupt:
        print("\n[Interrupted]")
        return 130
    except Exception as exc:
        print(f"\n[ERROR] {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        if bridge is not None:
            try:
                bridge.close()
            except Exception:
                pass
        if server is not None:
            try:
                server.stop()
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())
