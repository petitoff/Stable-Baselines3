import argparse
from pathlib import Path
from typing import Optional, List

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy

from phase1.data import fetch_bybit_ohlcv
from phase1.gym import TradingEnv


def _find_latest_model(explicit_path: Optional[str]) -> Path:
    if explicit_path:
        p = Path(explicit_path)
        if not p.exists():
            raise FileNotFoundError(f"Model path not found: {p}")
        return p

    models_root = Path("models")
    if not models_root.exists():
        raise FileNotFoundError("No 'models/' directory found. Run training first.")

    # Search for any .zip files under models/ (includes W&B checkpoints and final saves)
    candidates: List[Path] = list(models_root.rglob("*.zip"))
    if not candidates:
        raise FileNotFoundError("No model .zip files found under 'models/'. Run training first.")

    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return latest


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO trading agent")
    parser.add_argument("--model-path", type=str, default=None, help="Path to the saved .zip model. If omitted, auto-detect the latest model under models/")
    parser.add_argument("--symbol", type=str, default="BTC/USDT")
    parser.add_argument("--timeframe", type=str, default="1m")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--normalize-features", action="store_true")
    parser.add_argument("--random-start", action="store_true", help="Use random episode starts for evaluation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=5, help="Number of eval episodes")
    args = parser.parse_args()

    # Fetch evaluation data
    df = fetch_bybit_ohlcv(symbol=args.symbol, timeframe=args.timeframe, days=args.days)

    # Build evaluation environment (deterministic by default)
    env_fn = lambda: TradingEnv(
        df=df,
        window_size=args.window_size,
        normalize_features=args.normalize_features,
        random_start=args.random_start,
        seed=args.seed,
    )
    venv = VecMonitor(DummyVecEnv([env_fn]))

    # Load model
    model_path = _find_latest_model(args.model_path)
    print(f"Loading model from: {model_path}")
    model = PPO.load(str(model_path))

    # Evaluate
    mean_reward, std_reward = evaluate_policy(
        model,
        venv,
        n_eval_episodes=args.episodes,
        deterministic=True,
        render=False,
    )
    print("==== Evaluation Results ====")
    print(f"Episodes: {args.episodes}")
    print(f"Mean reward: {mean_reward:.6f} +/- {std_reward:.6f}")


if __name__ == "__main__":
    main()
