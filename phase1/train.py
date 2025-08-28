from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from phase1.data import fetch_bybit_ohlcv
from phase1.gym import TradingEnv
import wandb
from wandb.integration.sb3 import WandbCallback
from pathlib import Path

# Fetch market data via CCXT (Bybit)
df = fetch_bybit_ohlcv(symbol="BTC/USDT", timeframe="1m", days=30)

# Build environment directly from DataFrame
env_fn = lambda: TradingEnv(df=df, window_size=128, normalize_features=False, random_start=True)
venv = VecMonitor(DummyVecEnv([env_fn]))

# Init Weights & Biases run
run = wandb.init(
    project="phase1-sb3",
    config={
        "algo": "PPO",
        "policy": "MlpPolicy",
        "total_timesteps": 200_000,
        "symbol": "BTC/USDT",
        "timeframe": "1m",
        "days": 30,
        "window_size": 128,
    },
    sync_tensorboard=True,  # Sync TensorBoard metrics to W&B
)

# Directory for TensorBoard logs (grouped by W&B run id)
tb_log_dir = Path("tb") / (run.id if run else "default")
tb_log_dir.mkdir(parents=True, exist_ok=True)

model = PPO(
    "MlpPolicy",
    venv,
    verbose=1,
    tensorboard_log=str(tb_log_dir),  # Enable TensorBoard logging for W&B sync
)

# Directory for saving models (grouped by W&B run id)
models_dir = Path("models") / (run.id if run else "default")
models_dir.mkdir(parents=True, exist_ok=True)

# Train with periodic checkpointing to disk via W&B callback
model.learn(
    total_timesteps=200_000,
    callback=WandbCallback(
        model_save_path=str(models_dir),
        model_save_freq=10_000,
        verbose=2,
    ),
    log_interval=10,  # Log metrics every 10 environment steps
)

# Save final model
final_path = models_dir / "ppo_trading_env"
model.save(str(final_path))
print(f"Saved final model to {final_path}.zip")