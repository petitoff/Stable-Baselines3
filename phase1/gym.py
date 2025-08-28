import math
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


EPS = 1e-9


def _compute_indicators(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build an indicator feature matrix from OHLCV DataFrame.
    Expects columns: ['timestamp', 'open', 'high', 'low', 'close', 'volume'] (timestamp optional).
    Returns:
        features_df: DataFrame of engineered features (no NaNs)
        close_aligned: close prices aligned with features_df index
    """
    df = df.copy()

    # Ensure sorted by time if timestamp/datetime exists
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp")
    elif "datetime" in df.columns:
        df = df.sort_values("datetime")

    o = df["open"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)
    v = df["volume"].astype(float)

    # Log returns and rolling stats
    lr = (np.log(c / c.shift(1))).replace([np.inf, -np.inf], np.nan)
    mean5 = lr.rolling(5, min_periods=5).mean()
    std5 = lr.rolling(5, min_periods=5).std()
    mean20 = lr.rolling(20, min_periods=20).mean()
    std20 = lr.rolling(20, min_periods=20).std()

    # RSI(14) using Wilder's smoothing
    delta = c.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + EPS)
    rsi14 = 100.0 - (100.0 / (1.0 + rs))

    # MACD(12, 26, 9)
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - macd_signal

    # EMAs relative to price
    ema20 = c.ewm(span=20, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False).mean()
    ema20_rel = (ema20 / c) - 1.0
    ema50_rel = (ema50 / c) - 1.0

    # ATR(14) normalized
    prev_close = c.shift(1)
    tr = pd.concat(
        [(h - l).abs(), (h - prev_close).abs(), (l - prev_close).abs()], axis=1
    ).max(axis=1)
    atr14 = tr.ewm(span=14, adjust=False).mean()
    atr14_rel = atr14 / (c + EPS)

    # Stochastic %K(14), %D(3)
    low14 = l.rolling(14, min_periods=14).min()
    high14 = h.rolling(14, min_periods=14).max()
    denom = (high14 - low14).replace(0, np.nan)
    stoch_k = 100.0 * ((c - low14) / (denom + EPS))
    stoch_d = stoch_k.rolling(3, min_periods=3).mean()

    # Volume z-score(20)
    v_mean20 = v.rolling(20, min_periods=20).mean()
    v_std20 = v.rolling(20, min_periods=20).std()
    vol_z = (v - v_mean20) / (v_std20 + EPS)

    features_df = pd.DataFrame(
        {
            "lr": lr,
            "mean5": mean5,
            "std5": std5,
            "mean20": mean20,
            "std20": std20,
            "rsi14": rsi14,
            "macd": macd,
            "macd_signal": macd_signal,
            "macd_hist": macd_hist,
            "ema20_rel": ema20_rel,
            "ema50_rel": ema50_rel,
            "atr14_rel": atr14_rel,
            "stoch_k": stoch_k,
            "stoch_d": stoch_d,
            "vol_z": vol_z,
        },
        index=df.index,
    )

    features_df = features_df.replace([np.inf, -np.inf], np.nan).dropna()
    close_aligned = c.loc[features_df.index]
    return features_df, close_aligned


class TradingEnv(gym.Env):
    """
    Indicator-based trading environment for Stable-Baselines3 (Gymnasium API).
    - Continuous action: target exposure in [-1, 1] (short to long).
    - Observation: window of indicators + portfolio context (position, ROI).
    - Reward: step equity return (position * next_return - costs).
    - Episode ends:
        - Success when ROI >= target_roi (e.g., 20%).
        - Failure when equity <= (1 - stop_loss) * initial_balance (e.g., >20% drawdown).
        - Truncated at dataset end.
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        df: pd.DataFrame,
        window_size: int = 128,
        initial_balance: float = 10_000.0,
        fee_rate: float = 1e-3,
        slippage: float = 0.0,
        target_roi: float = 0.20,
        stop_loss: float = 0.20,
        normalize_features: bool = False,
        random_start: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.df = df.copy()
        self.window_size = int(window_size)
        self.initial_balance = float(initial_balance)
        self.fee_rate = float(fee_rate)
        self.slippage = float(slippage)
        self.target_roi = float(target_roi)
        self.stop_loss = float(stop_loss)
        self.normalize_features = bool(normalize_features)
        self.random_start = bool(random_start)

        # Build features and align closes
        features_df, close_aligned = _compute_indicators(self.df)
        self.base_features = features_df.to_numpy(dtype=np.float32)  # (N, F)
        self.close_prices = close_aligned.to_numpy(dtype=np.float32)  # (N,)
        self.timestamps = features_df.index  # may be DatetimeIndex or RangeIndex

        if self.normalize_features:
            mu = np.nanmean(self.base_features, axis=0)
            sigma = np.nanstd(self.base_features, axis=0) + 1e-8
            self.features = (self.base_features - mu) / sigma
        else:
            self.features = self.base_features

        self.num_rows, self.num_feat = self.features.shape
        if self.num_rows < self.window_size + 2:
            raise ValueError(
                f"Not enough data after indicator computation: "
                f"{self.num_rows} rows for window_size={self.window_size}."
            )

        # Observation: window of features + [position, roi] repeated across the window
        self.obs_feat_dim = self.num_feat + 2  # +2 for position and ROI
        self.obs_shape = (self.window_size * self.obs_feat_dim,)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=self.obs_shape, dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # State
        self.rng = np.random.default_rng(seed)
        self.position: float = 0.0
        self.equity: float = self.initial_balance
        self.n_trades: int = 0
        self._t: int = 0  # current pointer into features/close_prices

        # Cached last info (for render)
        self._last_info: Dict[str, Any] = {}

    def seed(self, seed: Optional[int] = None) -> None:
        self.rng = np.random.default_rng(seed)

    def _roi(self) -> float:
        return (self.equity - self.initial_balance) / (self.initial_balance + EPS)

    def _start_index(self) -> int:
        if self.random_start:
            # ensure we have at least one step after current for return computation
            return int(self.rng.integers(self.window_size, self.num_rows - 2))
        return self.window_size

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.seed(seed)

        start_idx = options.get("start_idx") if options else None
        if start_idx is not None:
            # clamp and ensure valid start
            start_idx = int(start_idx)
            start_idx = max(self.window_size, min(start_idx, self.num_rows - 2))
            self._t = start_idx
        else:
            self._t = self._start_index()

        self.position = 0.0
        self.equity = self.initial_balance
        self.n_trades = 0
        obs = self._build_observation()
        info = self._build_info(step_pnl=0.0, price=float(self.close_prices[self._t]))
        self._last_info = info
        return obs, info

    def _build_observation(self) -> np.ndarray:
        # Window features [t - window + 1 : t]
        start = self._t - self.window_size + 1
        end = self._t + 1
        window_feats = self.features[start:end, :]  # (W, F)

        # Append position and ROI as constant columns across window
        pos_col = np.full((self.window_size, 1), self.position, dtype=np.float32)
        roi_col = np.full((self.window_size, 1), self._roi(), dtype=np.float32)
        window_with_ctx = np.concatenate([window_feats, pos_col, roi_col], axis=1)

        return window_with_ctx.reshape(-1).astype(np.float32)

    def _build_info(self, step_pnl: float, price: float) -> Dict[str, Any]:
        info = {
            "equity": float(self.equity),
            "position": float(self.position),
            "roi": float(self._roi()),
            "step_pnl": float(step_pnl),
            "price": float(price),
            "n_trades": int(self.n_trades),
            "t": int(self._t),
            "timestamp": self.timestamps[self._t]
            if self._t < len(self.timestamps)
            else None,
        }
        return info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Current time t, we will apply action and realize PnL over [t -> t+1]
        t = self._t
        if t >= self.num_rows - 1:
            # No next price to compute return; truncate
            obs = self._build_observation()
            info = self._build_info(step_pnl=0.0, price=float(self.close_prices[t]))
            return obs, 0.0, False, True, info

        # Parse and clip action
        target_pos = float(np.clip(action[0], -1.0, 1.0))

        # Transaction costs on position change (immediate)
        delta = abs(target_pos - self.position)
        trade_cost_rate = self.fee_rate * delta + self.slippage * delta

        price_now = float(self.close_prices[t])
        price_next = float(self.close_prices[t + 1])
        r = (price_next / (price_now + EPS)) - 1.0  # next interval return

        # Update position (filled at current)
        self.position = target_pos

        # Equity update
        step_return = self.position * r - trade_cost_rate
        prev_equity = self.equity
        self.equity = self.equity * (1.0 + step_return)
        step_pnl = self.equity - prev_equity

        # Book trade count if we changed position
        if delta > 1e-6:
            self.n_trades += 1

        # Advance time
        self._t = t + 1

        # Terminal conditions
        roi = self._roi()
        hit_target = roi >= self.target_roi
        breached_dd = self.equity <= (1.0 - self.stop_loss) * self.initial_balance

        terminated = bool(hit_target or breached_dd)
        truncated = bool(self._t >= self.num_rows - 1)

        # Dense reward: step equity return (position*r - costs)
        reward = float(step_return)

        # Terminal bonus/penalty shaping
        if terminated:
            if hit_target:
                reward += 1.0  # success bonus
            elif breached_dd:
                reward -= 1.0  # failure penalty

        obs = self._build_observation()
        info = self._build_info(step_pnl=step_pnl, price=price_next)
        info["target_hit"] = bool(hit_target)
        info["drawdown_breached"] = bool(breached_dd)
        self._last_info = info

        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        info = self._last_info
        if not info:
            return
        print(
            f"t={info['t']} price={info['price']:.2f} pos={info['position']:+.2f} "
            f"equity={info['equity']:.2f} ROI={info['roi']*100:.2f}% pnl={info['step_pnl']:+.2f} "
            f"trades={info['n_trades']}"
        )


def make_env_from_csv(
    csv_path: str,
    window_size: int = 128,
    initial_balance: float = 10_000.0,
    fee_rate: float = 1e-3,
    slippage: float = 0.0,
    target_roi: float = 0.20,
    stop_loss: float = 0.20,
    normalize_features: bool = False,
    random_start: bool = True,
    seed: Optional[int] = None,
) -> TradingEnv:
    """
    Helper to build TradingEnv from your CSV (output of phase1/data.py).
    The CSV should contain: timestamp, open, high, low, close, volume (and optionally datetime).
    """
    df = pd.read_csv(csv_path)
    # Basic sanity: ensure required columns
    required = {"open", "high", "low", "close", "volume"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    return TradingEnv(
        df=df,
        window_size=window_size,
        initial_balance=initial_balance,
        fee_rate=fee_rate,
        slippage=slippage,
        target_roi=target_roi,
        stop_loss=stop_loss,
        normalize_features=normalize_features,
        random_start=random_start,
        seed=seed,
    )