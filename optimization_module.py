# optimization_module.py
# Production-ready optimization module for AI Stock Analyzer
# Replace naive algorithms with ML-enhanced versions

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS & DATA CLASSES
# ============================================================================

class RegimeType(Enum):
    """Market regime classification"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"


@dataclass
class TrendAnalysis:
    """Trend strength analysis result"""
    strength: int  # 0-100
    regime: RegimeType
    direction: str  # 'up', 'down', 'neutral'
    confidence: float  # 0-1
    details: Dict


@dataclass
class SRLevels:
    """Support/Resistance levels with strength"""
    supports: List[Tuple[float, int]]  # (price, touches)
    resistances: List[Tuple[float, int]]
    nearest_support: Optional[float]
    nearest_resistance: Optional[float]


# ============================================================================
# 1. TREND STRENGTH CALCULATION (ML-Enhanced)
# ============================================================================

def calculate_trend_strength_enhanced(
    df: pd.DataFrame,
    lookback_periods: int = 20
) -> TrendAnalysis:
    """
    ML-enhanced trend strength calculation with regime detection.
    
    Replaces: naive binary score ±20 calculation
    
    Returns: TrendAnalysis with strength 0-100
    """
    if df is None or df.empty or len(df) < lookback_periods:
        logger.warning("Insufficient data for trend analysis")
        return TrendAnalysis(
            strength=50, regime=RegimeType.RANGING,
            direction='neutral', confidence=0.0, details={}
        )
    
    recent = df.iloc[-1].copy()
    close = recent['Close']
    
    # ========== Distance-based trend (normalized by ATR) ==========
    atr14 = recent.get('atr14', close * 0.02)
    if atr14 <= 0:
        atr14 = close * 0.02
    
    ma50 = recent.get('ma50')
    ma200 = recent.get('ma200')
    
    dist_score = 50
    if not pd.isna(ma50):
        ma50_dist = (close - ma50) / atr14
        dist_score += np.clip(ma50_dist * 8, -30, 30)
    
    if not pd.isna(ma200):
        ma200_dist = (close - ma200) / atr14
        dist_score += np.clip(ma200_dist * 5, -20, 20)
    
    dist_score = np.clip(dist_score, 0, 100)
    
    # ========== Momentum confirmation (RSI + MACD) ==========
    rsi = recent.get('rsi14', 50)
    momentum_score = 50
    
    if rsi > 65:
        momentum_score = 75
    elif rsi > 55:
        momentum_score = 60
    elif rsi < 35:
        momentum_score = 25
    elif rsi < 45:
        momentum_score = 40
    
    # MACD confirmation
    macd = recent.get('macd')
    macd_signal = recent.get('macd_signal')
    if not pd.isna(macd) and not pd.isna(macd_signal):
        if macd > macd_signal:
            momentum_score += 5
        else:
            momentum_score -= 5
    
    momentum_score = np.clip(momentum_score, 0, 100)
    
    # ========== Volatility adjustment ==========
    vol_ratio = recent.get('Volume', 1) / (recent.get('vol20', 1) + 1e-9)
    vol_factor = 1.0
    if vol_ratio > 2.0:
        vol_factor = 0.85  # High vol = less reliable
    elif vol_ratio < 0.5:
        vol_factor = 0.9  # Low vol = slightly less reliable
    
    # ========== Regime detection ==========
    atr_to_price = atr14 / close
    sma_slope = _calculate_ma_slope(df, 50, 5)  # MA50 slope over 5 bars
    
    if atr_to_price > 0.025:  # > 2.5% ATR = volatile
        regime = RegimeType.VOLATILE
    elif sma_slope > 0.002 and close > ma50:  # Strong uptrend
        regime = RegimeType.TRENDING_UP
    elif sma_slope < -0.002 and close < ma50:  # Strong downtrend
        regime = RegimeType.TRENDING_DOWN
    else:
        regime = RegimeType.RANGING
    
    # ========== Regime bonus/penalty ==========
    regime_factor = 1.0
    if regime == RegimeType.TRENDING_UP:
        regime_factor = 1.1
    elif regime == RegimeType.TRENDING_DOWN:
        regime_factor = 0.9
    elif regime == RegimeType.VOLATILE:
        regime_factor = 0.85
    
    # ========== Combine all factors ==========
    strength = (
        dist_score * 0.45 +
        momentum_score * 0.35 +
        50 * 0.20  # Regime contribution
    ) * vol_factor * regime_factor
    
    strength = int(np.clip(strength, 0, 100))
    
    # Direction determination
    if strength > 65:
        direction = 'up'
        confidence = (strength - 65) / 35
    elif strength < 35:
        direction = 'down'
        confidence = (35 - strength) / 35
    else:
        direction = 'neutral'
        confidence = 0.5 - abs(strength - 50) / 100
    
    return TrendAnalysis(
        strength=strength,
        regime=regime,
        direction=direction,
        confidence=confidence,
        details={
            'distance_score': dist_score,
            'momentum_score': momentum_score,
            'vol_factor': vol_factor,
            'regime_factor': regime_factor,
            'ma50_distance_atr': (close - ma50) / atr14 if not pd.isna(ma50) else None,
            'rsi': rsi
        }
    )


def _calculate_ma_slope(df: pd.DataFrame, ma_period: int, lookback: int) -> float:
    """Calculate MA slope to measure trend direction"""
    if f'ma{ma_period}' not in df.columns or len(df) < lookback + 1:
        return 0.0
    
    ma_col = f'ma{ma_period}'
    recent_ma = df[ma_col].iloc[-1]
    old_ma = df[ma_col].iloc[-(lookback + 1)]
    
    if pd.isna(recent_ma) or pd.isna(old_ma):
        return 0.0
    
    return (recent_ma - old_ma) / old_ma if old_ma != 0 else 0.0


# ============================================================================
# 2. BAYESIAN PROBABILITY CALCULATION
# ============================================================================

def calculate_bayesian_probability(
    trend_strength: int,
    trend_direction: str,
    rsi: float,
    news_score: int,
    regime: RegimeType,
    historical_win_rate: float = 0.55,
    prior_p_bullish: float = 0.5
) -> Tuple[float, Dict]:
    """
    Bayesian probability calculation for up move.
    
    Replaces: linear probability formula
    
    Returns: (probability_up, details_dict)
    """
    
    # ========== Likelihood: P(signal | bullish) ==========
    p_signal_given_bullish = _calculate_likelihood(
        trend_strength, trend_direction, rsi, regime
    )
    
    # ========== Opposite likelihood: P(signal | bearish) ==========
    p_signal_given_bearish = 1.0 - p_signal_given_bullish
    
    # ========== Bayes theorem ==========
    likelihood_ratio = (p_signal_given_bullish + 1e-9) / (p_signal_given_bearish + 1e-9)
    
    posterior = (likelihood_ratio * prior_p_bullish) / (
        likelihood_ratio * prior_p_bullish + (1 - prior_p_bullish) + 1e-9
    )
    
    # ========== News sentiment adjustment (bounded) ==========
    news_deviation = (news_score - 50) / 100  # -0.5 to +0.5
    news_factor = 1.0 + np.clip(news_deviation * 0.2, -0.15, 0.15)
    
    adjusted_prob = posterior * news_factor
    adjusted_prob = np.clip(adjusted_prob, 0.1, 0.9)  # Bounds
    
    return adjusted_prob, {
        'p_signal_given_bullish': p_signal_given_bullish,
        'p_signal_given_bearish': p_signal_given_bearish,
        'likelihood_ratio': likelihood_ratio,
        'prior_p_bullish': prior_p_bullish,
        'posterior': posterior,
        'news_factor': news_factor,
        'final_probability': adjusted_prob,
        'historical_win_rate': historical_win_rate
    }


def _calculate_likelihood(
    trend_strength: int,
    direction: str,
    rsi: float,
    regime: RegimeType
) -> float:
    """
    Calculate P(signal | bullish move).
    
    These probabilities should be trained from backtests.
    Using reasonable defaults here.
    """
    
    base_likelihood = 0.5
    
    # Trend strength factor
    if direction == 'up':
        if trend_strength > 75:
            base_likelihood = 0.75
        elif trend_strength > 60:
            base_likelihood = 0.65
        elif trend_strength > 45:
            base_likelihood = 0.55
        else:
            base_likelihood = 0.45
    elif direction == 'down':
        base_likelihood = 1.0 - base_likelihood
    
    # RSI confirmation
    if 45 < rsi < 55:
        base_likelihood *= 0.95  # Neutral = less confirmative
    elif rsi > 60 and direction == 'up':
        base_likelihood *= 1.05
    elif rsi < 40 and direction == 'down':
        base_likelihood *= 1.05
    
    # Regime adjustment
    if regime == RegimeType.TRENDING_UP:
        base_likelihood *= 1.1
    elif regime == RegimeType.TRENDING_DOWN:
        base_likelihood *= 0.9
    elif regime == RegimeType.VOLATILE:
        base_likelihood *= 0.85
    
    return np.clip(base_likelihood, 0.15, 0.85)


# ============================================================================
# 3. PROFESSIONAL SUPPORT/RESISTANCE IDENTIFICATION
# ============================================================================

def identify_sr_levels_professional(
    df: pd.DataFrame,
    min_touches: int = 2,
    tolerance_pct: float = 0.8,
    use_body_only: bool = True
) -> SRLevels:
    """
    Professional S/R identification with volume + price action.
    
    Replaces: 2-bar pivot noisy implementation
    
    Returns: SRLevels with strength measurements
    """
    
    if df is None or df.empty or len(df) < 10:
        logger.warning("Insufficient data for S/R analysis")
        return SRLevels([], [], None, None)
    
    df_copy = df.copy().reset_index(drop=True)
    
    # ========== Extract price levels (use candle bodies if specified) ==========
    if use_body_only:
        df_copy['hl_high'] = df_copy[['Open', 'Close']].max(axis=1)
        df_copy['hl_low'] = df_copy[['Open', 'Close']].min(axis=1)
    else:
        df_copy['hl_high'] = df_copy['High']
        df_copy['hl_low'] = df_copy['Low']
    
    # 5-bar pivots (less noisy than 2-bar)
    df_copy['hl5_high'] = df_copy['hl_high'].rolling(window=5, center=True).max()
    df_copy['hl5_low'] = df_copy['hl_low'].rolling(window=5, center=True).min()
    
    # ========== Identify pivot points with volume confirmation ==========
    vol_ma = df_copy['Volume'].rolling(20).mean()
    vol_threshold = vol_ma.mean()
    
    potential_resistances = []
    potential_supports = []
    
    for i in range(2, len(df_copy) - 2):
        current_vol = df_copy.iloc[i]['Volume']
        vol_ok = current_vol > vol_threshold * 0.7  # Don't require too much volume
        
        # Resistance: High pivot with acceptable volume
        if (df_copy.iloc[i]['hl_high'] == df_copy.iloc[i]['hl5_high'] and vol_ok):
            potential_resistances.append({
                'price': df_copy.iloc[i]['hl_high'],
                'index': i,
                'date': df_copy.iloc[i].get('date'),
                'volume': current_vol
            })
        
        # Support: Low pivot with acceptable volume
        if (df_copy.iloc[i]['hl_low'] == df_copy.iloc[i]['hl5_low'] and vol_ok):
            potential_supports.append({
                'price': df_copy.iloc[i]['hl_low'],
                'index': i,
                'date': df_copy.iloc[i].get('date'),
                'volume': current_vol
            })
    
    # ========== Cluster nearby levels ==========
    support_clusters = _cluster_price_levels(
        [p['price'] for p in potential_supports],
        tolerance_pct
    )
    resistance_clusters = _cluster_price_levels(
        [p['price'] for p in potential_resistances],
        tolerance_pct
    )
    
    # ========== Count touches and filter ==========
    support_levels = []
    for level in support_clusters:
        touches = sum(1 for p in potential_supports
                     if abs(p['price'] - level) < level * tolerance_pct / 100)
        if touches >= min_touches:
            support_levels.append((level, touches))
    
    resistance_levels = []
    for level in resistance_clusters:
        touches = sum(1 for p in potential_resistances
                     if abs(p['price'] - level) < level * tolerance_pct / 100)
        if touches >= min_touches:
            resistance_levels.append((level, touches))
    
    # Sort by price, keep top 3
    support_levels = sorted(support_levels, key=lambda x: x[0], reverse=True)[:3]
    resistance_levels = sorted(resistance_levels, key=lambda x: x[0], reverse=True)[:3]
    
    # Find nearest levels
    current_price = df_copy.iloc[-1]['Close']
    nearest_support = None
    nearest_resistance = None
    
    for level, touches in support_levels:
        if level < current_price:
            nearest_support = level
            break
    
    for level, touches in reversed(resistance_levels):
        if level > current_price:
            nearest_resistance = level
            break
    
    return SRLevels(
        supports=support_levels,
        resistances=resistance_levels,
        nearest_support=nearest_support,
        nearest_resistance=nearest_resistance
    )


def _cluster_price_levels(prices: List[float], tolerance_pct: float) -> List[float]:
    """Cluster similar price levels within tolerance"""
    if not prices:
        return []
    
    prices = sorted(prices)
    clusters = []
    current_cluster = [prices[0]]
    
    for price in prices[1:]:
        avg_price = np.mean(current_cluster)
        tolerance_amount = avg_price * tolerance_pct / 100
        
        if abs(price - current_cluster[-1]) < tolerance_amount:
            current_cluster.append(price)
        else:
            clusters.append(np.median(current_cluster))
            current_cluster = [price]
    
    clusters.append(np.median(current_cluster))
    return clusters


# ============================================================================
# 4. MULTI-TIMEFRAME CONFLUENCE ANALYSIS
# ============================================================================

def analyze_multi_timeframe_confluence(
    tf_dfs: Dict[str, pd.DataFrame]
) -> Tuple[int, Dict]:
    """
    Multi-timeframe analysis with confluence measurement.
    
    Replaces: Static weight averaging
    
    Returns: (weighted_score, confluence_details)
    """
    
    tf_priority = {'1D': 3, '4H': 2, '1H': 1, '1W': 4}
    tf_order = sorted(
        [(tf, p) for tf, p in tf_priority.items() if tf in tf_dfs],
        key=lambda x: x[1], reverse=True
    )
    
    scores = {}
    analyses = {}
    signals = {}
    
    # ========== Calculate trend for each timeframe ==========
    for tf, priority in tf_order:
        if tf not in tf_dfs or tf_dfs[tf].empty:
            scores[tf] = 50
            signals[tf] = 'neutral'
            continue
        
        analysis = calculate_trend_strength_enhanced(tf_dfs[tf])
        scores[tf] = analysis.strength
        analyses[tf] = analysis
        
        if analysis.strength > 65:
            signals[tf] = 'bullish'
        elif analysis.strength < 35:
            signals[tf] = 'bearish'
        else:
            signals[tf] = 'neutral'
    
    # ========== Confluence strength ==========
    bullish_count = sum(1 for s in signals.values() if s == 'bullish')
    bearish_count = sum(1 for s in signals.values() if s == 'bearish')
    total_tf = len(signals)
    
    confluence_ratio = max(bullish_count, bearish_count) / max(total_tf, 1)
    
    # ========== Weighted calculation (dynamic weights) ==========
    total_weight = 0.0
    weighted_sum = 0.0
    
    for tf, priority in tf_order:
        if tf in scores:
            weight = priority / 10  # Normalized priority
            weighted_sum += scores[tf] * weight
            total_weight += weight
    
    weighted_score = weighted_sum / max(total_weight, 1)
    
    # ========== Penalize misalignment ==========
    misalignment = abs(bullish_count - bearish_count) / max(total_tf, 1)
    if misalignment < 0.6:  # Mixed signals
        weighted_score *= 0.90  # 10% penalty for confusion
    
    # ========== Direction determination ==========
    if bullish_count > bearish_count:
        direction = 'BULLISH'
    elif bearish_count > bullish_count:
        direction = 'BEARISH'
    else:
        direction = 'MIXED'
    
    weighted_score = int(np.clip(weighted_score, 0, 100))
    
    return weighted_score, {
        'final_score': weighted_score,
        'direction': direction,
        'confluence_strength': confluence_ratio,
        'bullish_tfs': bullish_count,
        'bearish_tfs': bearish_count,
        'mixed_signals': misalignment > 0.6,
        'tf_scores': scores,
        'tf_signals': signals,
        'alignment_quality': 1.0 - misalignment
    }


# ============================================================================
# 5. DYNAMIC POSITION SIZING WITH RISK ADJUSTMENT
# ============================================================================

def calculate_position_size_dynamic(
    account_balance: float,
    entry_price: float,
    stop_loss: float,
    trend_strength: int,
    volatility_regime: str = 'normal',
    max_risk_per_trade_pct: float = 1.0
) -> Tuple[float, Dict]:
    """
    Dynamic position sizing based on risk, trend strength, and volatility.
    
    Replaces: Simple fixed percentage sizing
    
    Returns: (shares, sizing_details)
    """
    
    base_risk = account_balance * max_risk_per_trade_pct / 100
    
    # Trend-based sizing
    trend_factor = 1.0
    if trend_strength > 70:
        trend_factor = 1.15  # Increase size in strong trend
    elif trend_strength < 40:
        trend_factor = 0.75  # Reduce size in weak trend
    
    # Volatility adjustment
    vol_factor = {
        'low': 1.15,  # Can size up in low volatility
        'normal': 1.0,
        'high': 0.70,  # Reduce in high volatility
        'extreme': 0.40
    }.get(volatility_regime, 1.0)
    
    adjusted_risk = base_risk * trend_factor * vol_factor
    
    # Calculate shares
    risk_distance = abs(entry_price - stop_loss)
    if risk_distance <= 0:
        logger.warning("Invalid stop loss distance")
        return 0.0, {'error': 'Invalid SL distance'}
    
    shares = adjusted_risk / risk_distance
    
    return shares, {
        'base_risk': base_risk,
        'adjusted_risk': adjusted_risk,
        'trend_factor': trend_factor,
        'vol_factor': vol_factor,
        'shares': shares,
        'risk_per_share': risk_distance,
        'potential_loss': adjusted_risk,
        'potential_loss_pct': (adjusted_risk / account_balance) * 100
    }


# ============================================================================
# 6. ENHANCED VSA ANALYSIS
# ============================================================================

def analyze_vsa_professional(
    df: pd.DataFrame,
    lookback: int = 20
) -> Dict:
    """
    Professional VSA (Volume Spread Analysis).
    
    Replaces: Simple on-bar volume check
    
    Returns: VSA signal with strength
    """
    
    if df is None or df.empty or len(df) < lookback:
        return {'signal': 'neutral', 'strength': 0.5}
    
    recent = df.iloc[-1].copy()
    vol_avg = df['Volume'].tail(lookback).mean()
    
    close = recent['Close']
    open_ = recent['Open']
    high = recent['High']
    low = recent['Low']
    volume = recent['Volume']
    
    # Candle metrics
    body = abs(close - open_)
    total_range = high - low
    upper_wick = high - max(close, open_)
    lower_wick = min(close, open_) - low
    
    # Volume analysis
    vol_ratio = volume / (vol_avg + 1e-9)
    
    signal = 'neutral'
    strength = 0.5
    
    # ========== Accumulation signals ==========
    # Strong close with heavy volume = buying pressure
    if close > open_ and volume > vol_avg * 1.3 and body > total_range * 0.6:
        signal = 'accumulation'
        strength = min(1.0, vol_ratio / 2.0)
    
    # Large lower wick with heavy volume = support test + buying
    elif lower_wick > body and volume > vol_avg * 1.2:
        signal = 'accumulation'
        strength = 0.7
    
    # ========== Distribution signals ==========
    # Weak close with heavy volume = selling pressure
    elif close < open_ and volume > vol_avg * 1.3 and body > total_range * 0.6:
        signal = 'distribution'
        strength = min(1.0, vol_ratio / 2.0)
    
    # Large upper wick with heavy volume = resistance test + selling
    elif upper_wick > body and volume > vol_avg * 1.2:
        signal = 'distribution'
        strength = 0.7
    
    return {
        'signal': signal,
        'strength': strength,
        'vol_ratio': vol_ratio,
        'body_ratio': body / (total_range + 1e-9),
        'upper_wick': upper_wick,
        'lower_wick': lower_wick
    }


# ============================================================================
# 7. EXPECTED VALUE CALCULATION (FOR TRADE EVALUATION)
# ============================================================================

def calculate_expected_value(
    entry_price: float,
    stop_loss: float,
    target_price: float,
    probability_up: float,
    risk_per_trade: float
) -> Dict:
    """
    Calculate expected value of a trade.
    
    Returns: EV metrics for decision making
    """
    
    if entry_price <= 0 or stop_loss >= entry_price or target_price <= entry_price:
        return {'valid': False, 'error': 'Invalid prices'}
    
    reward = target_price - entry_price
    risk = entry_price - stop_loss
    rr_ratio = reward / (risk + 1e-9)
    
    # Expected value calculation
    win_payout = risk_per_trade * rr_ratio
    loss_payout = -risk_per_trade
    
    ev = probability_up * win_payout + (1 - probability_up) * loss_payout
    ev_ratio = ev / risk_per_trade if risk_per_trade > 0 else 0
    
    return {
        'valid': True,
        'entry': entry_price,
        'stop_loss': stop_loss,
        'target': target_price,
        'reward': reward,
        'risk': risk,
        'rr_ratio': rr_ratio,
        'probability_up': probability_up,
        'probability_down': 1 - probability_up,
        'expected_value': ev,
        'ev_ratio': ev_ratio,
        'kelly_fraction': ev_ratio / rr_ratio if rr_ratio > 0 else 0,  # Kelly criterion
        'trade_worthy': ev > 0 and rr_ratio >= 1.5
    }


# ============================================================================
# UTILITY: Data Validation
# ============================================================================

def validate_dataframe_quality(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate OHLCV dataframe quality"""
    issues = []
    
    required_cols = {'Open', 'High', 'Low', 'Close', 'Volume'}
    missing = required_cols - set(df.columns)
    if missing:
        issues.append(f"Missing columns: {missing}")
    
    # Check for NaN values in OHLC
    for col in ['Open', 'High', 'Low', 'Close']:
        if col in df.columns and df[col].isna().any():
            issues.append(f"NaN values in {col}")
    
    # Check logical consistency
    if 'High' in df.columns and 'Low' in df.columns:
        if (df['High'] < df['Low']).any():
            issues.append("High < Low in some candles")
    
    if 'Volume' in df.columns and (df['Volume'] < 0).any():
        issues.append("Negative volumes detected")
    
    is_valid = len(issues) == 0
    return is_valid, issues


if __name__ == "__main__":
    print("Optimization module loaded successfully")
    print("Available functions:")
    print("- calculate_trend_strength_enhanced()")
    print("- calculate_bayesian_probability()")
    print("- identify_sr_levels_professional()")
    print("- analyze_multi_timeframe_confluence()")
    print("- calculate_position_size_dynamic()")
    print("- analyze_vsa_professional()")
    print("- calculate_expected_value()")
