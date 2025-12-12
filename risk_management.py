# risk_management.py
# Professional Risk Management System for Production Trading

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class TradePosition:
    """Represents an open trading position"""
    ticker: str
    entry_price: float
    stop_loss: float
    take_profit: float
    quantity: float
    entry_date: datetime
    entry_signal_strength: float
    risk_amount: float
    
    def current_pnl(self, current_price: float) -> float:
        """Calculate current P&L"""
        return (current_price - self.entry_price) * self.quantity
    
    def pnl_pct(self, current_price: float) -> float:
        """P&L percentage"""
        return ((current_price - self.entry_price) / self.entry_price) * 100
    
    def is_stopped_out(self, current_price: float) -> bool:
        """Check if position hit stop loss"""
        return current_price <= self.stop_loss
    
    def is_tp_hit(self, current_price: float) -> bool:
        """Check if position hit take profit"""
        return current_price >= self.take_profit


@dataclass
class PortfolioMetrics:
    """Portfolio performance metrics"""
    total_capital: float
    used_capital: float
    free_capital: float
    equity: float
    unrealized_pnl: float
    realized_pnl: float
    total_pnl: float
    pnl_pct: float
    num_open_positions: int
    num_winning_trades: int
    num_losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    max_drawdown: float
    max_drawdown_pct: float
    sharpe_ratio: float
    
    def __str__(self) -> str:
        return (
            f"Portfolio Status:\n"
            f"  Capital: ${self.total_capital:,.2f}\n"
            f"  Equity: ${self.equity:,.2f}\n"
            f"  P&L: ${self.total_pnl:,.2f} ({self.pnl_pct:.2f}%)\n"
            f"  Max DD: {self.max_drawdown_pct:.2f}%\n"
            f"  Win Rate: {self.win_rate:.1%}"
        )


# ============================================================================
# PROFESSIONAL RISK MANAGER
# ============================================================================

class RiskManager:
    """
    Professional risk management system for trading.
    
    Handles:
    - Position sizing (dynamic based on volatility + risk)
    - Daily/weekly loss limits with circuit breakers
    - Correlation analysis to prevent correlated positions
    - Portfolio-level risk monitoring
    - P&L tracking and performance metrics
    """
    
    def __init__(
        self,
        account_balance: float,
        daily_loss_limit_pct: float = 2.0,
        weekly_loss_limit_pct: float = 5.0,
        max_drawdown_pct: float = 20.0,
        max_risk_per_trade_pct: float = 1.0,
        max_position_correlation: float = 0.7
    ):
        """Initialize risk manager with limits"""
        self.account_balance = account_balance
        self.initial_equity = account_balance
        
        # Loss limits
        self.daily_loss_limit = account_balance * daily_loss_limit_pct / 100
        self.weekly_loss_limit = account_balance * weekly_loss_limit_pct / 100
        self.max_drawdown_limit = account_balance * max_drawdown_pct / 100
        
        # Risk parameters
        self.max_risk_per_trade = max_risk_per_trade_pct
        self.max_position_correlation = max_position_correlation
        
        # State tracking
        self.open_positions: Dict[str, TradePosition] = {}
        self.closed_trades: List[Dict] = []
        self.daily_pnl_history: Dict[datetime, float] = {}
        self.equity_curve: List[Tuple[datetime, float]] = [(datetime.now(), account_balance)]
        
        # Correlation matrix (updated periodically)
        self.correlation_matrix: Optional[pd.DataFrame] = None
        
        logger.info(f"RiskManager initialized with ${account_balance:,.2f} account")
    
    # ========== POSITION SIZING ==========
    
    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss: float,
        volatility_atr: float,
        trend_strength: int = 50,
        volatility_regime: str = 'normal'
    ) -> Tuple[float, Dict]:
        """
        Calculate position size using Kelly Criterion variant with volatility adjustment.
        
        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            volatility_atr: Current ATR (volatility measure)
            trend_strength: 0-100 trend score
            volatility_regime: 'low', 'normal', 'high', 'extreme'
        
        Returns:
            (shares_to_trade, details)
        """
        
        # Base risk amount
        base_risk = self.account_balance * self.max_risk_per_trade / 100
        
        # Risk distance
        risk_distance = abs(entry_price - stop_loss)
        if risk_distance <= 0:
            logger.warning(f"Invalid stop loss for {entry_price}")
            return 0.0, {'error': 'Invalid stop loss'}
        
        # ========== Trend-based adjustment ==========
        trend_factor = 1.0
        if trend_strength > 70:
            trend_factor = 1.2  # Increase in strong trends
        elif trend_strength > 55:
            trend_factor = 1.0
        elif trend_strength < 35:
            trend_factor = 0.6  # Reduce in weak trends
        
        # ========== Volatility adjustment ==========
        vol_factor = {
            'low': 1.25,      # Can take more in calm markets
            'normal': 1.0,
            'high': 0.65,     # Reduce in high vol
            'extreme': 0.30   # Minimal in extreme vol
        }.get(volatility_regime, 1.0)
        
        adjusted_risk = base_risk * trend_factor * vol_factor
        
        # ========== Calculate shares ==========
        shares = adjusted_risk / risk_distance
        
        # ========== Max position size safety check ==========
        max_position_value = self.account_balance * 0.3  # Never >30% of capital per position
        if entry_price * shares > max_position_value:
            shares = max_position_value / entry_price
        
        return shares, {
            'base_risk': base_risk,
            'trend_factor': trend_factor,
            'vol_factor': vol_factor,
            'adjusted_risk': adjusted_risk,
            'shares': shares,
            'position_value': entry_price * shares,
            'position_value_pct': (entry_price * shares / self.account_balance) * 100,
            'risk_per_trade': adjusted_risk
        }
    
    # ========== POSITION MANAGEMENT ==========
    
    def open_position(
        self,
        ticker: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        quantity: float,
        signal_strength: float = 0.5
    ) -> bool:
        """
        Open a new trading position with risk checks.
        
        Returns: True if position opened, False if rejected
        """
        
        # Check if daily loss limit hit
        if not self.check_daily_loss_limit():
            logger.warning(f"Daily loss limit exceeded. Cannot open {ticker}")
            return False
        
        # Check correlation with existing positions
        if not self.check_correlation_risk(ticker):
            logger.warning(f"Correlation risk too high for {ticker}")
            return False
        
        # Create position
        position = TradePosition(
            ticker=ticker,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            quantity=quantity,
            entry_date=datetime.now(),
            entry_signal_strength=signal_strength,
            risk_amount=abs(entry_price - stop_loss) * quantity
        )
        
        self.open_positions[ticker] = position
        logger.info(f"Position opened: {ticker} @ {entry_price} (qty: {quantity:.2f})")
        
        return True
    
    def close_position(
        self,
        ticker: str,
        exit_price: float,
        reason: str = 'manual'
    ) -> Optional[Dict]:
        """
        Close an open position.
        
        Returns: Trade details if successful
        """
        
        if ticker not in self.open_positions:
            logger.warning(f"No open position for {ticker}")
            return None
        
        position = self.open_positions[ticker]
        pnl = position.current_pnl(exit_price)
        pnl_pct = position.pnl_pct(exit_price)
        
        trade_record = {
            'ticker': ticker,
            'entry_price': position.entry_price,
            'exit_price': exit_price,
            'quantity': position.quantity,
            'entry_date': position.entry_date,
            'exit_date': datetime.now(),
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'reason': reason,
            'hold_days': (datetime.now() - position.entry_date).days
        }
        
        # Track in closed trades
        self.closed_trades.append(trade_record)
        
        # Update daily P&L
        today = datetime.now().date()
        if today not in self.daily_pnl_history:
            self.daily_pnl_history[today] = 0.0
        self.daily_pnl_history[today] += pnl
        
        # Remove from open
        del self.open_positions[ticker]
        
        logger.info(f"Position closed: {ticker} | P&L: ${pnl:,.2f} ({pnl_pct:.2f}%)")
        
        return trade_record
    
    def update_position(
        self,
        ticker: str,
        current_price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> None:
        """Update position stop loss / take profit"""
        
        if ticker not in self.open_positions:
            return
        
        position = self.open_positions[ticker]
        
        if stop_loss is not None:
            position.stop_loss = stop_loss
        if take_profit is not None:
            position.take_profit = take_profit
    
    # ========== RISK CHECKS ==========
    
    def check_daily_loss_limit(self) -> bool:
        """
        Circuit breaker: Check if daily loss limit exceeded.
        
        Returns: True if can continue trading, False if should stop
        """
        
        today = datetime.now().date()
        daily_loss = self.daily_pnl_history.get(today, 0.0)
        
        if abs(daily_loss) >= self.daily_loss_limit:
            logger.warning(f"Daily loss limit hit: ${daily_loss:,.2f}")
            return False
        
        return True
    
    def check_weekly_loss_limit(self) -> bool:
        """Check if weekly loss limit exceeded"""
        
        week_start = datetime.now() - timedelta(days=datetime.now().weekday())
        week_loss = sum(
            pnl for date, pnl in self.daily_pnl_history.items()
            if date >= week_start.date()
        )
        
        if abs(week_loss) >= self.weekly_loss_limit:
            logger.warning(f"Weekly loss limit hit: ${week_loss:,.2f}")
            return False
        
        return True
    
    def check_max_drawdown(self, current_equity: float) -> bool:
        """Check if maximum drawdown exceeded"""
        
        max_equity = max([e for _, e in self.equity_curve], default=self.account_balance)
        drawdown = (max_equity - current_equity) / max_equity
        
        if drawdown >= (self.max_drawdown_limit / self.account_balance):
            logger.warning(f"Max drawdown exceeded: {drawdown:.2%}")
            return False
        
        return True
    
    def check_correlation_risk(self, ticker: str) -> bool:
        """
        Check if adding position would create too high correlation.
        
        Returns: True if safe to add, False if too correlated
        """
        
        if not self.open_positions or self.correlation_matrix is None:
            return True
        
        for other_ticker in self.open_positions.keys():
            if ticker == other_ticker:
                continue
            
            try:
                corr = self.correlation_matrix.loc[ticker, other_ticker]
                if corr > self.max_position_correlation:
                    logger.warning(
                        f"Correlation too high: {ticker}-{other_ticker} = {corr:.2f}"
                    )
                    return False
            except (KeyError, ValueError):
                continue
        
        return True
    
    # ========== PORTFOLIO MONITORING ==========
    
    def get_portfolio_metrics(self, current_prices: Dict[str, float]) -> PortfolioMetrics:
        """Calculate comprehensive portfolio metrics"""
        
        # Calculate unrealized P&L
        unrealized_pnl = 0.0
        used_capital = 0.0
        
        for ticker, position in self.open_positions.items():
            current_price = current_prices.get(ticker, position.entry_price)
            unrealized_pnl += position.current_pnl(current_price)
            used_capital += position.entry_price * position.quantity
        
        # Calculate realized P&L from closed trades
        realized_pnl = sum(t['pnl'] for t in self.closed_trades)
        
        # Total metrics
        total_pnl = unrealized_pnl + realized_pnl
        equity = self.account_balance + total_pnl
        
        # Trade statistics
        winning_trades = [t for t in self.closed_trades if t['pnl'] > 0]
        losing_trades = [t for t in self.closed_trades if t['pnl'] < 0]
        
        win_rate = len(winning_trades) / max(len(self.closed_trades), 1)
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0.0
        avg_loss = np.mean([abs(t['pnl']) for t in losing_trades]) if losing_trades else 0.0
        profit_factor = avg_win / (avg_loss + 1e-9) if avg_loss > 0 else 0.0
        
        # Drawdown calculation
        equity_values = [e for _, e in self.equity_curve]
        if equity_values:
            max_equity = max(equity_values)
            max_drawdown = max_equity - min(equity_values)
            max_drawdown_pct = (max_drawdown / max_equity) * 100
        else:
            max_drawdown = 0.0
            max_drawdown_pct = 0.0
        
        # Sharpe ratio (simplified: daily returns)
        daily_returns = []
        for date in sorted(self.daily_pnl_history.keys()):
            daily_returns.append(
                self.daily_pnl_history[date] / self.account_balance
            )
        
        sharpe_ratio = 0.0
        if daily_returns:
            returns_std = np.std(daily_returns)
            returns_mean = np.mean(daily_returns)
            if returns_std > 0:
                sharpe_ratio = (returns_mean / returns_std) * np.sqrt(252)
        
        return PortfolioMetrics(
            total_capital=self.account_balance,
            used_capital=used_capital,
            free_capital=self.account_balance - used_capital,
            equity=equity,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=realized_pnl,
            total_pnl=total_pnl,
            pnl_pct=(total_pnl / self.account_balance) * 100,
            num_open_positions=len(self.open_positions),
            num_winning_trades=len(winning_trades),
            num_losing_trades=len(losing_trades),
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            sharpe_ratio=sharpe_ratio
        )
    
    def update_correlation_matrix(self, returns_df: pd.DataFrame) -> None:
        """Update correlation matrix from returns data"""
        
        try:
            self.correlation_matrix = returns_df.corr()
            logger.info("Correlation matrix updated")
        except Exception as e:
            logger.error(f"Failed to update correlation matrix: {e}")
    
    def update_equity(self, current_prices: Dict[str, float]) -> float:
        """Update equity and add to curve"""
        
        metrics = self.get_portfolio_metrics(current_prices)
        self.equity_curve.append((datetime.now(), metrics.equity))
        
        return metrics.equity
    
    # ========== MONITORING & ALERTS ==========
    
    def check_positions_for_exit(
        self,
        current_prices: Dict[str, float]
    ) -> List[Dict]:
        """
        Check all open positions for stop loss / take profit hits.
        
        Returns: List of positions to close
        """
        
        positions_to_close = []
        
        for ticker, position in self.open_positions.items():
            current_price = current_prices.get(ticker)
            if current_price is None:
                continue
            
            if position.is_stopped_out(current_price):
                positions_to_close.append({
                    'ticker': ticker,
                    'exit_price': position.stop_loss,
                    'reason': 'stop_loss_hit',
                    'pnl': position.current_pnl(position.stop_loss)
                })
            
            elif position.is_tp_hit(current_price):
                positions_to_close.append({
                    'ticker': ticker,
                    'exit_price': position.take_profit,
                    'reason': 'take_profit_hit',
                    'pnl': position.current_pnl(position.take_profit)
                })
        
        return positions_to_close
    
    def get_risk_alerts(self, current_prices: Dict[str, float]) -> List[str]:
        """Generate list of risk alerts"""
        
        alerts = []
        metrics = self.get_portfolio_metrics(current_prices)
        
        # Daily loss warning
        today = datetime.now().date()
        daily_loss = self.daily_pnl_history.get(today, 0.0)
        if abs(daily_loss) > self.daily_loss_limit * 0.7:
            alerts.append(f"WARNING: Daily loss at {abs(daily_loss)/self.daily_loss_limit:.0%} of limit")
        
        # Drawdown warning
        if metrics.max_drawdown_pct > 15:
            alerts.append(f"WARNING: Drawdown at {metrics.max_drawdown_pct:.1f}%")
        
        # Low win rate
        if metrics.win_rate < 0.35 and len(self.closed_trades) > 10:
            alerts.append(f"WARNING: Low win rate {metrics.win_rate:.1%} - review strategy")
        
        return alerts


# ============================================================================
# HELPER: Volatility Regime Detection
# ============================================================================

def detect_volatility_regime(
    atr: float,
    atr_sma_20: Optional[float],
    price: float
) -> str:
    """
    Detect current volatility regime.
    
    Returns: 'low', 'normal', 'high', or 'extreme'
    """
    
    atr_pct = (atr / price) * 100
    
    if atr_sma_20 is None:
        # Use absolute thresholds
        if atr_pct < 0.8:
            return 'low'
        elif atr_pct < 1.5:
            return 'normal'
        elif atr_pct < 2.5:
            return 'high'
        else:
            return 'extreme'
    
    # Use relative to average
    atr_ratio = atr / atr_sma_20
    if atr_ratio < 0.8:
        return 'low'
    elif atr_ratio < 1.0:
        return 'normal'
    elif atr_ratio < 1.5:
        return 'high'
    else:
        return 'extreme'


if __name__ == "__main__":
    # Example usage
    rm = RiskManager(account_balance=100000.0)
    
    # Open a position
    rm.open_position(
        ticker="BRPT.JK",
        entry_price=15200.0,
        stop_loss=15000.0,
        take_profit=15800.0,
        quantity=5.0,
        signal_strength=0.75
    )
    
    # Get metrics
    metrics = rm.get_portfolio_metrics({"BRPT.JK": 15400.0})
    print(metrics)
