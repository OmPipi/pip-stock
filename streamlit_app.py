"""
streamlit_app.py
Automated Stock Analysis (Technical + Fundamental + News) with probabilistic recommendations.
Author: (template) - adapt as needed
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import plotly.graph_objs as go
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import math

st.set_page_config(layout="wide", page_title="Auto Stock Analyzer", initial_sidebar_state="expanded")

# ---------- Helper utils ----------
def fetch_price_data(ticker, period="6mo", interval="1d"):
    """Fetch OHLCV with yfinance"""
    ticker_obj = yf.Ticker(ticker)
    df = ticker_obj.history(period=period, interval=interval, actions=False)
    if df.empty:
        raise ValueError("No price data found for ticker: " + ticker)
    df.index = pd.to_datetime(df.index)
    df = df.rename_axis("date").reset_index()
    return df

def compute_technical_indicators(df):
    """Add common indicators: MA20/50/200, RSI, MACD, ATR, SMA behavior"""
    d = df.copy()
    d.set_index("date", inplace=True)
    # Moving averages
    d["ma20"] = d["Close"].rolling(20).mean()
    d["ma50"] = d["Close"].rolling(50).mean()
    d["ma200"] = d["Close"].rolling(200).mean()
    # RSI
    d["rsi14"] = ta.rsi(d["Close"], length=14)
    # MACD
    macd = ta.macd(d["Close"])
    if macd is not None:
        d["macd"] = macd["MACD_12_26_9"]
        d["macd_signal"] = macd["MACDs_12_26_9"]
    # ATR
    d["atr14"] = ta.atr(d["High"], d["Low"], d["Close"], length=14)
    d = d.reset_index()
    return d

def fetch_fundamentals(ticker):
    """Quick fundamental snapshot from yfinance"""
    tk = yf.Ticker(ticker)
    info = tk.info
    # Safely pull key metrics
    metrics = {
        "shortName": info.get("shortName"),
        "sector": info.get("sector"),
        "marketCap": info.get("marketCap"),
        "trailingPE": info.get("trailingPE"),
        "forwardPE": info.get("forwardPE"),
        "priceToBook": info.get("priceToBook"),
        "epsTrailingTwelveMonths": info.get("trailingEps"),
        "returnOnEquity": info.get("returnOnEquity"),
        "debtToEquity": info.get("debtToEquity"),
    }
    return metrics

# --------- News sentiment (simple) ----------
analyzer = SentimentIntensityAnalyzer()

def google_news_headlines(ticker, max_headlines=10):
    """Scrape Google News search for ticker/keyword. Fallback if NewsAPI not provided."""
    query = ticker + " saham OR stock"
    url = f"https://www.google.com/search?q={requests.utils.quote(query)}&tbm=nws"
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=10)
    soup = BeautifulSoup(r.text, "html.parser")
    articles = soup.select("div.dbsr")
    headlines = []
    for a in articles[:max_headlines]:
        title_tag = a.select_one("div.JheGif.nDgy9d")
        snippet = a.select_one("div.Y3v8qd")
        title = title_tag.get_text() if title_tag else None
        snippet = snippet.get_text() if snippet else ""
        if title:
            headlines.append({"title": title, "snippet": snippet})
    return headlines

def compute_news_sentiment(headlines):
    if not headlines:
        return {"score": 50, "details": []}  # neutral
    scores = []
    details = []
    for h in headlines:
        text = h.get("title", "") + " " + h.get("snippet", "")
        vs = analyzer.polarity_scores(text)
        compound = vs["compound"]
        scores.append(compound)
        details.append({"text": text, "compound": compound})
    # normalize compound (-1..1) to 0..100
    avg = np.mean(scores)
    norm = int((avg + 1) / 2 * 100)
    return {"score": norm, "details": details}

# ---------- Scoring logic ----------
def score_technical(df):
    """Produce technical score (0-100) based on rules"""
    recent = df.iloc[-1]
    score = 50
    reasons = []
    # Trend: price vs ma50/ma200
    price = recent["Close"]
    ma50 = recent.get("ma50", np.nan)
    ma200 = recent.get("ma200", np.nan)
    if not np.isnan(ma50) and not np.isnan(ma200):
        if price > ma50:
            score += 8
            reasons.append("Price > MA50")
        else:
            score -= 8
            reasons.append("Price < MA50")
        if price > ma200:
            score += 10
            reasons.append("Price > MA200")
        else:
            score -= 10
            reasons.append("Price < MA200")
    # RSI
    rsi = recent.get("rsi14", np.nan)
    if not np.isnan(rsi):
        if rsi < 30:
            score += 6; reasons.append("RSI oversold")
        elif rsi > 70:
            score -= 6; reasons.append("RSI overbought")
        else:
            score += 2; reasons.append("RSI neutral")
    # MACD bullish?
    macd = recent.get("macd", np.nan)
    macd_sig = recent.get("macd_signal", np.nan)
    if not np.isnan(macd) and not np.isnan(macd_sig):
        if macd > macd_sig:
            score += 6; reasons.append("MACD bullish")
        else:
            score -= 6; reasons.append("MACD bearish")
    # Volume spikes (compare last volume to 20-day average)
    if "Volume" in df.columns:
        vol = recent["Volume"]
        vol_avg20 = df["Volume"].tail(20).mean()
        if vol > vol_avg20 * 1.8:
            score += 5; reasons.append("Volume spike")
    # Normalize to 0-100
    score = max(0, min(100, int(score)))
    return {"score": score, "reasons": reasons}

def score_fundamental(fund):
    """Simple fundamental scoring 0-100"""
    score = 50
    reasons = []
    pe = fund.get("trailingPE")
    pb = fund.get("priceToBook")
    roe = fund.get("returnOnEquity")
    debt2eq = fund.get("debtToEquity")
    # P/E: lower vs industry is good — simple heuristic
    if pe and pe > 0:
        if pe < 10:
            score += 10; reasons.append("Low P/E")
        elif pe < 20:
            score += 5; reasons.append("Moderate P/E")
        else:
            score -= 8; reasons.append("High P/E")
    if pb and pb > 0:
        if pb < 1:
            score += 6; reasons.append("Low P/B")
        elif pb < 3:
            score += 2; reasons.append("Moderate P/B")
        else:
            score -= 4; reasons.append("High P/B")
    if roe:
        if roe > 0.15:
            score += 8; reasons.append("Healthy ROE")
        elif roe > 0.08:
            score += 3; reasons.append("Acceptable ROE")
        else:
            score -= 4; reasons.append("Low ROE")
    if debt2eq:
        if debt2eq < 50:
            score += 4; reasons.append("Low debt/equity")
        else:
            score -= 4; reasons.append("High debt/equity")
    score = max(0, min(100, int(score)))
    return {"score": score, "reasons": reasons}

def score_liquidity_volatility(df):
    """Score based on average volume and ATR (liquidity & volatility)"""
    recent = df.iloc[-1]
    vol20 = df["Volume"].tail(20).mean()
    price = recent["Close"]
    atr = recent.get("atr14", np.nan)
    score = 50
    reasons = []
    # Liquidity heuristic: higher avg volume relative to price is better
    if vol20 is not None and price > 0:
        turn = (vol20 * price)  # not perfect but proxy for daily turnover
        if turn > 1e8:
            score += 10; reasons.append("High daily turnover")
        elif turn > 1e7:
            score += 4; reasons.append("Moderate turnover")
        else:
            score -= 6; reasons.append("Low turnover")
    # Volatility: some volatility is fine; too high is risky
    if not np.isnan(atr) and price>0:
        atr_pct = atr / price
        if atr_pct < 0.01:
            score += 4; reasons.append("Low volatility")
        elif atr_pct < 0.04:
            score += 2; reasons.append("Moderate volatility")
        else:
            score -= 6; reasons.append("High volatility")
    score = max(0, min(100, int(score)))
    return {"score": score, "reasons": reasons}

def combine_scores(tech, fund, news, liq, weights=None):
    if weights is None:
        weights = {"tech": 0.5, "fund":0.25, "news":0.15, "liq":0.10}
    composite = (tech["score"] * weights["tech"] +
                 fund["score"] * weights["fund"] +
                 news["score"] * weights["news"] +
                 liq["score"] * weights["liq"])
    composite = int(round(composite))
    return composite

# ---------- Risk & Reward, sizing ----------
def compute_risk_reward(df, composite_score, user_stop=None, rr_target_mult=2):
    last = df.iloc[-1]
    price = last["Close"]
    atr = last.get("atr14", np.nan)
    # default stop: use support = price - 1.5*ATR if ATR available
    if user_stop:
        stop = user_stop
    else:
        stop = price - 1.5 * (atr if not np.isnan(atr) else price*0.02)
    if stop <= 0 or stop >= price:
        stop = price * 0.98  # fallback 2% stop
    # target = price + rr_target_mult * (price - stop)
    target = price + rr_target_mult * (price - stop)
    reward = target - price
    risk = price - stop
    rr = reward / risk if risk>0 else np.nan
    # Probabilities from composite_score approx:
    prob_win = composite_score / 100.0
    prob_loss = 1 - prob_win
    expected = prob_win * reward - prob_loss * risk
    return {
        "entry": price,
        "stop": round(stop, 4),
        "target": round(target, 4),
        "reward": round(reward, 4),
        "risk": round(risk, 4),
        "rr": round(rr, 3),
        "prob_win": round(prob_win, 3),
        "expected_value": round(expected, 4)
    }

def position_size(account_balance, risk_pct, entry, stop):
    """Return number of shares/units to buy given risk percent of account."""
    risk_amount = account_balance * (risk_pct/100.0)
    per_share_risk = abs(entry - stop)
    if per_share_risk <= 0:
        return 0
    qty = int(risk_amount / per_share_risk)
    return max(qty, 0)

# ---------- Streamlit UI ----------
st.title("Automated Stock Analysis — Technical + Fundamental + News")
st.write("Masukkan 1 atau lebih ticker (contoh: `BBCA.JK`, `TLKM.JK`, `AAPL`) — sistem akan menggabungkan sinyal dan memberi rekomendasi probabilistik.")

with st.sidebar:
    tickers_input = st.text_area("Tickers (comma separated)", value="BBCA.JK, TLKM.JK", height=80)
    period = st.selectbox("History period", options=["3mo", "6mo", "1y", "2y"], index=1)
    interval = st.selectbox("Interval", options=["1d", "1wk", "1h"], index=0)
    use_news = st.checkbox("Fetch news headlines (Google News fallback)", value=True)
    api_key_news = st.text_input("(Optional) NewsAPI key", placeholder="Leave blank to use Google News scraping")
    account_balance = st.number_input("Account balance (for position sizing)", value=10000000.0, step=100000.0, format="%.2f")
    risk_pct = st.slider("Risk per trade (%)", 0.1, 5.0, 1.0, 0.1)
    rr_mult = st.slider("Default target RR multiple", 1.0, 5.0, 2.0, 0.5)
    st.markdown("---")
    st.markdown("Scoring weights")
    w_tech = st.slider("Technical weight", 0.0, 1.0, 0.5, 0.05)
    w_fund = st.slider("Fundamental weight", 0.0, 1.0, 0.25, 0.05)
    w_news = st.slider("News weight", 0.0, 1.0, 0.15, 0.05)
    w_liq = st.slider("Liquidity weight", 0.0, 1.0, 0.10, 0.05)
    # normalize weights
    ssum = w_tech + w_fund + w_news + w_liq
    if ssum == 0:
        w_tech, w_fund, w_news, w_liq = 0.5, 0.25, 0.15, 0.10
    else:
        w_tech, w_fund, w_news, w_liq = w_tech/ssum, w_fund/ssum, w_news/ssum, w_liq/ssum

st.write("**Tickers to analyze:**", tickers_input)

tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
if not tickers:
    st.warning("Masukkan minimal 1 ticker di sidebar.")
    st.stop()

# Process each ticker
results = []
for t in tickers:
    st.markdown(f"---\n## {t}")
    try:
        df = fetch_price_data(t, period=period, interval=interval)
        df_ind = compute_technical_indicators(df)
        fund = fetch_fundamentals(t)  # may contain None fields
        headlines = []
        if use_news:
            try:
                if api_key_news:
                    # placeholder for NewsAPI usage (user must provide key)
                    # user can implement Article fetch via NewsAPI here.
                    pass
                headlines = google_news_headlines(t, max_headlines=6)
            except Exception as e:
                headlines = []
        news_sent = compute_news_sentiment(headlines) if headlines else {"score": 50, "details": []}
        tech_score = score_technical(df_ind)
        fund_score = score_fundamental(fund)
        liq_score = score_liquidity_volatility(df_ind)
        composite = combine_scores(tech_score, fund_score, news_sent, liq_score, weights={
            "tech": w_tech, "fund": w_fund, "news": w_news, "liq": w_liq
        })
        # map to recommendation label
        if composite >= 70:
            label = "BUY"
        elif composite >= 50:
            label = "WAIT"
        else:
            label = "AVOID"
        rr = compute_risk_reward(df_ind, composite, rr_target_mult=rr_mult)
        qty = position_size(account_balance, risk_pct, rr["entry"], rr["stop"])
        # Display summary cards
        col1, col2, col3, col4 = st.columns([2,2,2,2])
        col1.metric("Composite score", composite)
        col2.metric("Recommendation", label)
        col3.metric("Estimated prob win", f"{round(rr['prob_win']*100,1)}%")
        col4.metric("R:R", rr["rr"])
        # price chart
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df_ind["date"], open=df_ind["Open"], high=df_ind["High"],
            low=df_ind["Low"], close=df_ind["Close"], name="Price"
        ))
        fig.add_trace(go.Scatter(x=df_ind["date"], y=df_ind["ma20"], name="MA20", line=dict(width=1)))
        fig.add_trace(go.Scatter(x=df_ind["date"], y=df_ind["ma50"], name="MA50", line=dict(width=1)))
        fig.add_trace(go.Scatter(x=df_ind["date"], y=df_ind["ma200"], name="MA200", line=dict(width=1)))
        fig.update_layout(height=450, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)
        # Show fundamentals
        st.subheader("Fundamental snapshot")
        fund_df = pd.DataFrame(list(fund.items()), columns=["metric","value"])
        st.table(fund_df.set_index("metric"))
        # Show technical reasons
        st.subheader("Technical score breakdown")
        st.write("Score:", tech_score["score"])
        st.write("Reasons:", tech_score["reasons"])
        # Show fundamental reasons
        st.subheader("Fundamental score breakdown")
        st.write("Score:", fund_score["score"])
        st.write("Reasons:", fund_score["reasons"])
        # Show news sentiment
        st.subheader("News sentiment")
        st.write("News score:", news_sent["score"])
        for d in news_sent.get("details", [])[:6]:
            st.write("-", d["text"], f"(compound {round(d['compound'],3)})")
        # Show liquidity/volatility
        st.subheader("Liquidity / Volatility")
        st.write("Score:", liq_score["score"])
        st.write("Reasons:", liq_score["reasons"])
        # Risk & reward and position sizing
        st.subheader("Risk & Reward / Position Sizing")
        st.write(rr)
        st.write(f"Suggested position size (qty) for account {account_balance:,.0f} risking {risk_pct}%: **{qty}** shares/units")
        results.append({
            "ticker": t,
            "composite": composite,
            "label": label,
            "prob_win": rr["prob_win"],
            "rr": rr["rr"],
            "qty": qty
        })
    except Exception as e:
        st.error(f"Error processing {t}: {e}")

st.markdown("---")
st.subheader("Summary table")
if results:
    sr = pd.DataFrame(results).set_index("ticker")
    st.dataframe(sr)
