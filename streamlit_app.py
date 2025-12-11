# streamlit_app.py
# AI Stock Analyzer PRO — Final (API keys hidden from UI; safe_secret fallback)
# Run: streamlit run streamlit_app.py

import os
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import plotly.graph_objs as go
import requests
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import json
from datetime import datetime, timedelta
import math
import time
import logging
import traceback

# ML libs
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import lightgbm as lgb

# ---------------- Page config ----------------
st.set_page_config(layout="wide", page_title="AI Stock Analyzer PRO", initial_sidebar_state="expanded")
st.title("AI Stock Analyzer PRO")

# ---------------- Logging ----------------
logger = logging.getLogger("ai_stock_analyzer")
logger.setLevel(logging.INFO)
if "logs" not in st.session_state:
    st.session_state["logs"] = []

def append_log(msg):
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    full = f"[{ts} UTC] {msg}"
    st.session_state["logs"].append(full)
    logger.info(full)

# ---------------- Session-state init (optional dev entry) ----------------
for k in ["GOOGLE_API_KEY", "GROQ_API_KEY", "DEEPSEEK_API_KEY", "MISTRAL_API_KEY"]:
    if k not in st.session_state:
        st.session_state[k] = ""

# ---------------- Safe secret accessor ----------------
def safe_secret(key: str):
    """
    Return secret value from (in order):
    1) st.session_state (if user entered via dev button)
    2) st.secrets (Streamlit Cloud) - accessed safely
    3) os.environ (other hosting)
    If none found, return None.
    This avoids StreamlitSecretNotFoundError when no secrets.toml present.
    """
    # 1) session_state (explicit dev entry)
    try:
        v = st.session_state.get(key)
        if v:
            return v
    except Exception:
        pass

    # 2) st.secrets (Streamlit Cloud) - guard access
    try:
        # Use dict-style get only if st.secrets exists
        if hasattr(st, "secrets") and st.secrets is not None:
            # avoid 'in' which may trigger lookup in some contexts; use try/except
            try:
                v = st.secrets.get(key)
            except Exception:
                # older streamlit versions might not support .get -> try subscription
                try:
                    v = st.secrets[key]
                except Exception:
                    v = None
            if v:
                return v
    except Exception:
        pass

    # 3) environment variables
    try:
        v = os.environ.get(key)
        if v:
            return v
    except Exception:
        pass

    return None

# ---------------- Sidebar: settings & model selection (no key inputs) ----------------
with st.sidebar:
    st.header("Global settings")
    tickers_input = st.text_area("Tickers (comma separated)", value="BBCA.JK, TLKM.JK", height=80)
    mode = st.selectbox("Mode", ["Swing Trading", "Intraday"])
    timeframe_select = st.multiselect("Multi-timeframe", ["1D","4H","1H"], default=["1D"])

    # News toggle
    use_news = st.checkbox("Fetch news (Google News)", value=True)

    st.markdown("---")
    st.header("AI Provider & Model Selection (keys hidden)")
    ai_provider = st.selectbox(
        "AI Provider",
        ["Groq (Llama)", "Google Gemini", "DeepSeek", "Mistral"]
    )

    # Model selectors (no API key inputs here)
    if ai_provider == "Google Gemini":
        ai_model = st.selectbox("Gemini model", ["gemini-1.5-flash", "gemini-1.5-pro"])
    elif ai_provider == "Groq (Llama)":
        ai_model = st.selectbox("Groq model", ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "llama-3.3-70b-versatile"])
    elif ai_provider == "DeepSeek":
        ai_model = st.selectbox("DeepSeek model", ["deepseek-chat", "deepseek-coder"])
    else:  # Mistral
        ai_model = st.selectbox("Mistral model", ["mistral-large-latest", "mistral-small-latest"])

    st.markdown("---")
    st.subheader("Position sizing")
    account_balance = st.number_input("Account balance", value=10000000.0, step=100000.0, format="%.2f")
    risk_pct = st.slider("Risk per trade (%)", 0.1, 5.0, 1.0, 0.1)

    st.markdown("---")
    st.subheader("Telegram Alerts")
    enable_telegram = st.checkbox("Enable Telegram alert", value=False)
    telegram_bot_token = st.text_input("Telegram Bot Token (visible)", type="password")
    telegram_chat_id = st.text_input("Telegram Chat ID")

    st.markdown("---")
    st.subheader("Backtest / ML")
    enable_backtest = st.checkbox("Enable backtest (LightGBM)", value=False)
    backtest_horizon_days = st.number_input("Backtest horizon (days)", value=5, min_value=1)
    backtest_train_ratio = st.slider("Train ratio", 0.5, 0.9, 0.75, 0.05)

    st.markdown("---")
    st.caption("API keys are read from (session_state -> st.secrets -> env). They are NOT displayed here.")

    # Optional dev-only: one-time key entry (hidden unless button pressed)
    st.markdown("---")
    if st.button("Enter API keys (local dev)"):
        with st.form("dev_keys_form", clear_on_submit=False):
            st.write("One-time local entry. Keys will be kept in session only (not persistent to disk).")
            s_google = st.text_input("GOOGLE_API_KEY", type="password")
            s_groq = st.text_input("GROQ_API_KEY", type="password")
            s_deepseek = st.text_input("DEEPSEEK_API_KEY", type="password")
            s_mistral = st.text_input("MISTRAL_API_KEY", type="password")
            submitted = st.form_submit_button("Save keys to session")
            if submitted:
                st.session_state["GOOGLE_API_KEY"] = s_google or ""
                st.session_state["GROQ_API_KEY"] = s_groq or ""
                st.session_state["DEEPSEEK_API_KEY"] = s_deepseek or ""
                st.session_state["MISTRAL_API_KEY"] = s_mistral or ""
                st.success("Keys saved to session_state for this browser session (will not persist to disk).")

# ---------------- Constants & analyzer ----------------
INTERVAL_MAP = {"1D":("1y","1d"), "4H":("6mo","4h"), "1H":("3mo","1h")}
analyzer = SentimentIntensityAnalyzer()

# ---------------- Utility functions ----------------
def normalize_datetime_column(df):
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        return df
    for c in df.columns:
        if c.lower() in ["date","datetime","time"]:
            df = df.rename(columns={c:"date"})
            df["date"] = pd.to_datetime(df["date"])
            return df
    if isinstance(df.index, pd.DatetimeIndex):
        name = df.index.name or "index"
        df = df.reset_index().rename(columns={name:"date"})
        df["date"] = pd.to_datetime(df["date"])
        return df
    for c in df.columns:
        if df[c].dtype == object:
            try:
                parsed = pd.to_datetime(df[c], errors="coerce")
                if parsed.notna().sum() > len(df)*0.5:
                    df = df.rename(columns={c:"date"})
                    df["date"] = parsed
                    return df
            except Exception:
                continue
    raise KeyError("Could not find/normalize datetime column")

def safe_fetch_yfinance(ticker, period="6mo", interval="1d"):
    append_log(f"Fetching {ticker} period={period} interval={interval}")
    try:
        tk = yf.Ticker(ticker)
        df = tk.history(period=period, interval=interval, actions=False)
        if df is None:
            df = pd.DataFrame()
    except Exception as e:
        append_log(f"yfinance.Ticker.history error for {ticker}: {e}")
        df = pd.DataFrame()
    # fallback for intraday
    if df.empty and interval in ["1h","4h","30m","15m"]:
        append_log(f"No intraday from Ticker.history for {ticker} — fallback to yf.download")
        for p in ["7d","30d"]:
            try:
                df2 = yf.download(tickers=ticker, period=p, interval=interval, progress=False, threads=False)
                if isinstance(df2, pd.DataFrame) and not df2.empty:
                    df = df2
                    append_log(f"yf.download success for {ticker} period={p} interval={interval}")
                    break
            except Exception as e:
                append_log(f"yf.download error for {ticker} p={p}: {e}")
            time.sleep(0.3)
    # final fallback to daily 1y
    if (df is None) or (hasattr(df, "empty") and df.empty):
        try:
            append_log(f"Final fallback: Ticker.history 1y/1d for {ticker}")
            df = yf.Ticker(ticker).history(period="1y", interval="1d", actions=False)
        except Exception as e:
            append_log(f"final fallback error: {e}")
    if df is None or (hasattr(df, "empty") and df.empty):
        raise ValueError(f"No price data for {ticker} interval={interval}")

    df = df.reset_index()
    df = normalize_datetime_column(df)
    if "Volume" not in df.columns and "volume" in df.columns:
        df["Volume"] = df["volume"]
    for col in ["Open","High","Low","Close","Volume"]:
        if col not in df.columns:
            if col.lower() in df.columns:
                df[col] = df[col.lower()]
            else:
                append_log(f"Warning: {col} not found for {ticker}; filling NaN")
                df[col] = np.nan
    df = df.sort_values("date").reset_index(drop=True)
    return df

def fetch_price_data(ticker, period="6mo", interval="1d"):
    try:
        return safe_fetch_yfinance(ticker, period=period, interval=interval)
    except Exception as e:
        raise RuntimeError(f"fetch_price_data failed for {ticker} {period}/{interval}: {e}")

def compute_indicators(df):
    d = df.copy()
    for col in ["Open","High","Low","Close","Volume"]:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d["ma20"] = d["Close"].rolling(20).mean()
    d["ma50"] = d["Close"].rolling(50).mean()
    d["ma200"] = d["Close"].rolling(200).mean()
    d["rsi14"] = ta.rsi(d["Close"], length=14)
    macd = ta.macd(d["Close"])
    if macd is not None:
        d["macd"] = macd["MACD_12_26_9"]
        d["macd_signal"] = macd["MACDs_12_26_9"]
    d["atr14"] = ta.atr(d["High"], d["Low"], d["Close"], length=14)
    d["vol20"] = d["Volume"].rolling(20).mean()
    return d

def fetch_fundamentals(ticker):
    try:
        tk = yf.Ticker(ticker)
        info = tk.info or {}
        return {
            "shortName": info.get("shortName"),
            "sector": info.get("sector"),
            "marketCap": info.get("marketCap"),
            "trailingPE": info.get("trailingPE"),
            "priceToBook": info.get("priceToBook"),
            "trailingEps": info.get("trailingEps"),
            "returnOnEquity": info.get("returnOnEquity"),
            "debtToEquity": info.get("debtToEquity")
        }
    except Exception as e:
        append_log(f"fetch_fundamentals error for {ticker}: {e}")
        return {}

def google_news_headlines(query, max_headlines=6):
    if not query:
        return []
    try:
        q = requests.utils.quote(f"{query} saham OR stock")
        url = f"https://www.google.com/search?q={q}&tbm=nws"
        headers = {"User-Agent":"Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        articles = soup.select("div.dbsr")
        headlines = []
        for a in articles[:max_headlines]:
            title_tag = a.select_one("div.JheGif.nDgy9d")
            snippet_tag = a.select_one("div.Y3v8qd")
            title = title_tag.get_text() if title_tag else None
            snippet = snippet_tag.get_text() if snippet_tag else ""
            if title:
                headlines.append({"title": title, "snippet": snippet})
        return headlines
    except Exception as e:
        append_log(f"google_news_headlines error for {query}: {e}")
        return []

def compute_news_sentiment(headlines):
    if not headlines:
        return {"score":50,"details":[]}
    scores=[]; details=[]
    for h in headlines:
        text = (h.get("title","") + " " + h.get("snippet","")).strip()
        vs = analyzer.polarity_scores(text)
        scores.append(vs["compound"])
        details.append({"text":text,"compound":vs["compound"]})
    avg = np.mean(scores) if scores else 0.0
    norm = int((avg + 1)/2 * 100)
    return {"score": norm, "details": details}

# ---------------- Heuristics & ML helpers ----------------
def simple_scores(df, fund, news_sent):
    recent = df.iloc[-1]
    score = 50; reasons=[]
    price = recent["Close"]
    if not pd.isna(recent.get("ma50")):
        score += 8 if price > recent["ma50"] else -8; reasons.append("price vs ma50")
    if not pd.isna(recent.get("ma200")):
        score += 10 if price > recent["ma200"] else -10; reasons.append("price vs ma200")
    rsi = recent.get("rsi14")
    if not pd.isna(rsi):
        if rsi < 30: score +=6; reasons.append("RSI oversold")
        elif rsi > 70: score -=6; reasons.append("RSI overbought")
    pe = fund.get("trailingPE")
    if pe and pe>0:
        if pe<10: score +=6; reasons.append("low PE")
        elif pe<20: score +=2; reasons.append("moderate PE")
        else: score -=6; reasons.append("high PE")
    ns = news_sent.get("score",50)
    combined = score * 0.8 + ns * 0.2
    return int(max(0,min(100,round(combined)))), reasons

def compute_rr_heuristic(df, rr_mult=2):
    last = df.iloc[-1]
    price = last["Close"]
    atr = last.get("atr14", np.nan)
    stop = price - 1.5*(atr if not pd.isna(atr) else price*0.02)
    if stop <= 0 or stop >= price:
        stop = price * 0.98
    target = price + rr_mult * (price - stop)
    reward = target - price
    risk = price - stop
    rr_ratio = reward / risk if risk>0 else np.nan
    return {"entry":round(price,4), "stop":round(stop,4), "target":round(target,4), "rr":round(rr_ratio,3)}

def create_features_for_ml(df):
    dfc = df.copy()
    dfc = dfc.dropna().reset_index(drop=True)
    dfc["ma5"] = dfc["Close"].rolling(5).mean()
    dfc["ma10"] = dfc["Close"].rolling(10).mean()
    dfc["ma20"] = dfc["Close"].rolling(20).mean()
    dfc["rsi14"] = ta.rsi(dfc["Close"], length=14)
    dfc["atr14"] = ta.atr(dfc["High"], dfc["Low"], dfc["Close"], length=14)
    dfc["vol20"] = dfc["Volume"].rolling(20).mean()
    dfc["ret1"] = dfc["Close"].pct_change(1)
    dfc["ret5"] = dfc["Close"].pct_change(5)
    dfc = dfc.dropna().reset_index(drop=True)
    return dfc

def build_label(df, horizon=5, threshold_pct=0.01):
    df = df.copy()
    df["future_ret"] = df["Close"].shift(-horizon) / df["Close"] - 1
    df["label_up"] = (df["future_ret"] > threshold_pct).astype(int)
    df = df.dropna(subset=["label_up"]).reset_index(drop=True)
    return df

def train_lgb_model(df_features, train_ratio=0.75):
    features = [c for c in df_features.columns if c not in ["date","label_up","future_ret","Open","High","Low","Close","Volume","Adj Close","index"]]
    X = df_features[features]
    y = df_features["label_up"]
    split = int(len(X)*train_ratio)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    if len(X_test) < 1 or len(X_train) < 1:
        raise ValueError("Not enough data to train/test with given train_ratio.")
    lgb_train = lgb.Dataset(X_train, y_train)
    params = {"objective":"binary","metric":"binary_logloss","verbosity":-1}
    model = lgb.train(params, lgb_train, num_boost_round=200)
    preds_prob = model.predict(X_test)
    preds = (preds_prob > 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1": float(f1_score(y_test, preds, zero_division=0))
    }
    feat_imp = sorted(zip(features, model.feature_importance(importance_type='gain')), key=lambda x: x[1], reverse=True)
    return model, metrics, feat_imp, preds_prob, y_test

# ---------------- AI wrapper (multi-provider) ----------------
def ai_analyze_with_model(provider, model, payload, api_keys):
    prompt_header = (
        "Anda adalah analis saham profesional. Berikan analisis probabilistik dan rekomendasi BUY/WAIT/AVOID.\n"
        "Keluaran harus valid JSON dengan struktur:\n"
        "{\n"
        ' "rekomendasi": "BUY|WAIT|AVOID",\n'
        ' "probabilitas": {"naik_Xd":0.0,"turun_Xd":0.0},\n'
        ' "alasan": "...",\n'
        ' "risk_reward": {"entry":0.0,"stoploss":0.0,"target":0.0,"rr_ratio":0.0},\n'
        ' "catatan_risiko": "..." \n'
        "}\n\n"
    )
    prompt = prompt_header + "Data:\n" + json.dumps(payload, indent=2)

    try:
        if provider == "Google Gemini":
            try:
                import google.generativeai as genai
            except Exception as e:
                return None, f"Google Gemini SDK not installed: {e}"
            try:
                genai.configure(api_key=api_keys.get("gemini"))
            except Exception as e:
                return None, f"Failed to configure Gemini API key: {e}"
            model_name = model
            try:
                model_obj = genai.GenerativeModel(model_name)
                response = model_obj.generate_content(prompt)
                content = response.text.strip()
            except Exception as e:
                return None, f"Gemini API error: {e}"
            try:
                parsed = json.loads(content)
                return parsed, None
            except Exception:
                import re
                m = re.search(r"\{.*\}", content, flags=re.DOTALL)
                if m:
                    try:
                        parsed = json.loads(m.group(0))
                        return parsed, None
                    except Exception as e:
                        return None, f"Gemini response parse error: {e}"
                return None, "Gemini response not valid JSON."

        elif provider == "Groq (Llama)":
            try:
                from groq import Groq
            except Exception:
                return None, "Groq SDK not installed. Install `groq` or choose another provider."
            try:
                client = Groq(api_key=api_keys.get("groq"))
                resp = client.chat.completions.create(model=model, messages=[{"role":"user","content":prompt}])
                content = resp.choices[0].message.content
            except Exception as e:
                return None, f"Groq API error: {e}"
            try:
                return json.loads(content), None
            except Exception:
                import re
                m = re.search(r"\{.*\}", content, flags=re.DOTALL)
                if m:
                    return json.loads(m.group(0)), None
                return None, "Groq response not valid JSON."

        elif provider == "DeepSeek":
            key = api_keys.get("deepseek")
            if not key:
                return None, "DeepSeek API key not provided."
            url = "https://api.deepseek.com/chat/completions"
            headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
            body = {"model": model, "messages":[{"role":"user","content":prompt}]}
            try:
                r = requests.post(url, json=body, headers=headers, timeout=30)
                data = r.json()
                content = data["choices"][0]["message"]["content"]
            except Exception as e:
                return None, f"DeepSeek API error: {e}"
            try:
                return json.loads(content), None
            except Exception:
                import re
                m = re.search(r"\{.*\}", content, flags=re.DOTALL)
                if m:
                    return json.loads(m.group(0)), None
                return None, "DeepSeek response not valid JSON."

        elif provider == "Mistral":
            key = api_keys.get("mistral")
            if not key:
                return None, "Mistral API key not provided."
            url = "https://api.mistral.ai/v1/chat/completions"
            headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
            body = {"model": model, "messages":[{"role":"user","content":prompt}]}
            try:
                r = requests.post(url, json=body, headers=headers, timeout=30)
                data = r.json()
                content = data["choices"][0]["message"]["content"]
            except Exception as e:
                return None, f"Mistral API error: {e}"
            try:
                return json.loads(content), None
            except Exception:
                import re
                m = re.search(r"\{.*\}", content, flags=re.DOTALL)
                if m:
                    return json.loads(m.group(0)), None
                return None, "Mistral response not valid JSON."

        else:
            return None, f"Unknown provider: {provider}"
    except Exception as e:
        return None, f"AI wrapper unexpected error: {e}"

# ---------------- Telegram send helper ----------------
def send_telegram_alert(bot_token, chat_id, message):
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {"chat_id": chat_id, "text": message, "parse_mode": "HTML"}
        r = requests.post(url, data=payload, timeout=10)
        success = r.status_code == 200
        if not success:
            append_log(f"Telegram send failed status={r.status_code} text={r.text}")
        return success
    except Exception as e:
        append_log(f"Telegram send exception: {e}")
        return False

# ---------------- Main processing ----------------
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
if not tickers:
    st.warning("Masukkan minimal 1 ticker di sidebar.")
    st.stop()

# prepare api_keys dict (safe_secret reads from session_state, st.secrets, or env)
api_keys = {
    "gemini": safe_secret("GOOGLE_API_KEY"),
    "groq": safe_secret("GROQ_API_KEY"),
    "deepseek": safe_secret("DEEPSEEK_API_KEY"),
    "mistral": safe_secret("MISTRAL_API_KEY")
}

# Inform which providers have keys (without printing keys)
available_providers = [p for p,k in [("Gemini", api_keys["gemini"]), ("Groq", api_keys["groq"]), ("DeepSeek", api_keys["deepseek"]), ("Mistral", api_keys["mistral"])] if k]
if not any(api_keys.values()):
    st.warning("No AI API keys found. Add keys to ~/.streamlit/secrets.toml, Streamlit Cloud Secrets, or set environment variables. Optionally use 'Enter API keys (local dev)' button in sidebar.")
else:
    st.info(f"API keys found for: {', '.join(available_providers)} (keys hidden)")

summary = []

for ticker in tickers:
    st.markdown("---")
    st.header(f"{ticker}  — Mode: {mode}")
    per_ticker_result = {"ticker": ticker, "timeframes": {}}

    for tf in timeframe_select:
        st.subheader(f"Timeframe: {tf}")
        period, interval = INTERVAL_MAP.get(tf, ("6mo","1d"))

        try:
            df = fetch_price_data(ticker, period=period, interval=interval)
        except Exception as e:
            append_log(f"[{ticker} {tf}] fetch_price_data failed: {e}")
            st.error(f"Failed to fetch price data for {ticker} {tf}: {e}")
            per_ticker_result["timeframes"][tf] = {"error": str(e)}
            continue

        try:
            df = compute_indicators(df)
        except Exception as e:
            append_log(f"[{ticker} {tf}] compute_indicators failed: {e}")
            st.error(f"Indicator calculation failed for {ticker} {tf}: {e}")
            per_ticker_result["timeframes"][tf] = {"error": str(e)}
            continue

        fund = fetch_fundamentals(ticker)
        headlines = google_news_headlines(ticker) if use_news else []
        news_sent = compute_news_sentiment(headlines)

        try:
            heuristic_score, reasons = simple_scores(df, fund, news_sent)
        except Exception as e:
            append_log(f"[{ticker} {tf}] simple_scores error: {e}")
            heuristic_score, reasons = 50, ["error computing heuristic"]

        rr = compute_rr_heuristic(df, rr_mult=2)

        # Chart
        try:
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=df["date"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"
            ))
            if "ma20" in df.columns:
                fig.add_trace(go.Scatter(x=df["date"], y=df["ma20"], name="MA20", line=dict(width=1)))
                fig.add_trace(go.Scatter(x=df["date"], y=df["ma50"], name="MA50", line=dict(width=1)))
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            append_log(f"[{ticker} {tf}] Chart render error: {e}")
            st.error(f"Chart render error: {e}")

        last_price = float(df.iloc[-1]["Close"])
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Heuristic score", heuristic_score)
        c2.metric("Last close", f"{last_price:,.2f}")
        c3.metric("Heuristic R:R", rr.get("rr"))
        c4.metric("News score", news_sent.get("score",50))

        with st.expander("Fundamental snapshot"):
            try:
                st.table(pd.DataFrame(list(fund.items()), columns=["metric","value"]).set_index("metric"))
            except Exception as e:
                st.write("Fundamental fetch error:", e)
        with st.expander("Headlines"):
            if headlines:
                for h in headlines[:8]:
                    st.write("-", h.get("title"), "|", h.get("snippet"))
            else:
                st.write("No headlines found / disabled")

        payload = {
            "ticker": ticker,
            "timeframe": tf,
            "date": str(datetime.utcnow().date()),
            "technical": {
                "last_close": float(last_price),
                "ma20": float(df.iloc[-1].get("ma20")) if not pd.isna(df.iloc[-1].get("ma20")) else None,
                "ma50": float(df.iloc[-1].get("ma50")) if not pd.isna(df.iloc[-1].get("ma50")) else None,
                "ma200": float(df.iloc[-1].get("ma200")) if not pd.isna(df.iloc[-1].get("ma200")) else None,
                "rsi14": float(df.iloc[-1].get("rsi14")) if not pd.isna(df.iloc[-1].get("rsi14")) else None,
                "atr14": float(df.iloc[-1].get("atr14")) if not pd.isna(df.iloc[-1].get("atr14")) else None,
                "vol20": float(df.iloc[-1].get("vol20")) if not pd.isna(df.iloc[-1].get("vol20")) else None
            },
            "fundamental": fund,
            "news_sentiment": news_sent,
            "heuristic_score": heuristic_score,
            "heuristic_reasons": reasons,
            "heuristic_rr": rr
        }

        # Call AI wrapper
        try:
            ai_result, ai_err = ai_analyze_with_model(ai_provider, ai_model, payload, api_keys)
        except Exception as e:
            append_log(f"[{ticker} {tf}] AI wrapper exception: {e}\n{traceback.format_exc()}")
            ai_result, ai_err = None, str(e)

        if ai_err:
            st.error(f"AI error: {ai_err}")
        else:
            st.subheader("AI analysis")
            try:
                st.json(ai_result)
            except Exception:
                st.write("AI output (raw):", str(ai_result)[:1000])

        # Telegram alerts
        try:
            if enable_telegram and telegram_bot_token and telegram_chat_id and ai_result:
                reco = (ai_result.get("rekomendasi","") or "").upper()
                prob_map = ai_result.get("probabilitas",{}) or {}
                prob_keys = [k for k in prob_map.keys() if "naik" in k.lower() or "up" in k.lower()]
                prob_up = None
                if prob_keys:
                    prob_up = prob_map.get(prob_keys[0])
                should_alert = (reco == "BUY") or (prob_up is not None and prob_up >= 0.6)
                if should_alert:
                    msg = f"<b>{ticker} {tf}</b>\nReco: {reco}\nProb up: {prob_up}\nPrice: {last_price}\nEntry: {rr['entry']}\nStop: {rr['stop']}\nTarget: {rr['target']}\nTime: {datetime.utcnow().isoformat()} UTC"
                    sent = send_telegram_alert(telegram_bot_token, telegram_chat_id, msg)
                    if sent:
                        st.success("Telegram alert sent")
                        append_log(f"Telegram alert sent for {ticker} {tf}")
                    else:
                        st.error("Failed to send Telegram alert")
                        append_log(f"Telegram send failed for {ticker} {tf}")
        except Exception as e:
            append_log(f"Telegram logic exception: {e}")

        # Backtest ML
        if enable_backtest:
            st.subheader("Backtest & ML (LightGBM)")
            try:
                df_feat = create_features_for_ml(df)
                df_labeled = build_label(df_feat, horizon=int(backtest_horizon_days), threshold_pct=0.0)
                model, metrics, feat_imp, preds_prob, y_test = train_lgb_model(df_labeled, train_ratio=backtest_train_ratio)
                st.write("Backtest metrics (LightGBM):", metrics)
                st.write("Top feature importances:")
                st.table(pd.DataFrame(feat_imp[:10], columns=["feature","importance"]).set_index("feature"))
                latest_features = df_labeled.iloc[-1:]
                feature_cols = [c for c in df_labeled.columns if c not in ["date","label_up","future_ret","Open","High","Low","Close","Volume","Adj Close","index"]]
                latest_prob = model.predict(latest_features[feature_cols])[0]
                st.metric("Model predicted prob up (latest)", f"{latest_prob*100:.1f}%")
            except Exception as e:
                append_log(f"[{ticker} {tf}] Backtest error: {e}\n{traceback.format_exc()}")
                st.error(f"Backtest error: {e}")

        per_ticker_result["timeframes"][tf] = {
            "heuristic_score": heuristic_score,
            "ai_result": ai_result,
            "ai_err": ai_err,
            "rr": rr
        }

        time.sleep(0.25)

    summary.append(per_ticker_result)

# ---------------- Summary table ----------------
st.markdown("---")
st.subheader("Summary")
flat = []
for s in summary:
    t = s["ticker"]
    for tf, info in s["timeframes"].items():
        flat.append({
            "ticker": t,
            "timeframe": tf,
            "heuristic_score": info.get("heuristic_score"),
            "ai_reco": (info.get("ai_result") or {}).get("rekomendasi") if info.get("ai_result") else None,
            "ai_err": info.get("ai_err")
        })
if flat:
    st.dataframe(pd.DataFrame(flat).set_index(["ticker","timeframe"]))

# ---------------- Logs in sidebar ----------------
with st.sidebar.expander("Application logs (recent)"):
    logs = st.session_state.get("logs", [])
    if logs:
        st.write("\n".join(logs[-80:]))
    else:
        st.write("No logs yet.")

st.markdown("**Note**: Tool ini untuk riset. Validate with paper trading/backtest before live trading. Do not commit API keys to public repo.")
