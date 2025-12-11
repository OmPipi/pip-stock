# streamlit_app_pro.py
# AI Stock Analyzer PRO — Enhanced (prompt, multi-timeframe, SR, VSA, weighted sentiment, EV-based ML)
# Save and run: streamlit run streamlit_app_pro.py

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
import hashlib
import pickle
import math
import time
import logging
import traceback

# ML libs
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb

# ---------------- Page config ----------------
st.set_page_config(layout="wide", page_title="AI Stock Analyzer PRO — Enhanced", initial_sidebar_state="expanded")
st.title("AI Stock Analyzer PRO — Enhanced")

# ---------------- Logging ----------------
logger = logging.getLogger("ai_stock_analyzer_pro")
logger.setLevel(logging.INFO)
if "logs" not in st.session_state:
    st.session_state["logs"] = []

def append_log(msg):
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    full = f"[{ts} UTC] {msg}"
    st.session_state["logs"].append(full)
    logger.info(full)

# ---------------- Session-state init ----------------
for k in ["GOOGLE_API_KEY", "GROQ_API_KEY", "DEEPSEEK_API_KEY", "MISTRAL_API_KEY"]:
    if k not in st.session_state:
        st.session_state[k] = ""

# ---------------- Safe secret accessor ----------------
def safe_secret(key: str):
    try:
        v = st.session_state.get(key)
        if v:
            return v
    except Exception:
        pass
    try:
        if hasattr(st, "secrets") and st.secrets is not None:
            try:
                v = st.secrets.get(key)
            except Exception:
                try:
                    v = st.secrets[key]
                except Exception:
                    v = None
            if v:
                return v
    except Exception:
        pass
    try:
        v = os.environ.get(key)
        if v:
            return v
    except Exception:
        pass
    return None

# ---------------- Sidebar: settings & model selection ----------------
issi_stocks = [
	"AADI.JK", "AALI.JK", "ABMM.JK", "ACES.JK", "ACRO.JK", "ACST.JK", "ADCP.JK", "ADES.JK", "ADHI.JK",
	"ADMG.JK",
	"ADMR.JK", "ADRO.JK", "AGAR.JK", "AGII.JK", "AIMS.JK", "AISA.JK", "AKPI.JK", "AKRA.JK", "AKSI.JK",
	"ALDO.JK",
	"ALKA.JK", "AMAN.JK", "AMFG.JK", "AMIN.JK", "AMMN.JK", "ANDI.JK", "ANJT.JK", "ANTM.JK", "APII.JK",
	"APLI.JK",
	"APLN.JK", "ARCI.JK", "AREA.JK", "ARII.JK", "ARNA.JK", "ARTA.JK", "ASGR.JK", "ASHA.JK", "ASII.JK",
	"ASLC.JK",
	"ASPI.JK", "ASRI.JK", "ATAP.JK", "ATIC.JK", "ATLA.JK", "AUTO.JK", "AVIA.JK", "AWAN.JK", "AXIO.JK",
	"AYAM.JK",
	"AYLS.JK", "BABY.JK", "BAIK.JK", "BANK.JK", "BAPA.JK", "BAPI.JK", "BATR.JK", "BAUT.JK", "BAYU.JK",
	"BBRM.JK",
	"BBSS.JK", "BCIP.JK", "BDKR.JK", "BEEF.JK", "BELI.JK", "BELL.JK", "BESS.JK", "BEST.JK", "BIKE.JK",
	"BINO.JK",
	"BIPP.JK", "BIRD.JK", "BISI.JK", "BKDP.JK", "BKSL.JK", "BLES.JK", "BLTA.JK", "BLTZ.JK", "BLUE.JK",
	"BMHS.JK",
	"BMSR.JK", "BMTR.JK", "BNBR.JK", "BOAT.JK", "BOBA.JK", "BOGA.JK", "BOLA.JK", "BOLT.JK", "BRAM.JK",
	"BRIS.JK",
	"BRMS.JK", "BRNA.JK", "BRPT.JK", "BRRC.JK", "BSBK.JK", "BSDE.JK", "BSML.JK", "BSSR.JK", "BTON.JK",
	"BTPS.JK",
	"BUAH.JK", "BUDI.JK", "BUKK.JK", "BUMI.JK", "BUVA.JK", "BYAN.JK", "CAKK.JK", "CAMP.JK", "CANI.JK",
	"CARE.JK",
	"CASS.JK", "CBDK.JK", "CBPE.JK", "CBRE.JK", "CCSI.JK", "CEKA.JK", "CGAS.JK", "CHEM.JK", "CINT.JK",
	"CITA.JK",
	"CITY.JK", "CLEO.JK", "CLPI.JK", "CMNP.JK", "CMPP.JK", "CMRY.JK", "CNMA.JK", "COAL.JK", "CPIN.JK",
	"CPRO.JK",
	"CRAB.JK", "CRSN.JK", "CSAP.JK", "CSIS.JK", "CSMI.JK", "CSRA.JK", "CTBN.JK", "CTRA.JK", "CUAN.JK",
	"CYBR.JK",
	"DAAZ.JK", "DATA.JK", "DAYA.JK", "DCII.JK", "DEPO.JK", "DEWA.JK", "DEWI.JK", "DGIK.JK", "DGNS.JK",
	"DGWG.JK",
	"DIVA.JK", "DKFT.JK", "DKHH.JK", "DMAS.JK", "DMMX.JK", "DMND.JK", "DOOH.JK", "DOSS.JK", "DRMA.JK",
	"DSFI.JK",
	"DSNG.JK", "DSSA.JK", "DUTI.JK", "DVLA.JK", "DWGL.JK", "DYAN.JK", "EAST.JK", "ECII.JK", "EDGE.JK",
	"EKAD.JK",
	"ELIT.JK", "ELPI.JK", "ELSA.JK", "ELTY.JK", "EMDE.JK", "EMTK.JK", "ENAK.JK", "ENRG.JK", "ENZO.JK",
	"EPAC.JK",
	"EPMT.JK", "ERAA.JK", "ERAL.JK", "ESIP.JK", "ESSA.JK", "ESTA.JK", "EXCL.JK", "FAPA.JK", "FAST.JK",
	"FILM.JK",
	"FIRE.JK", "FISH.JK", "FITT.JK", "FMII.JK", "FOLK.JK", "FOOD.JK", "FORE.JK", "FPNI.JK", "FUTR.JK",
	"FWCT.JK",
	"GDST.JK", "GDYR.JK", "GEMA.JK", "GEMS.JK", "GGRP.JK", "GHON.JK", "GIAA.JK", "GJTL.JK", "GLVA.JK",
	"GMTD.JK",
	"GOLD.JK", "GOLF.JK", "GOOD.JK", "GPRA.JK", "GPSO.JK", "GRIA.JK", "GRPH.JK", "GTBO.JK", "GTRA.JK",
	"GTSI.JK",
	"GULA.JK", "GUNA.JK", "GWSA.JK", "GZCO.JK", "HADE.JK", "HAIS.JK", "HALO.JK", "HATM.JK", "HDIT.JK",
	"HEAL.JK",
	"HERO.JK", "HEXA.JK", "HGII.JK", "HILL.JK", "HOKI.JK", "HOMI.JK", "HOPE.JK", "HRME.JK", "HRUM.JK",
	"HUMI.JK",
	"HYGN.JK", "IATA.JK", "IBST.JK", "ICBP.JK", "ICON.JK", "IDPR.JK", "IFII.JK", "IFSH.JK", "IGAR.JK",
	"IIKP.JK",
	"IKAI.JK", "IKAN.JK", "IKBI.JK", "IKPM.JK", "IMPC.JK", "INCI.JK", "INCO.JK", "INDF.JK", "INDR.JK",
	"INDS.JK",
	"INDX.JK", "INDY.JK", "INET.JK", "INKP.JK", "INTD.JK", "INTP.JK", "IOTF.JK", "IPCC.JK", "IPCM.JK",
	"IPOL.JK",
	"IPTV.JK", "IRRA.JK", "IRSX.JK", "ISAT.JK", "ISSP.JK", "ITMA.JK", "ITMG.JK", "JARR.JK", "JAST.JK",
	"JATI.JK",
	"JAWA.JK", "JAYA.JK", "JECC.JK", "JGLE.JK", "JIHD.JK", "JKON.JK", "JMAS.JK", "JPFA.JK", "JRPT.JK",
	"JSMR.JK",
	"JSPT.JK", "JTPE.JK", "KAQI.JK", "KARW.JK", "KBAG.JK", "KBLI.JK", "KBLM.JK", "KDSI.JK", "KDTN.JK",
	"KEEN.JK",
	"KEJU.JK", "KETR.JK", "KIAS.JK", "KICI.JK", "KIJA.JK", "KINO.JK", "KIOS.JK", "KKES.JK", "KKGI.JK",
	"KLAS.JK",
	"KLBF.JK", "KMDS.JK", "KOCI.JK", "KOIN.JK", "KOKA.JK", "KONI.JK", "KOPI.JK", "KOTA.JK", "KPIG.JK",
	"KREN.JK",
	"KRYA.JK", "KSIX.JK", "KUAS.JK", "LABA.JK", "LABS.JK", "LAJU.JK", "LAND.JK", "LCKM.JK", "LION.JK",
	"LIVE.JK",
	"LMPI.JK", "LMSH.JK", "LPCK.JK", "LPIN.JK", "LPKR.JK", "LPPF.JK", "LRNA.JK", "LSIP.JK", "LTLS.JK",
	"LUCK.JK",
	"MAHA.JK", "MAIN.JK", "MAPA.JK", "MAPB.JK", "MAPI.JK", "MARK.JK", "MASA.JK", "MAXI.JK", "MBAP.JK",
	"MBMA.JK",
	"MBTO.JK", "MCAS.JK", "MCOL.JK", "MDIY.JK", "MDKA.JK", "MDKI.JK", "MDLA.JK", "MEDC.JK", "MEDS.JK",
	"MERK.JK",
	"META.JK", "MFMI.JK", "MHKI.JK", "MICE.JK", "MIDI.JK", "MIKA.JK", "MINA.JK", "MINE.JK", "MIRA.JK",
	"MITI.JK",
	"MKAP.JK", "MKPI.JK", "MKTR.JK", "MLIA.JK", "MLPL.JK", "MLPT.JK", "MMIX.JK", "MMLP.JK", "MNCN.JK",
	"MORA.JK",
	"MPIX.JK", "MPMX.JK", "MPOW.JK", "MPPA.JK", "MPRO.JK", "MRAT.JK", "MSIN.JK", "MSJA.JK", "MSKY.JK",
	"MSTI.JK",
	"MTDL.JK", "MTEL.JK", "MTLA.JK", "MTMH.JK", "MTSM.JK", "MUTU.JK", "MYOH.JK", "MYOR.JK", "NAIK.JK",
	"NASA.JK",
	"NASI.JK", "NCKL.JK", "NELY.JK", "NEST.JK", "NETV.JK", "NFCX.JK", "NICE.JK", "NICL.JK", "NPGF.JK",
	"NRCA.JK",
	"NTBK.JK", "NZIA.JK", "OBMD.JK", "OILS.JK", "OKAS.JK", "OMED.JK", "OMRE.JK", "OPMS.JK", "PADA.JK",
	"PAMG.JK",
	"PANI.JK", "PANR.JK", "PART.JK", "PBID.JK", "PBSA.JK", "PCAR.JK", "PDPP.JK", "PEHA.JK", "PEVE.JK",
	"PGAS.JK",
	"PGEO.JK", "PGUN.JK", "PICO.JK", "PIPA.JK", "PJAA.JK", "PKPK.JK", "PLIN.JK", "PMJS.JK", "PNBS.JK",
	"PNGO.JK",
	"PNSE.JK", "POLI.JK", "PORT.JK", "POWR.JK", "PPRE.JK", "PPRI.JK", "PRAY.JK", "PRDA.JK", "PRIM.JK",
	"PSAB.JK",
	"PSDN.JK", "PSGO.JK", "PSKT.JK", "PSSI.JK", "PTBA.JK", "PTIS.JK", "PTMP.JK", "PTMR.JK", "PTPP.JK",
	"PTPS.JK",
	"PTPW.JK", "PTRO.JK", "PTSN.JK", "PTSP.JK", "PURA.JK", "PURI.JK", "PWON.JK", "PZZA.JK", "RAAM.JK",
	"RAJA.JK",
	"RALS.JK", "RANC.JK", "RATU.JK", "RBMS.JK", "RDTX.JK", "REAL.JK", "RGAS.JK", "RIGS.JK", "RISE.JK",
	"RMKE.JK",
	"RMKO.JK", "ROCK.JK", "RODA.JK", "RONY.JK", "ROTI.JK", "RSCH.JK", "RSGK.JK", "RUIS.JK", "SAFE.JK",
	"SAGE.JK",
	"SAME.JK", "SAMF.JK", "SAPX.JK", "SATU.JK", "SBMA.JK", "SCCO.JK", "SCNP.JK", "SCPI.JK", "SEMA.JK",
	"SGER.JK",
	"SGRO.JK", "SHID.JK", "SICO.JK", "SIDO.JK", "SILO.JK", "SIMP.JK", "SIPD.JK", "SKBM.JK", "SKLT.JK",
	"SKRN.JK",
	"SLIS.JK", "SMAR.JK", "SMBR.JK", "SMCB.JK", "SMDM.JK", "SMDR.JK", "SMGA.JK", "SMGR.JK", "SMIL.JK",
	"SMKL.JK",
	"SMLE.JK", "SMMT.JK", "SMRA.JK", "SMSM.JK", "SNLK.JK", "SOCI.JK", "SOHO.JK", "SOLA.JK", "SONA.JK",
	"SOSS.JK",
	"SOTS.JK", "SPMA.JK", "SPTO.JK", "SRAJ.JK", "SRTG.JK", "SSIA.JK", "SSTM.JK", "STAA.JK", "STTP.JK",
	"SULI.JK",
	"SUNI.JK", "SUPR.JK", "SURI.JK", "SWID.JK", "TALF.JK", "TAMA.JK", "TAMU.JK", "TAPG.JK", "TARA.JK",
	"TAXI.JK",
	"TBMS.JK", "TCID.JK", "TCPI.JK", "TEBE.JK", "TFAS.JK", "TFCO.JK", "TGKA.JK", "TINS.JK", "TIRA.JK",
	"TIRT.JK",
	"TKIM.JK", "TLDN.JK", "TLKM.JK", "TMAS.JK", "TMPO.JK", "TNCA.JK", "TOBA.JK", "TOOL.JK", "TOSK.JK",
	"TOTL.JK",
	"TOTO.JK", "TPIA.JK", "TPMA.JK", "TRIS.JK", "TRON.JK", "TRST.JK", "TRUE.JK", "TRUK.JK", "TSPC.JK",
	"TYRE.JK",
	"UANG.JK", "UCID.JK", "UFOE.JK", "ULTJ.JK", "UNIC.JK", "UNIQ.JK", "UNTD.JK", "UNTR.JK", "UNVR.JK",
	"UVCR.JK",
	"VAST.JK", "VERN.JK", "VICI.JK", "VISI.JK", "VOKS.JK", "WAPO.JK", "WEGE.JK", "WEHA.JK", "WIFI.JK",
	"WINR.JK",
	"WINS.JK", "WIRG.JK", "WMUU.JK", "WOOD.JK", "WOWS.JK", "WTON.JK", "YELO.JK", "YPAS.JK", "YUPI.JK",
	"ZATA.JK",
	"ZONE.JK", "ZYRX.JK"
]

with st.sidebar:
    st.header("Global settings")
    tickers_input = st.multiselect("Tickers (select)", options=issi_stocks, default=["BRPT.JK","ITMG.JK"])
    mode = st.selectbox("Mode", ["Swing Trading", "Intraday"])
    timeframe_select = st.multiselect("Multi-timeframe", ["1D","4H","1H"], default=["1D"])
    use_news = st.checkbox("Fetch news (Google News)", value=True)

    st.markdown("---")
    st.header("AI Provider & Model Selection (keys hidden)")
    ai_provider = st.selectbox("AI Provider", ["Groq (Llama)", "Google Gemini", "DeepSeek", "Mistral"])
    if ai_provider == "Google Gemini":
        ai_model = st.selectbox("Gemini model", ["gemini-1.5-flash", "gemini-1.5-pro"])
    elif ai_provider == "Groq (Llama)":
        ai_model = st.selectbox("Groq model", ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "llama-3.3-70b-versatile"])
    elif ai_provider == "DeepSeek":
        ai_model = st.selectbox("DeepSeek model", ["deepseek-chat", "deepseek-coder"])
    else:
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
    enable_backtest = st.checkbox("Enable backtest (LightGBM EV Regression)", value=False)
    backtest_horizon_days = st.number_input("Backtest horizon (days)", value=5, min_value=1)
    backtest_train_ratio = st.slider("Train ratio", 0.5, 0.9, 0.75, 0.05)

    st.markdown("---")
    st.caption("API keys are read from (session_state -> st.secrets -> env). They are NOT displayed here.")

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

@st.cache_data(show_spinner=False)
def fetch_price_data_cached(ticker, period="6mo", interval="1d"):
    try:
        return safe_fetch_yfinance(ticker, period=period, interval=interval)
    except Exception as e:
        append_log(f"fetch_price_data_cached error for {ticker} {period}/{interval}: {e}")
        return pd.DataFrame()

def compute_indicators(df):
    d = df.copy()
    for col in ["Open","High","Low","Close","Volume"]:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d["ma5"] = d["Close"].rolling(5).mean()
    d["ma10"] = d["Close"].rolling(10).mean()
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
    # candle body and wick ratios
    d["body"] = (d["Close"] - d["Open"]).abs()
    d["upper_wick"] = d["High"] - d[["Close","Open"]].max(axis=1)
    d["lower_wick"] = d[["Close","Open"]].min(axis=1) - d["Low"]
    return d

def compute_sr_pivots(df):
    df2 = df.copy().reset_index(drop=True)
    df2['pivot_high'] = (df2['High'] > df2['High'].shift(1)) & (df2['High'] > df2['High'].shift(-1))
    df2['pivot_low']  = (df2['Low'] < df2['Low'].shift(1)) & (df2['Low'] < df2['Low'].shift(-1))
    resistances = list(df2[df2['pivot_high']].tail(8)['High'].round(2))
    supports = list(df2[df2['pivot_low']].tail(8)['Low'].round(2))
    # return top 3 recent
    return supports[-3:], resistances[-3:]

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

@st.cache_data(ttl=1800, show_spinner=False)  # cache 30 menit
def google_news_headlines(query, max_headlines=8):
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
            source_tag = a.select_one("div.CEMjEf span")
            title = title_tag.get_text() if title_tag else None
            snippet = snippet_tag.get_text() if snippet_tag else ""
            source = source_tag.get_text() if source_tag else ""
            if title:
                headlines.append({"title": title, "snippet": snippet, "source": source})
        return headlines
    except Exception as e:
        append_log(f"google_news_headlines error for {query}: {e}")
        return []

def compute_news_sentiment_weighted(headlines):
    if not headlines:
        return {"score":50,"details":[]}
    scores = []; details = []
    for h in headlines:
        text = (h.get("title","") + " " + h.get("snippet","")).strip()
        vs = analyzer.polarity_scores(text)
        weight = 1.0
        src = (h.get("source") or "").lower()
        # weight major financial outlets higher; penalize negative heavier (asymmetric impact)
        if any(x in src for x in ["cnn", "reuters", "bloomberg", "jakarta post", "kompas"]):
            weight = 1.5
        if "ihsg" in text.lower() or "ojk" in text.lower() or "rupiah" in text.lower() or "bank indonesia" in text.lower():
            weight = 2.0
        # asymmetry: negative headlines have slightly higher impact
        asym = 1.25 if vs["compound"] < -0.05 else 1.0
        final_weight = weight * asym
        compound_weighted = vs["compound"] * final_weight
        scores.append(compound_weighted)
        details.append({"text":text,"compound":vs["compound"], "weight":final_weight})
    avg = np.mean(scores) if scores else 0.0
    # normalize to 0-100
    norm = int((avg + 1)/2 * 100)
    norm = max(0, min(100, norm))
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

def compute_rr_heuristic(df, rr_mult=2, trend_strength=50):
    last = df.iloc[-1]
    price = last["Close"]
    atr = last.get("atr14", np.nan)
    # dynamic ATR multiplier based on trend_strength
    if np.isnan(atr) or atr <= 0:
        atr = price * 0.02
    if trend_strength >= 65:
        sl_mult = 1.2
    elif trend_strength >= 45:
        sl_mult = 1.5
    else:
        sl_mult = 1.8
    stop = price - sl_mult * atr
    if stop <= 0 or stop >= price:
        stop = price * 0.98
    target = price + rr_mult * (price - stop)
    reward = target - price
    risk = price - stop
    rr_ratio = reward / risk if risk>0 else np.nan
    return {"entry":round(price,4), "stop":round(stop,4), "target":round(target,4), "rr":round(rr_ratio,3), "atr_mult":sl_mult}

def create_features_for_ml(df):
    dfc = df.copy()

    dfc["ma5"] = dfc["Close"].rolling(5).mean()
    dfc["ma10"] = dfc["Close"].rolling(10).mean()
    dfc["ma20"] = dfc["Close"].rolling(20).mean()
    dfc["rsi14"] = ta.rsi(dfc["Close"], length=14)
    dfc["atr14"] = ta.atr(dfc["High"], dfc["Low"], dfc["Close"], length=14)
    dfc["vol20"] = dfc["Volume"].rolling(20).mean()

    # returns
    dfc["ret1"] = dfc["Close"].pct_change(1)
    dfc["ret5"] = dfc["Close"].pct_change(5)

    # additional features
    dfc["pct_vs_ma50"] = dfc["Close"] / dfc["ma50"] - 1
    dfc["body_wick_ratio"] = dfc["body"] / (dfc["upper_wick"] + dfc["lower_wick"] + 1e-9)
    dfc["vol_spike"] = dfc["Volume"] / (dfc["vol20"] + 1e-9)

    # VSA encoding
    if "vsa_signal" in dfc.columns:
        dfc["vsa_enc"] = dfc["vsa_signal"].map({
            "accumulation": 1,
            "distribution": -1
        }).fillna(0)
        dfc = dfc.drop(columns=["vsa_signal"])

    dfc = dfc.dropna().reset_index(drop=True)
    return dfc

def build_label_ev(df, horizon=5):
    df = df.copy()
    df["future_ret"] = df["Close"].shift(-horizon) / df["Close"] - 1
    df = df.dropna(subset=["future_ret"]).reset_index(drop=True)
    return df

def train_lgb_model_ev(df_features, train_ratio=0.75):
    features = [c for c in df_features.columns if c not in ["date","future_ret","Open","High","Low","Close","Volume","Adj Close","index"]]
    X = df_features[features]
    y = df_features["future_ret"]
    split = int(len(X)*train_ratio)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    if len(X_test) < 1 or len(X_train) < 1:
        raise ValueError("Not enough data to train/test with given train_ratio.")
    lgb_train = lgb.Dataset(X_train, y_train)
    params = {"objective":"regression","metric":"rmse","verbosity":-1}
    model = lgb.train(params, lgb_train, num_boost_round=300)
    preds = model.predict(X_test)
    metrics = {
        "rmse": float(mean_squared_error(y_test, preds) ** 0.5),
        "r2": float(r2_score(y_test, preds))
    }
    feat_imp = sorted(zip(features, model.feature_importance(importance_type='gain')), key=lambda x: x[1], reverse=True)
    return model, metrics, feat_imp, preds, y_test

# ---------------- AI wrapper (multi-provider) ----------------
# Replaced prompt header with quantitative decision prompt (EV-based)
def build_prompt(payload):
    prompt_header = (
        "Anda adalah analis kuantitatif pasar modal Indonesia. Gunakan data teknikal, fundamental, sentimen, "
        "multi-timeframe, dan heuristic untuk memberikan rekomendasi swing trading berbasis Expected Value (EV). "
        "Fokus pada probabilitas dan angka: jangan menulis opini tanpa data atau angka. Gunakan aturan berikut: "
        "BUY jika trend_strength > 60 DAN probabilitas naik_5d > 0.55 DAN rr_ratio >= 2.0. WAIT jika sinyal belum terkonfirmasi. "
        "AVOID jika trend turun atau probabilitas naik_5d < 0.45. "
        "Output harus valid JSON dengan nilai numerik. Struktur JSON wajib:\n"
        "{\n"
        ' "rekomendasi": "BUY|WAIT|AVOID",\n'
        ' "probabilitas": {"naik_3d":0.0,"naik_5d":0.0,"turun_3d":0.0,"turun_5d":0.0},\n'
        ' "trend_strength":0,\n'
        ' "alasan":"...",\n'
        ' "risk_reward": {"entry":0.0,"stoploss":0.0,"target_tp1":0.0,"target_tp2":0.0,"rr_ratio":0.0},\n'
        ' "expected_value":0.0,\n'
        ' "catatan_risiko":"..."\n'
        "}\n\n"
    )
    # include the payload data
    prompt = prompt_header + "Data:\n" + json.dumps(payload, indent=2, default=str)
    return prompt

@st.cache_data(show_spinner=False)
def cached_ai_call(provider, model, payload, api_keys):
    key = hashlib.md5(
        pickle.dumps((provider, model, payload))
    ).hexdigest()
    # execute AI
    result, err = ai_analyze_with_model(provider, model, payload, api_keys)
    return result, err

def ai_analyze_with_model(provider, model, payload, api_keys):
    prompt = build_prompt(payload)
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

def render_ai_result(ai_output, container=None, show_raw=False):
    ui = container if container is not None else st
    if ai_output is None:
        ui.write("No AI analysis available.")
        return
    parsed = None
    if isinstance(ai_output, dict):
        parsed = ai_output
    else:
        try:
            parsed = json.loads(ai_output)
        except:
            import re
            m = re.search(r"\{.*\}", str(ai_output), flags=re.DOTALL)
            if m:
                try:
                    parsed = json.loads(m.group(0))
                except:
                    parsed = None
    if parsed is None:
        ui.error("AI response could not be parsed.")
        if show_raw:
            ui.code(str(ai_output)[:2000])
        return
    rekom = parsed.get("rekomendasi", "-")
    prob = parsed.get("probabilitas", {})
    rr = parsed.get("risk_reward", {})
    alasan = parsed.get("alasan", "")
    risiko = parsed.get("catatan_risiko", "")
    ui.markdown(
        f"""
        <div style='padding:12px 20px; border-radius:12px; 
        background-color:#1e1e1e; border:1px solid #444; margin-bottom:20px;'>
            <div style='font-size:22px; opacity:0.8;'>Recommendation</div>
            <div style='font-size:40px; font-weight:700; margin-top:-5px;'>{rekom.upper()}</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    colA, colB = ui.columns(2)
    with colA:
        ui.markdown("#### Probabilities")
        if prob:
            rows = []
            for k, v in prob.items():
                try:
                    val = float(v)
                    if 0 <= val <= 1:
                        val *= 100
                    val = f"{val:.1f}%"
                except:
                    val = str(v)
                rows.append({"metric": k, "value": val})
            df_prob = pd.DataFrame(rows)
            ui.table(df_prob)
        else:
            ui.write("No probability data provided.")
    with colB:
        ui.markdown("#### Risk & Reward")
        if rr:
            clean_rr = []
            for key, val in rr.items():
                clean_rr.append({"field": key, "value": val})
            df_rr = pd.DataFrame(clean_rr)
            ui.table(df_rr)
        else:
            ui.write("No risk/reward data.")
    if alasan:
        with ui.expander("AI Explanation / Reasons"):
            ui.markdown(f"<div style='line-height:1.6;'>{alasan}</div>", unsafe_allow_html=True)
    if risiko:
        with ui.expander("Risk Notes"):
            ui.markdown(f"<div style='line-height:1.6;'>{risiko}</div>", unsafe_allow_html=True)
    if show_raw:
        with ui.expander("Raw AI Output (debug)"):
            ui.json(parsed)

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
tickers = tickers_input
if not tickers:
    st.warning("Masukkan minimal 1 ticker di sidebar.")
    st.stop()

api_keys = {
    "gemini": safe_secret("GOOGLE_API_KEY"),
    "groq": safe_secret("GROQ_API_KEY"),
    "deepseek": safe_secret("DEEPSEEK_API_KEY"),
    "mistral": safe_secret("MISTRAL_API_KEY")
}
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

    # Fetch multi-timeframe datasets (1D, 4H, 1H) if requested
    tf_dfs = {}
    for tf in ["1D","4H","1H"]:
        period, interval = INTERVAL_MAP.get(tf, ("6mo","1d"))
        try:
            df_tf = fetch_price_data_cached(ticker, period=period, interval=interval)
            if df_tf.empty:
                append_log(f"No data for {ticker} {tf}")
            else:
                df_tf = compute_indicators(df_tf)
            tf_dfs[tf] = df_tf
        except Exception as e:
            append_log(f"Error fetching {ticker} {tf}: {e}")
            tf_dfs[tf] = pd.DataFrame()

    for tf in (timeframe_select or ["1D"]):
        st.subheader(f"Timeframe: {tf}")
        df = tf_dfs.get(tf)
        if df is None or df.empty:
            st.error(f"No price data for {ticker} {tf}")
            per_ticker_result["timeframes"][tf] = {"error":"no_data"}
            continue

        fund = fetch_fundamentals(ticker)
        headlines = google_news_headlines(ticker) if use_news else []
        news_sent = compute_news_sentiment_weighted(headlines)

        try:
            heuristic_score, reasons = simple_scores(df, fund, news_sent)
        except Exception as e:
            append_log(f"[{ticker} {tf}] simple_scores error: {e}")
            heuristic_score, reasons = 50, ["error computing heuristic"]

        # compute SR pivots and VSA
        supports, resistances = compute_sr_pivots(df)
        df["vsa_signal"] = np.where(
            (df["Close"] > df["Open"]) & (df["Volume"] > df["vol20"]), "accumulation",
            np.where((df["Close"] < df["Open"]) & (df["Volume"] > df["vol20"]), "distribution", "neutral")
        )

        # Multi-timeframe summary: compute simple trend_strength
        trend_strength = 50
        # heavier weight to higher timeframe
        tf_weights = {"1D":0.5, "4H":0.3, "1H":0.2}
        trend_score_acc = 0.0
        weight_sum = 0.0
        for tff, df_t in tf_dfs.items():
            if df_t is None or df_t.empty:
                continue
            recent = df_t.iloc[-1]
            score_local = 50
            # price vs ma50/ma200 signals
            if not pd.isna(recent.get("ma50")):
                score_local += 20 if recent["Close"] > recent["ma50"] else -20
            if not pd.isna(recent.get("ma200")):
                score_local += 20 if recent["Close"] > recent["ma200"] else -20
            # rsi momentum
            rsi = recent.get("rsi14")
            if not pd.isna(rsi):
                if rsi < 30: score_local += 10
                elif rsi > 70: score_local -= 10
            w = tf_weights.get(tff, 0.0)
            trend_score_acc += score_local * w
            weight_sum += w
        if weight_sum > 0:
            trend_strength = int(max(0,min(100, round(trend_score_acc / weight_sum))))

        rr = compute_rr_heuristic(df, rr_mult=2, trend_strength=trend_strength)

        # Chart
        try:
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=df["date"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"
            ))
            if "ma20" in df.columns:
                fig.add_trace(go.Scatter(x=df["date"], y=df["ma20"], name="MA20", line=dict(width=1)))
                fig.add_trace(go.Scatter(x=df["date"], y=df["ma50"], name="MA50", line=dict(width=1)))
            # show support/resistance as horizontal lines (recent)
            for s in supports:
                fig.add_hline(y=s, line_dash="dot", line_color="green", annotation_text=f"S {s}", annotation_position="bottom left")
            for r in resistances:
                fig.add_hline(y=r, line_dash="dot", line_color="red", annotation_text=f"R {r}", annotation_position="top left")
            fig.update_layout(height=480, margin=dict(l=10, r=10, t=30, b=10))
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
                    st.write("-", h.get("title"), "|", h.get("source"), "|", h.get("snippet"))
            else:
                st.write("No headlines found / disabled")

        # Build multi-timeframe feature payload for AI
        def extract_feature_summary(df_local):
            if df_local is None or df_local.empty:
                return {}
            last = df_local.iloc[-1]
            return {
                "last_close": float(last["Close"]),
                "ma20": float(last.get("ma20")) if not pd.isna(last.get("ma20")) else None,
                "ma50": float(last.get("ma50")) if not pd.isna(last.get("ma50")) else None,
                "ma200": float(last.get("ma200")) if not pd.isna(last.get("ma200")) else None,
                "rsi14": float(last.get("rsi14")) if not pd.isna(last.get("rsi14")) else None,
                "atr14": float(last.get("atr14")) if not pd.isna(last.get("atr14")) else None,
                "vol20": float(last.get("vol20")) if not pd.isna(last.get("vol20")) else None,
                "vsa": str(last.get("vsa_signal")) if "vsa_signal" in last.index else None
            }

        multi_tf_payload = {k: extract_feature_summary(v) for k,v in tf_dfs.items()}

        payload = {
            "ticker": ticker,
            "timeframe": tf,
            "date": str(datetime.utcnow().date()),
            "technical": extract_feature_summary(df),
            "multi_timeframe": multi_tf_payload,
            "fundamental": fund,
            "news_sentiment": news_sent,
            "heuristic_score": heuristic_score,
            "heuristic_reasons": reasons,
            "heuristic_rr": rr,
            "support_levels": supports,
            "resistance_levels": resistances,
            "trend_strength": trend_strength
        }

        # Expected value calculation with a simple heuristic pre-AI (for AI and telegram)
        # If AI not available, use heuristic EV as fallback
        # estimate prob_up from heuristic_score & news
        prob_up_est = min(0.9, max(0.05, 0.5 + (heuristic_score - 50)/200 + (news_sent.get("score",50)-50)/200))
        reward = rr["target"] - rr["entry"]
        risk = rr["entry"] - rr["stop"]
        ev = prob_up_est * reward - (1-prob_up_est) * risk if (reward is not None and risk is not None) else 0.0
        payload["heuristic_ev"] = {"prob_up_est":prob_up_est, "ev":ev}

        # Call AI wrapper
        try:
            ai_result, ai_err = cached_ai_call(ai_provider, ai_model, payload, api_keys)
        except Exception as e:
            append_log(f"[{ticker} {tf}] AI wrapper exception: {e}\n{traceback.format_exc()}")
            ai_result, ai_err = None, str(e)

        if ai_err:
            st.error(f"AI error: {ai_err}")
        else:
            st.subheader("AI analysis")
            try:
                render_ai_result(ai_result)
            except Exception:
                st.write("AI output (raw):", str(ai_result)[:1000])

        # Telegram alerts
        try:
            if enable_telegram and telegram_bot_token and telegram_chat_id:
                should_alert = False
                # check AI result if available
                if ai_result:
                    reco = (ai_result.get("rekomendasi","") or "").upper()
                    prob_map = ai_result.get("probabilitas",{}) or {}
                    prob_up = prob_map.get("naik_5d") or prob_map.get("naik_3d")
                    rr_ratio = (ai_result.get("risk_reward") or {}).get("rr_ratio") or rr.get("rr")
                    if reco == "BUY":
                        should_alert = True
                    elif prob_up is not None and isinstance(prob_up, (int,float)) and prob_up >= 0.6:
                        should_alert = True
                    elif rr_ratio and rr_ratio >= 2.0 and prob_up is not None and prob_up >= 0.55:
                        should_alert = True
                else:
                    # fallback to heuristic EV
                    if ev is not None and ev > 0 and prob_up_est >= 0.6 and rr.get("rr",0) >= 2.0:
                        should_alert = True
                if should_alert:
                    # prepare message
                    msg = f"<b>{ticker} {tf}</b>\nReco: {ai_result.get('rekomendasi') if ai_result else 'HEURISTIC'}\nProb up est: {round(prob_up_est*100,1)}%\nPrice: {last_price}\nEntry: {rr['entry']}\nStop: {rr['stop']}\nTarget: {rr['target']}\nEV: {ev:.4f}\nTime: {datetime.utcnow().isoformat()} UTC"
                    sent = send_telegram_alert(telegram_bot_token, telegram_chat_id, msg)
                    if sent:
                        st.success("Telegram alert sent")
                        append_log(f"Telegram alert sent for {ticker} {tf}")
                    else:
                        st.error("Failed to send Telegram alert")
                        append_log(f"Telegram send failed for {ticker} {tf}")
        except Exception as e:
            append_log(f"Telegram logic exception: {e}")

        # Backtest ML (EV regression)
        if enable_backtest:
            st.subheader("Backtest & ML (LightGBM EV Regression)")
            try:
                df_feat = create_features_for_ml(df)
                df_labeled = build_label_ev(df_feat, horizon=int(backtest_horizon_days))
                model, metrics, feat_imp, preds, y_test = train_lgb_model_ev(df_labeled, train_ratio=backtest_train_ratio)
                st.write("Backtest metrics (EV Regression):", metrics)
                st.write("Top feature importances:")
                st.table(pd.DataFrame(feat_imp[:10], columns=["feature","importance"]).set_index("feature"))
                latest_features = df_labeled.iloc[-1:]
                feature_cols = [c for c in df_labeled.columns if c not in ["date","future_ret","Open","High","Low","Close","Volume","Adj Close","index"]]
                latest_pred = model.predict(latest_features[feature_cols])[0]
                st.metric("Model predicted future_ret (latest)", f"{latest_pred*100:.2f}%")
            except Exception as e:
                append_log(f"[{ticker} {tf}] Backtest error: {e}\n{traceback.format_exc()}")
                st.error(f"Backtest error: {e}")

        per_ticker_result["timeframes"][tf] = {
            "heuristic_score": heuristic_score,
            "ai_result": ai_result,
            "ai_err": ai_err,
            "rr": rr,
            "heuristic_ev": {"prob_up_est":prob_up_est, "ev":ev}
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
            "ai_err": info.get("ai_err"),
            "heuristic_ev": info.get("heuristic_ev")
        })
if flat:
    st.dataframe(pd.DataFrame(flat).set_index(["ticker","timeframe"]))

# ---------------- Logs in sidebar ----------------
with st.sidebar.expander("Application logs (recent)"):
    logs = st.session_state.get("logs", [])
    if logs:
        st.write("\n".join(logs[-120:]))
    else:
        st.write("No logs yet.")

st.markdown("**Note**: Tool ini untuk riset. Validate with paper trading/backtest before live trading. Do not commit API keys to public repo.")
