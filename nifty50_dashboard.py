"""
NIFTY 50 TWO-DECADE ANALYTICS DASHBOARD
=======================================
Run:
    pip install -r requirements.txt
    streamlit run nifty50_dashboard.py

Upload:
    Combined_P3(1).xlsx  (or any CSV with the same columns)
    stock_metadata.csv   (optional)

This is an educational research tool. Historical returns are not guarantees
of future returns and the portfolio output is a historical screening result,
not personalised investment advice.
"""

import io
import os
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Nifty 50 Analytics", layout="wide")

REQUIRED = ["File Name","Date","Prev Close","Open","High","Low","Close","Volume","%Deliverble"]

@st.cache_data(show_spinner=False)
def load_file(data, name):
    if name.lower().endswith(".csv"):
        df = pd.read_csv(io.BytesIO(data), low_memory=False)
    else:
        df = pd.read_excel(io.BytesIO(data))
    df.columns = [str(c).strip() for c in df.columns]
    # Remove accidental repeated header rows.
    if "Date" in df:
        df = df[df["Date"].astype(str).str.lower() != "date"].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    nums = ["Prev Close","Open","High","Low","Last","Close","VWAP",
            "Volume","Turnover","Trades","Deliverable Volume","%Deliverble"]
    for c in nums:
        if c in df:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",","",regex=False),
                                  errors="coerce")
    df["File Name"] = df["File Name"].astype(str).str.strip().str.upper()
    df = df[df["Date"].notna() & df["Close"].notna()].copy()
    return df.sort_values(["File Name","Date"])

def stock_metrics(data):
    rows=[]
    for ticker,g in data.groupby("File Name"):
        if ticker == "NIFTY50_ALL": continue
        g=g.sort_values("Date")
        r=g["Close"].pct_change().dropna()
        if len(r)<500: continue
        years=max((g.Date.iloc[-1]-g.Date.iloc[0]).days/365.25,1e-9)
        wealth=(1+r).cumprod()
        vol=r.std()*np.sqrt(252)
        cagr=(g.Close.iloc[-1]/g.Close.iloc[0])**(1/years)-1
        sharpe=(r.mean()*252)/vol if vol else np.nan
        downside=r[r<0].std()*np.sqrt(252)
        sortino=(r.mean()*252)/downside if downside else np.nan
        mdd=(wealth/wealth.cummax()-1).min()
        rows.append([ticker,cagr,vol,sharpe,sortino,mdd,(r>0).mean(),len(r)])
    return pd.DataFrame(rows,columns=["Ticker","CAGR","Volatility","Sharpe",
        "Sortino","Max Drawdown","Win Rate","Observations"])

def feature_engineer(data):
    allx=[]
    for ticker,g in data.groupby("File Name"):
        if ticker=="NIFTY50_ALL": continue
        g=g.sort_values("Date").copy()
        g["ret1"]=g.Close.pct_change()
        g["ret5"]=g.Close.pct_change(5)
        g["ret20"]=g.Close.pct_change(20)
        g["vol20"]=g.ret1.rolling(20).std()
        g["vol60"]=g.ret1.rolling(60).std()
        g["sma20"]=g.Close/g.Close.rolling(20).mean()-1
        g["sma50"]=g.Close/g.Close.rolling(50).mean()-1
        g["range"]=(g.High-g.Low)/g.Close
        g["gap"]=g.Open/g["Prev Close"]-1
        g["volchg"]=g.Volume.pct_change()
        g["delivery"]=g["%Deliverble"]
        gain=g.ret1.clip(lower=0).rolling(14).mean()
        loss=(-g.ret1.clip(upper=0)).rolling(14).mean()
        g["rsi"]=100-100/(1+gain/loss.replace(0,np.nan))
        g["future5"]=g.Close.shift(-5)/g.Close-1
        allx.append(g[["Date","File Name","ret1","ret5","ret20","vol20","vol60",
                        "sma20","sma50","range","gap","volchg","delivery","rsi","future5"]])
    x=pd.concat(allx,ignore_index=True).replace([np.inf,-np.inf],np.nan).dropna()
    threshold=x.future5.abs().median()
    x["high_volatility"]=(x.future5.abs()>threshold).astype(int)
    return x,threshold

def choose_candidates(m, data, n=6):
    ret=data.pivot(index="Date",columns="File Name",values="Close").pct_change()
    # Historical screening score.
    def norm(s, inverse=False):
        lo,hi=s.quantile(.05),s.quantile(.95)
        z=((s-lo)/(hi-lo)).clip(0,1)
        return 1-z if inverse else z
    m=m.copy()
    m["Screening Score"]=(.25*norm(m.CAGR)+.25*norm(m.Sharpe)+
                          .15*norm(m.Sortino)+.15*norm(m.Volatility,True)+
                          .15*norm(-m["Max Drawdown"],True)+.05*norm(m["Win Rate"]))
    chosen=[]
    for t in m.sort_values("Screening Score",ascending=False).Ticker:
        if t not in ret: continue
        if not chosen or (ret[chosen+[t]].corr()[t].drop(t).abs()<.75).all():
            chosen.append(t)
        if len(chosen)==n: break
    return chosen,m,ret

st.title("Nifty 50 — Two-Decade Historical Analytics")
st.caption("Risk analysis • stock screening • portfolio research • machine learning")

up=st.sidebar.file_uploader("Combined market dataset",type=["xlsx","xls","csv"])
meta=st.sidebar.file_uploader("stock_metadata.csv (optional)",type=["csv"])

meta_df = None
if meta is not None:
    try:
        meta_df = pd.read_csv(meta)
    except Exception:
        pass
elif os.path.exists("stock_metadata.csv"):
    try:
        meta_df = pd.read_csv("stock_metadata.csv")
    except Exception:
        pass

if up is not None:
    df=load_file(up.getvalue(),up.name)
elif os.path.exists("Combined_P3.xlsx"):
    st.sidebar.success("Using local file: `Combined_P3.xlsx`")
    with open("Combined_P3.xlsx", "rb") as f:
        df=load_file(f.read(), "Combined_P3.xlsx")
elif os.path.exists("Combined_P3(1).xlsx"):
    st.sidebar.success("Using local file: `Combined_P3(1).xlsx`")
    with open("Combined_P3(1).xlsx", "rb") as f:
        df=load_file(f.read(), "Combined_P3(1).xlsx")
else:
    st.info("Upload the supplied Combined_P3.xlsx file (or place it in the project root) to begin.")
    st.stop()

stocks=df[df["File Name"]!="NIFTY50_ALL"].copy()
last_decade=stocks[stocks.Date>=pd.Timestamp("2011-01-01")].copy()
m=stock_metrics(last_decade)
selected,m,returns=choose_candidates(m,last_decade)

overview,stocktab,port,mltab=st.tabs(["Overview","Stock Analysis","Portfolio","ML"])

with overview:
    a,b,c,d=st.columns(4)
    a.metric("Valid rows",f"{len(df):,}")
    b.metric("Stock groups",f"{stocks['File Name'].nunique():,}")
    c.metric("Data start",str(df.Date.min().date()))
    d.metric("Data end",str(df.Date.max().date()))
    st.subheader("Historical stock metrics — last decade")
    show=m.sort_values("Screening Score",ascending=False).copy()
    st.dataframe(show.style.format({"CAGR":"{:.2%}","Volatility":"{:.2%}",
        "Sharpe":"{:.2f}","Sortino":"{:.2f}","Max Drawdown":"{:.2%}",
        "Win Rate":"{:.2%}","Screening Score":"{:.3f}"}),use_container_width=True)

with stocktab:
    ticker=st.selectbox("Stock",sorted(last_decade["File Name"].unique()))
    if meta_df is not None and "Symbol" in meta_df.columns:
        meta_row = meta_df[meta_df["Symbol"].astype(str).str.strip().str.upper() == ticker]
        if not meta_row.empty:
            c_name = meta_row.iloc[0].get("Company Name", ticker)
            c_ind = meta_row.iloc[0].get("Industry", "N/A")
            st.markdown(f"**Company:** {c_name} &nbsp;|&nbsp; **Industry:** `{c_ind}`")
    g=last_decade[last_decade["File Name"]==ticker].sort_values("Date")
    r=g.Close.pct_change().dropna()
    a,b,c=st.columns(3)
    a.metric("CAGR",f"{float(m.loc[m.Ticker==ticker,'CAGR'].iloc[0]):.2%}")
    b.metric("Annual volatility",f"{r.std()*np.sqrt(252):.2%}")
    c.metric("Max drawdown",f"{float(m.loc[m.Ticker==ticker,'Max Drawdown'].iloc[0]):.2%}")
    fig=plt.figure(figsize=(11,4)); plt.plot(g.Date,g.Close)
    plt.title(f"{ticker} closing price"); plt.xlabel("Date"); plt.ylabel("Price")
    st.pyplot(fig); plt.close(fig)
    x,_=feature_engineer(g)
    st.subheader("Engineered features")
    st.dataframe(x.tail(25),use_container_width=True)

with port:
    st.subheader("Historical candidate portfolio")
    st.write("Candidates:",", ".join(selected))
    st.dataframe(pd.DataFrame({"Ticker":selected,"Equal Weight":[1/len(selected)]*len(selected)}),
                 use_container_width=True)
    p=returns[selected].fillna(0).mean(axis=1)
    wealth=(1+p).cumprod()
    fig=plt.figure(figsize=(11,4)); plt.plot(wealth.index,wealth.values)
    plt.title("Illustrative equal-weight historical backtest")
    plt.xlabel("Date"); plt.ylabel("Growth of ₹1")
    st.pyplot(fig); plt.close(fig)
    years=max((wealth.index[-1]-wealth.index[0]).days/365.25,1e-9)
    pv=wealth.iloc[-1]**(1/years)-1
    vv=p.std()*np.sqrt(252)
    sh=p.mean()*252/vv
    dd=(wealth/wealth.cummax()-1).min()
    a,b,c=st.columns(3)
    a.metric("Backtest CAGR",f"{pv:.2%}")
    b.metric("Volatility",f"{vv:.2%}")
    c.metric("Sharpe",f"{sh:.2f}")
    st.write(f"Maximum drawdown: **{dd:.2%}**")
    st.warning("Illustrative historical backtest only. It does not model taxes, slippage, transaction costs, dividends or survivorship/index-membership effects.")

with mltab:
    x,threshold=feature_engineer(last_decade)
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score,accuracy_score
    feats=["ret1","ret5","ret20","vol20","vol60","sma20","sma50","range",
           "gap","volchg","delivery","rsi"]
    sample=x.sample(min(60000,len(x)),random_state=42)
    Xtr,Xte,ytr,yte=train_test_split(sample[feats],sample.high_volatility,
        test_size=.20,random_state=42,stratify=sample.high_volatility)
    model=RandomForestClassifier(n_estimators=150,max_depth=10,min_samples_leaf=20,
        class_weight="balanced",random_state=42,n_jobs=-1)
    model.fit(Xtr,ytr)
    prob=model.predict_proba(Xte)[:,1]
    auc=roc_auc_score(yte,prob)
    acc=accuracy_score(yte,(prob>=.5).astype(int))
    a,b=st.columns(2); a.metric("ROC-AUC",f"{auc:.3f}"); b.metric("Accuracy",f"{acc:.3f}")
    imp=pd.Series(model.feature_importances_,index=feats).sort_values()
    fig=plt.figure(figsize=(9,5)); plt.barh(imp.index,imp.values)
    plt.title("ML feature importance"); plt.xlabel("Importance")
    st.pyplot(fig); plt.close(fig)
    st.write(f"Target definition: absolute future 5-day return above the sample median ({threshold:.2%}).")
