import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import platform
import numpy as np
from datetime import datetime, timedelta
import os

# 📌 한글 폰트 설정
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif platform.system() == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'
else:
    plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

# 📌 업비트 일봉 데이터 수집
def get_ohlcv_extended(market="KRW-BTC", total_days=365):
    url = "https://api.upbit.com/v1/candles/days"
    headers = {"Accept": "application/json"}
    all_data, to, remaining = [], None, total_days

    while remaining > 0:
        count = min(200, remaining)
        params = {"market": market, "count": count}
        if to:
            params["to"] = to
        res = requests.get(url, headers=headers, params=params)
        data = res.json()
        all_data.extend(data)
        last_date = data[-1]['candle_date_time_kst']
        to = (datetime.strptime(last_date, "%Y-%m-%dT%H:%M:%S") - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%S")
        remaining -= len(data)

    df = pd.DataFrame(all_data)
    df['날짜'] = pd.to_datetime(df['candle_date_time_kst'])
    df = df.sort_values(by='날짜')
    df = df[['날짜', 'opening_price', 'high_price', 'low_price', 'trade_price', 'candle_acc_trade_volume']]
    df.columns = ['날짜', '시가', '고가', '저가', '종가', '거래량']
    return df

# 📌 업비트 10분봉 데이터 수집 (ENA 단타)
def get_ohlcv_10min(market="KRW-ENA", count=200):
    url = f"https://api.upbit.com/v1/candles/minutes/10?market={market}&count={count}"
    res = requests.get(url).json()
    df = pd.DataFrame(res)
    df['날짜'] = pd.to_datetime(df['candle_date_time_kst'])
    df = df.sort_values(by='날짜')
    df = df[['날짜', 'opening_price', 'high_price', 'low_price', 'trade_price', 'candle_acc_trade_volume']]
    df.columns = ['날짜', '시가', '고가', '저가', '종가', '거래량']
    return df

# 📌 기술적 지표 계산 (일봉)
def compute_indicators(df):
    df['MA20'] = df['종가'].rolling(window=20).mean()
    df['MA60'] = df['종가'].rolling(window=60).mean()
    df['STD'] = df['종가'].rolling(window=20).std()
    df['Upper'] = df['MA20'] + 2 * df['STD']
    df['Lower'] = df['MA20'] - 2 * df['STD']
    delta = df['종가'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['EMA12'] = df['종가'].ewm(span=12).mean()
    df['EMA26'] = df['종가'].ewm(span=26).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal']
    df['VOL_MA20'] = df['거래량'].rolling(window=20).mean()
    df['VOL_RISE'] = df['거래량'] > df['VOL_MA20']
    df['TR'] = df[['고가', '저가', '종가']].apply(lambda x: max(x[0] - x[1], abs(x[0] - x[2]), abs(x[1] - x[2])), axis=1)
    df['ATR'] = df['TR'].rolling(window=14).mean()
    df['OBV'] = (np.sign(df['종가'].diff()) * df['거래량']).fillna(0).cumsum()
    return df

# 📌 기술적 지표 계산 (단타용)
def compute_intraday_indicators(df):
    df['EMA5'] = df['종가'].ewm(span=5).mean()
    df['EMA20'] = df['종가'].ewm(span=20).mean()
    df['STD'] = df['종가'].rolling(20).std()
    df['Upper'] = df['EMA20'] + 2 * df['STD']
    df['Lower'] = df['EMA20'] - 2 * df['STD']
    delta = df['종가'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['EMA12'] = df['종가'].ewm(span=12).mean()
    df['EMA26'] = df['종가'].ewm(span=26).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal']
    return df

# 📌 전략 제안 (일봉용)
def strategy_suggestion(df):
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    signals, score = [], 0

    if latest['RSI'] < 30:
        signals.append("📉 RSI < 30 → 과매도: 매수 유력")
        score += 1
    elif latest['RSI'] > 70:
        signals.append("📈 RSI > 70 → 과매수: 매도 유력")
    else:
        signals.append(f"RSI {latest['RSI']:.2f}: 중립 구간")

    if latest['종가'] > latest['MA20'] and latest['MA20'] > latest['MA60']:
        signals.append("🔼 이평선 정배열: 상승 추세")
        score += 1
    elif latest['종가'] < latest['MA20'] and latest['MA20'] < latest['MA60']:
        signals.append("🔽 이평선 역배열: 하락 추세")

    if latest['MACD'] > latest['Signal']:
        signals.append("🟢 MACD > Signal → 매수 모멘텀")
        score += 1
    elif latest['MACD'] < latest['Signal']:
        signals.append("🔴 MACD < Signal → 매도 모멘텀")

    if latest['MACD_Hist'] > 0 and prev['MACD_Hist'] < 0:
        signals.append("🟢 MACD Histogram 양전환 → 매수 시그널")
        score += 1
    elif latest['MACD_Hist'] < 0 and prev['MACD_Hist'] > 0:
        signals.append("🔴 MACD Histogram 음전환 → 매도 시그널")

    if latest['종가'] < latest['Lower']:
        signals.append("📉 밴드 하단 이탈 → 기술적 반등")
        score += 1
    elif latest['종가'] > latest['Upper']:
        signals.append("📈 밴드 상단 돌파 → 과열")

    if latest['VOL_RISE']: score += 1
    if latest['OBV'] > prev['OBV']: score += 1

    if score >= 4:
        signals.append("✅ 종합 판단: 강한 매수 신호")
    elif score <= 2:
        signals.append("⛔ 종합 판단: 매도 또는 관망")
    else:
        signals.append("⏳ 종합 판단: 중립 또는 약한 매수")

    return signals

# 📌 전략 제안 (ENA 단타용)
def strategy_ena(df):
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    signals, score = [], 0

    if latest['RSI'] < 30:
        signals.append("📉 RSI < 30: 과매도")
        score += 1
    elif latest['RSI'] > 70:
        signals.append("📈 RSI > 70: 과매수")

    if latest['EMA5'] > latest['EMA20']:
        signals.append("🟢 EMA5 > EMA20: 상승세")
        score += 1
    else:
        signals.append("🔴 EMA5 < EMA20: 하락세")

    if latest['MACD'] > latest['Signal']:
        signals.append("🟢 MACD > Signal → 매수 모멘텀")
        score += 1
    else:
        signals.append("🔴 MACD < Signal → 매도 모멘텀")

    if latest['종가'] < latest['Lower']:
        signals.append("📉 밴드 하단 이탈 → 반등 가능성")
        score += 1
    elif latest['종가'] > latest['Upper']:
        signals.append("📈 밴드 상단 돌파 → 과열 신호")

    if score >= 3:
        signals.append("✅ 종합 판단: 매수 시도 가능")
    elif score <= 1:
        signals.append("⛔ 종합 판단: 관망 또는 매도")
    else:
        signals.append("⏳ 종합 판단: 중립 또는 약한 매수")

    return signals

# 📌 Streamlit 앱
def main():
    st.set_page_config(page_title="암호화폐 전략 분석기", layout="wide")
    st.title("📊 암호화폐 전략 분석 (BTC, ETH, XRP, ENA)")

    period_map = {"100일": 100, "180일 (6개월)": 180, "365일 (1년)": 365}
    selected_period = st.radio("분석 기간 선택:", list(period_map.keys()), horizontal=True)
    days = period_map[selected_period]

    coin_dict = {
        "비트코인 (BTC)": "KRW-BTC",
        "이더리움 (ETH)": "KRW-ETH",
        "리플 (XRP)": "KRW-XRP",
        "에테나 (ENA 단타)": "KRW-ENA"
    }

    selected_coin = st.selectbox("분석할 코인 선택:", list(coin_dict.keys()))
    market_code = coin_dict[selected_coin]

    if market_code == "KRW-ENA":
        df = get_ohlcv_10min()
        df = compute_intraday_indicators(df)
        st.subheader("📈 ENA 10분봉 차트")
        fig, ax = plt.subplots()
        ax.plot(df['날짜'], df['종가'], label='종가', color='black')
        ax.plot(df['날짜'], df['EMA5'], label='EMA5', color='green')
        ax.plot(df['날짜'], df['EMA20'], label='EMA20', color='orange')
        ax.fill_between(df['날짜'], df['Upper'], df['Lower'], alpha=0.2, label='볼린저 밴드')
        ax.legend()
        st.pyplot(fig)

        st.subheader("📉 RSI / MACD")
        fig2, ax2 = plt.subplots(2, 1, sharex=True)
        ax2[0].plot(df['날짜'], df['RSI'], label='RSI', color='purple')
        ax2[0].axhline(70, linestyle='--', color='red')
        ax2[0].axhline(30, linestyle='--', color='green')
        ax2[0].legend()
        ax2[1].plot(df['날짜'], df['MACD'], label='MACD', color='blue')
        ax2[1].plot(df['날짜'], df['Signal'], label='Signal', color='red')
        ax2[1].legend()
        st.pyplot(fig2)

        st.subheader("💡 단타 전략 제안")
        for s in strategy_ena(df):
            st.write("- " + s)

    else:
        df = get_ohlcv_extended(market_code, total_days=days)
        df = compute_indicators(df)

        st.subheader(f"📈 {selected_coin} 일봉 차트")
        fig, ax = plt.subplots()
        ax.plot(df['날짜'], df['종가'], label='종가', color='blue')
        ax.plot(df['날짜'], df['MA20'], label='MA20', color='orange')
        ax.plot(df['날짜'], df['MA60'], label='MA60', color='green')
        ax.fill_between(df['날짜'], df['Upper'], df['Lower'], color='gray', alpha=0.2)
        ax.legend()
        st.pyplot(fig)

        st.subheader("📉 RSI / MACD / 거래량")
        fig2, ax2 = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        ax2[0].plot(df['날짜'], df['RSI'], color='purple')
        ax2[0].axhline(70, color='red', linestyle='--')
        ax2[0].axhline(30, color='green', linestyle='--')
        ax2[1].plot(df['날짜'], df['MACD'], color='blue')
        ax2[1].plot(df['날짜'], df['Signal'], color='red')
        ax2[1].axhline(0, color='gray', linestyle='--')
        ax2[2].bar(df['날짜'], df['거래량'], color='gray')
        ax2[2].plot(df['날짜'], df['VOL_MA20'], color='orange')
        st.pyplot(fig2)

        st.subheader("💡 전략 제안")
        for s in strategy_suggestion(df):
            st.write("- " + s)

if __name__ == "__main__":
    main()
