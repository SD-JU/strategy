import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import platform
import numpy as np
from datetime import datetime, timedelta
import os

# 한글 폰트 설정
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif platform.system() == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'
else:
    plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

# 📌 업비트 OHLCV 데이터 수집
def get_ohlcv_extended(market="KRW-BTC", total_days=365):
    url = "https://api.upbit.com/v1/candles/days"
    headers = {"Accept": "application/json"}
    all_data = []
    to = None
    remaining = total_days

    while remaining > 0:
        count = min(200, remaining)
        params = {"market": market, "count": count}
        if to:
            params["to"] = to
        response = requests.get(url, headers=headers, params=params)
        data = response.json()
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

# 📌 기술적 지표 계산
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
    df['EMA12'] = df['종가'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['종가'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal']
    df['VOL_MA20'] = df['거래량'].rolling(window=20).mean()
    df['VOL_RISE'] = df['거래량'] > df['VOL_MA20']
    df['H-L'] = df['고가'] - df['저가']
    df['H-PC'] = abs(df['고가'] - df['종가'].shift(1))
    df['L-PC'] = abs(df['저가'] - df['종가'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=14).mean()
    df['OBV'] = (np.sign(df['종가'].diff()) * df['거래량']).fillna(0).cumsum()
    return df

# 📌 전략 제안
def strategy_suggestion(df):
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    signals = []

    if latest['RSI'] < 30:
        signals.append("📉 RSI < 30 → 과매도: 매수 유력")
    elif latest['RSI'] > 70:
        signals.append("📈 RSI > 70 → 과매수: 매도 유력")
    else:
        signals.append(f"RSI {latest['RSI']:.2f}: 중립 구간")

    if latest['종가'] > latest['MA20'] and latest['MA20'] > latest['MA60']:
        signals.append("🔼 이평선 정배열: 상승 추세")
    elif latest['종가'] < latest['MA20'] and latest['MA20'] < latest['MA60']:
        signals.append("🔽 이평선 역배열: 하락 추세")
    else:
        signals.append("이평선 혼조: 방향성 불분명")

    if latest['MACD'] > latest['Signal']:
        signals.append("🟢 MACD > Signal → 매수 모멘텀")
    elif latest['MACD'] < latest['Signal']:
        signals.append("🔴 MACD < Signal → 매도 모멘텀")

    if latest['MACD_Hist'] > 0 and prev['MACD_Hist'] < 0:
        signals.append("🟢 MACD Histogram 양전환 → 매수 시그널 발생")
    elif latest['MACD_Hist'] < 0 and prev['MACD_Hist'] > 0:
        signals.append("🔴 MACD Histogram 음전환 → 매도 시그널 발생")

    if latest['종가'] < latest['Lower']:
        signals.append("📉 볼린저 밴드 하단 이탈 → 기술적 반등 가능성")
    elif latest['종가'] > latest['Upper']:
        signals.append("📈 볼린저 밴드 상단 돌파 → 과열 신호")

    if latest['RSI'] < 30 and latest['종가'] < latest['Lower']:
        signals.append("📌 과매도 + 밴드 하단: 반등 확률 ↑")

    if latest['VOL_RISE']:
        signals.append("💹 거래량 평균 상회 → 관심 집중")
    else:
        signals.append("🔕 거래량 평균 이하 → 관망")

    if latest['OBV'] > prev['OBV']:
        signals.append("📈 OBV 상승 → 매수세 유입")
    else:
        signals.append("📉 OBV 하락 → 매도세 우위")

    if latest['ATR'] > df['ATR'].mean():
        signals.append("📊 ATR 상승 → 높은 변동성")
    else:
        signals.append("📉 ATR 하락 → 낮은 변동성")

    score = 0
    if latest['RSI'] < 30: score += 1
    if latest['종가'] < latest['Lower']: score += 1
    if latest['MACD'] > latest['Signal']: score += 1
    if latest['MACD_Hist'] > 0 and prev['MACD_Hist'] < 0: score += 1
    if latest['OBV'] > prev['OBV']: score += 1
    if latest['VOL_RISE']: score += 1

    if score >= 4:
        signals.append("✅ 종합 판단: 강한 매수 신호")
    elif score <= 2:
        signals.append("⛔ 종합 판단: 매도 또는 관망")
    else:
        signals.append("⏳ 종합 판단: 중립 또는 약한 매수")

    return signals

# 📌 Streamlit 앱 시작
def main():
    st.set_page_config(page_title="종합 암호화폐 전략 분석기", layout="wide")
    st.title("📊 BTC / ETH / XRP 전략 분석 (기술적 + 심리적 지표 기반)")

    # 분석 기간 선택
    period_map = {"100일": 100, "180일 (6개월)": 180, "365일 (1년)": 365}
    selected_period_str = st.radio("분석 기간을 선택하세요:", list(period_map.keys()), horizontal=True)
    selected_period = period_map[selected_period_str]

    # 코인 선택
    coin_dict = {
        "비트코인 (BTC)": "KRW-BTC",
        "이더리움 (ETH)": "KRW-ETH",
        "리플 (XRP)": "KRW-XRP"
    }
    selected_coin = st.selectbox("분석할 코인을 선택하세요:", list(coin_dict.keys()))
    market_code = coin_dict[selected_coin]

    # 데이터 수집 및 분석
    df = get_ohlcv_extended(market_code, total_days=selected_period)
    df = compute_indicators(df)

    # 시세 차트
    st.subheader(f"📈 {selected_coin} 가격 및 기술적 지표")
    fig, ax = plt.subplots()
    ax.plot(df['날짜'], df['종가'], label='Close', color='blue')
    ax.plot(df['날짜'], df['MA20'], label='MA20', color='orange')
    ax.plot(df['날짜'], df['MA60'], label='MA60', color='green')
    ax.fill_between(df['날짜'], df['Upper'], df['Lower'], color='gray', alpha=0.2, label='Bollinger Bands')
    ax.legend()
    st.pyplot(fig)

    # 보조 지표 차트
    st.subheader("📉 RSI / MACD / 거래량")
    fig2, ax2 = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    ax2[0].plot(df['날짜'], df['RSI'], label='RSI', color='purple')
    ax2[0].axhline(70, color='red', linestyle='--')
    ax2[0].axhline(30, color='green', linestyle='--')
    ax2[0].legend()
    ax2[1].plot(df['날짜'], df['MACD'], label='MACD', color='blue')
    ax2[1].plot(df['날짜'], df['Signal'], label='Signal', color='red')
    ax2[1].axhline(0, color='gray', linestyle='--')
    ax2[1].legend()
    ax2[2].bar(df['날짜'], df['거래량'], label='Volume', color='gray')
    ax2[2].plot(df['날짜'], df['VOL_MA20'], label='Volume Avg', color='orange')
    ax2[2].legend()
    st.pyplot(fig2)

    # 전략 제안
    st.subheader("💡 전략 제안")
    suggestions = strategy_suggestion(df)
    for s in suggestions:
        st.write("- " + s)

    # 📘 해설서 다운로드 버튼
    st.markdown("---")
    st.subheader("📘 기술적 지표 해설서 보기")
    if os.path.exists("crypto_strategy_guide.html"):
        with open("crypto_strategy_guide.html", "rb") as f:
            st.download_button(
                label="📥 해설서 다운로드 (.html)",
                data=f,
                file_name="crypto_strategy_guide.html",
                mime="text/html"
            )
    else:
        st.warning("guide 파일이 현재 디렉토리에 없습니다.")

if __name__ == "__main__":
    main()
