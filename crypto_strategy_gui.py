# crypto_strategy_app.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import platform
import os
import matplotlib.font_manager as fm

# ✅ 1. NanumGothic 폰트 설정 (Streamlit Cloud용)
font_path = "/tmp/NanumGothic.ttf"
if not os.path.exists(font_path):
    import urllib.request
    urllib.request.urlretrieve(
        "https://github.com/naver/nanumfont/blob/master/ttf/NanumGothic.ttf?raw=true",
        font_path
    )
    fm.fontManager.addfont(font_path)

plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()
plt.rcParams['axes.unicode_minus'] = False

# ✅ 2. 업비트 OHLCV 데이터 수집
def get_ohlcv(market="KRW-BTC", count=100):
    url = "https://api.upbit.com/v1/candles/days"
    headers = {"Accept": "application/json"}
    params = {"market": market, "count": count}
    response = requests.get(url, headers=headers, params=params)
    data = response.json()
    df = pd.DataFrame(data)
    df['날짜'] = pd.to_datetime(df['candle_date_time_kst'])
    df = df.sort_values(by='날짜')
    df = df[['날짜', 'opening_price', 'high_price', 'low_price', 'trade_price', 'candle_acc_trade_volume']]
    df.columns = ['날짜', '시가', '고가', '저가', '종가', '거래량']
    return df

# ✅ 3. 기술적 지표 계산

def compute_rsi(df, period=14):
    delta = df['종가'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_indicators(df):
    df['RSI'] = compute_rsi(df)
    df['MA20'] = df['종가'].rolling(window=20).mean()
    df['MA60'] = df['종가'].rolling(window=60).mean()
    df['STD'] = df['종가'].rolling(window=20).std()
    df['Upper'] = df['MA20'] + 2 * df['STD']
    df['Lower'] = df['MA20'] - 2 * df['STD']
    df['EMA12'] = df['종가'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['종가'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['ATR'] = df['고가'] - df['저가']
    df['VOL_MA20'] = df['거래량'].rolling(window=20).mean()
    return df

# ✅ 4. 전략 해석

def strategy_suggestion(df):
    latest = df.iloc[-1]
    signals = []
    score = 0

    # RSI
    if latest['RSI'] < 30:
        signals.append("📉 RSI < 30 → 과매도: 매수 유력"); score += 1
    elif latest['RSI'] > 70:
        signals.append("📈 RSI > 70 → 과매수: 매도 유력")
    else:
        signals.append(f"RSI {latest['RSI']:.2f}: 중립 구간")

    # 이평선
    if latest['종가'] > latest['MA20'] > latest['MA60']:
        signals.append("🔼 이평선 정배열: 상승 추세"); score += 1
    elif latest['종가'] < latest['MA20'] < latest['MA60']:
        signals.append("🔽 이평선 역배열: 하락 추세")
    else:
        signals.append("이평선 혼조: 방향성 불분명")

    # MACD
    if latest['MACD'] > latest['Signal']:
        signals.append("🟢 MACD > Signal → 매수 모멘텀"); score += 1
    elif latest['MACD'] < latest['Signal']:
        signals.append("🔴 MACD < Signal → 매도 모멘텀")
    else:
        signals.append("MACD 중립 상태")

    # 볼린저밴드
    if latest['종가'] < latest['Lower']:
        signals.append("📉 볼린저 밴드 하단 이탈 → 기술적 반등 가능성"); score += 1
    elif latest['종가'] > latest['Upper']:
        signals.append("📈 볼린저 밴드 상단 돌파 → 과열 신호")
    else:
        signals.append("볼린저 밴드 내 안정 구간")

    # 거래량 분석
    if latest['거래량'] > latest['VOL_MA20'] * 1.2:
        signals.append("📊 거래량 급증 → 매수세 유입 가능성"); score += 1

    # 종합 판단
    if score >= 4:
        signals.append("📌 종합 판단: ✅ 강한 매수 시점")
    elif score >= 2:
        signals.append("📌 종합 판단: ⏳ 관망 또는 약한 매수")
    else:
        signals.append("📌 종합 판단: ⛔ 매도 또는 관망 추천")

    return signals

# ✅ 5. Streamlit 앱 실행

def main():
    st.set_page_config(page_title="코인 전략 분석기", layout="wide")
    st.title("📊 BTC/ETH/XRP 종합 전략 분석 (RSI, MACD, MA, 볼린저밴드, 거래량)")

    coin_dict = {
        "비트코인 (BTC)": "KRW-BTC",
        "이더리움 (ETH)": "KRW-ETH",
        "리플 (XRP)": "KRW-XRP"
    }
    selected_coin = st.selectbox("분석할 코인을 선택하세요:", list(coin_dict.keys()))
    market_code = coin_dict[selected_coin]

    df = get_ohlcv(market_code)
    df = compute_indicators(df)

    # 📈 차트 출력
    st.subheader(f"📈 {selected_coin} 가격 및 기술적 지표")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df['날짜'], df['종가'], label='종가', color='blue')
    ax.plot(df['날짜'], df['MA20'], label='MA20', color='orange')
    ax.plot(df['날짜'], df['MA60'], label='MA60', color='green')
    ax.fill_between(df['날짜'], df['Upper'], df['Lower'], color='gray', alpha=0.3, label='볼린저 밴드')
    ax.legend()
    st.pyplot(fig)

    # 📉 RSI & MACD
    fig2, ax2 = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax2[0].plot(df['날짜'], df['RSI'], label='RSI', color='purple')
    ax2[0].axhline(70, color='red', linestyle='--')
    ax2[0].axhline(30, color='green', linestyle='--')
    ax2[0].legend()
    ax2[1].plot(df['날짜'], df['MACD'], label='MACD', color='blue')
    ax2[1].plot(df['날짜'], df['Signal'], label='Signal', color='red')
    ax2[1].axhline(0, color='gray', linestyle='--')
    ax2[1].legend()
    st.pyplot(fig2)

    # 💡 전략 제안
    st.subheader("💡 전략 제안")
    for s in strategy_suggestion(df):
        st.write("- " + s)

if __name__ == '__main__':
    main()
