import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import platform

# 한글 폰트 설정
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif platform.system() == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'
else:
    plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

# 업비트 OHLCV 데이터 수집
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

# RSI 계산
def compute_rsi(df, period=14):
    delta = df['종가'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# 지표 통합 계산
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
    return df

# 전략 해석
def strategy_suggestion(df):
    latest = df.iloc[-1]
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
    else:
        signals.append("MACD 중립 상태")

    if latest['종가'] < latest['Lower']:
        signals.append("📉 볼린저 밴드 하단 이탈 → 기술적 반등 가능성")
    elif latest['종가'] > latest['Upper']:
        signals.append("📈 볼린저 밴드 상단 돌파 → 과열 신호")
    else:
        signals.append("볼린저 밴드 내 안정 구간")

    # 종합 판단
    score = 0
    if latest['RSI'] < 30: score += 1
    if latest['종가'] < latest['Lower']: score += 1
    if latest['MACD'] > latest['Signal']: score += 1
    if latest['종가'] > latest['MA20'] and latest['MA20'] > latest['MA60']: score += 1
    
    if score >= 3:
        signals.append("📌 종합 판단: ✅ 매수 시점으로 유력")
    elif score <= 1:
        signals.append("📌 종합 판단: ⛔ 매도 또는 보류 추천")
    else:
        signals.append("📌 종합 판단: ⏳ 관망 구간")

    return signals

# Streamlit 앱 구성
def main():
    st.set_page_config(page_title="종합 코인 전략 분석기", layout="wide")
    st.title("📊 BTC/ETH/XRP RSI, 이평선, MACD, 볼린저밴드 기반 종합 전략 분석")

    coin_dict = {
        "비트코인 (BTC)": "KRW-BTC",
        "이더리움 (ETH)": "KRW-ETH",
        "리플 (XRP)": "KRW-XRP"
    }
    selected_coin = st.selectbox("분석할 코인을 선택하세요:", list(coin_dict.keys()))
    market_code = coin_dict[selected_coin]

    df = get_ohlcv(market_code)
    df = compute_indicators(df)

    st.subheader(f"📈 {selected_coin} 가격, 이동평균선, 볼린저밴드 차트")
    st.caption("파란선: 종가 / 주황선: 20일 이평선 / 초록선: 60일 이평선 / 회색 음영: 볼린저 밴드")
    fig, ax = plt.subplots()
    ax.plot(df['날짜'], df['종가'], label='종가', color='blue')
    ax.plot(df['날짜'], df['MA20'], label='MA20', color='orange')
    ax.plot(df['날짜'], df['MA60'], label='MA60', color='green')
    ax.fill_between(df['날짜'], df['Upper'], df['Lower'], color='gray', alpha=0.2, label='볼린저 밴드')
    ax.legend()
    st.pyplot(fig)

    st.subheader("📉 RSI와 MACD 차트")
    st.caption("상단: RSI (보라색), 하단: MACD(파랑) & Signal(빨강)")
    fig2, ax2 = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax2[0].plot(df['날짜'], df['RSI'], label='RSI', color='purple')
    ax2[0].axhline(70, color='red', linestyle='--')
    ax2[0].axhline(30, color='green', linestyle='--')
    ax2[0].set_ylabel("RSI")
    ax2[0].legend()

    ax2[1].plot(df['날짜'], df['MACD'], label='MACD', color='blue')
    ax2[1].plot(df['날짜'], df['Signal'], label='Signal', color='red')
    ax2[1].axhline(0, color='gray', linestyle='--')
    ax2[1].set_ylabel("MACD")
    ax2[1].legend()

    st.pyplot(fig2)

    st.subheader("💡 전략 제안")
    suggestions = strategy_suggestion(df)
    for s in suggestions:
        st.write("- " + s)

if __name__ == "__main__":
    main()
