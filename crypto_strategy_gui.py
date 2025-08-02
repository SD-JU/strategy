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
        if not data:
            break
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

# 📌 기술적 지표 계산 (기본 + ENA 단타 버전)
def compute_indicators(df, is_ena=False):
    # 이동평균
    if is_ena:
        df['EMA5'] = df['종가'].ewm(span=5, adjust=False).mean()
        df['EMA20_short'] = df['종가'].ewm(span=20, adjust=False).mean()
    df['MA20'] = df['종가'].rolling(window=20).mean()
    df['MA60'] = df['종가'].rolling(window=60).mean()

    # 볼린저 밴드 (모두 동일: 20, 2)
    df['STD'] = df['종가'].rolling(window=20).std()
    df['Upper'] = df['MA20'] + 2 * df['STD']
    df['Lower'] = df['MA20'] - 2 * df['STD']

    # RSI
    rsi_window = 7 if is_ena else 14
    delta = df['종가'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=rsi_window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=rsi_window).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Stochastic RSI (ENA만)
    if is_ena:
        # 먼저 RSI series
        rsi_series = df['RSI']
        stoch_rsi_period = 14
        smooth_k = 3
        smooth_d = 3
        min_rsi = rsi_series.rolling(window=stoch_rsi_period).min()
        max_rsi = rsi_series.rolling(window=stoch_rsi_period).max()
        df['StochRSI_raw'] = (rsi_series - min_rsi) / (max_rsi - min_rsi)
        df['StochRSI_K'] = df['StochRSI_raw'].rolling(window=smooth_k).mean()
        df['StochRSI_D'] = df['StochRSI_K'].rolling(window=smooth_d).mean()

    # MACD
    if is_ena:
        ema_fast = df['종가'].ewm(span=8, adjust=False).mean()
        ema_slow = df['종가'].ewm(span=21, adjust=False).mean()
        df['MACD'] = ema_fast - ema_slow
        df['Signal'] = df['MACD'].ewm(span=5, adjust=False).mean()
    else:
        df['EMA12'] = df['종가'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['종가'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    df['MACD_Hist'] = df['MACD'] - df['Signal']

    # 볼륨 기반
    df['VOL_MA20'] = df['거래량'].rolling(window=20).mean()
    df['VOL_RISE'] = df['거래량'] > df['VOL_MA20']

    # ATR 계산
    df['H-L'] = df['고가'] - df['저가']
    df['H-PC'] = abs(df['고가'] - df['종가'].shift(1))
    df['L-PC'] = abs(df['저가'] - df['종가'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=14).mean()

    # OBV
    df['OBV'] = (np.sign(df['종가'].diff()) * df['거래량']).fillna(0).cumsum()

    return df

# 📌 전략 제안 (ENA 단타용/기존 분기 처리)
def strategy_suggestion(df, is_ena=False):
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    signals = []

    # RSI 기반
    if latest['RSI'] < 30:
        signals.append("📉 RSI < 30 → 과매도: 매수 유력")
    elif latest['RSI'] > 70:
        signals.append("📈 RSI > 70 → 과매수: 매도 유력")
    else:
        signals.append(f"RSI {latest['RSI']:.2f}: 중립 구간")

    # EMA / 이평선 정배열
    if is_ena:
        if latest['종가'] > latest['EMA5'] > latest['EMA20_short']:
            signals.append("🔼 ENA 단기 정배열 (5EMA>20EMA): 상승 흐름")
        elif latest['종가'] < latest['EMA5'] < latest['EMA20_short']:
            signals.append("🔽 ENA 단기 역배열: 하락 흐름")
        else:
            signals.append("ENA 단기 이평선 혼조")
    else:
        if latest['종가'] > latest['MA20'] and latest['MA20'] > latest['MA60']:
            signals.append("🔼 이평선 정배열: 상승 추세")
        elif latest['종가'] < latest['MA20'] and latest['MA20'] < latest['MA60']:
            signals.append("🔽 이평선 역배열: 하락 추세")
        else:
            signals.append("이평선 혼조: 방향성 불분명")

    # MACD 모멘텀
    if latest['MACD'] > latest['Signal']:
        signals.append("🟢 MACD > Signal → 매수 모멘텀")
    elif latest['MACD'] < latest['Signal']:
        signals.append("🔴 MACD < Signal → 매도 모멘텀")

    # MACD 히스토그램 전환
    if latest['MACD_Hist'] > 0 and prev['MACD_Hist'] < 0:
        signals.append("🟢 MACD Histogram 양전환 → 매수 시그널 발생")
    elif latest['MACD_Hist'] < 0 and prev['MACD_Hist'] > 0:
        signals.append("🔴 MACD Histogram 음전환 → 매도 시그널 발생")

    # 볼린저
    if latest['종가'] < latest['Lower']:
        signals.append("📉 볼린저 밴드 하단 이탈 → 기술적 반등 가능성")
    elif latest['종가'] > latest['Upper']:
        signals.append("📈 볼린저 밴드 상단 돌파 → 과열 신호")

    # 과매도 + 밴드 하단 (ENA 포함)
    if latest['RSI'] < 30 and latest['종가'] < latest['Lower']:
        signals.append("📌 과매도 + 밴드 하단: 반등 확률 ↑")

    # 거래량
    if latest['VOL_RISE']:
        signals.append("💹 거래량 평균 상회 → 관심 집중")
    else:
        signals.append("🔕 거래량 평균 이하 → 관망")

    # OBV
    if latest['OBV'] > prev['OBV']:
        signals.append("📈 OBV 상승 → 매수세 유입")
    else:
        signals.append("📉 OBV 하락 → 매도세 우위")

    # 변동성
    if latest['ATR'] > df['ATR'].mean():
        signals.append("📊 ATR 상승 → 높은 변동성")
    else:
        signals.append("📉 ATR 하락 → 낮은 변동성")

    # StochRSI (ENA만)
    if is_ena:
        if 'StochRSI_K' in latest and 'StochRSI_D' in latest:
            k = latest['StochRSI_K']
            d = latest['StochRSI_D']
            if k > d and k > 0.8:
                signals.append("🔼 StochRSI 과매수 경고 (K>D & K>0.8)")
            elif k < d and k < 0.2:
                signals.append("🔽 StochRSI 과매도 반전 가능성 (K<D & K<0.2)")
            else:
                signals.append(f"StochRSI K:{k:.2f} D:{d:.2f} 중립")

    # 종합 점수 (단타 기준으로 ENA는 민감도 높음)
    score = 0
    if latest['RSI'] < 30: score += 1
    if latest['종가'] < latest['Lower']: score += 1
    if latest['MACD'] > latest['Signal']: score += 1
    if latest['MACD_Hist'] > 0 and prev['MACD_Hist'] < 0: score += 1
    if latest['OBV'] > prev['OBV']: score += 1
    if latest['VOL_RISE']: score += 1
    if is_ena:
        # ENA에 대해서는 StochRSI 긍정 조건 가중치
        if 'StochRSI_K' in latest and 'StochRSI_D' in latest:
            if latest['StochRSI_K'] > latest['StochRSI_D']:
                score += 1

    if score >= 5:
        signals.append("✅ 종합 판단: 강한 매수 신호")
    elif score <= 2:
        signals.append("⛔ 종합 판단: 매도 또는 관망")
    else:
        signals.append("⏳ 종합 판단: 중립 또는 약한 매수")

    return signals

# 📌 Streamlit 앱 시작
def main():
    st.set_page_config(page_title="종합 암호화폐 전략 분석기", layout="wide")
    st.title("📊 BTC / ETH / XRP / ENA 전략 분석 (기술적 + 심리적 지표 기반)")

    # 분석 기간 선택
    period_map = {"100일": 100, "180일 (6개월)": 180, "365일 (1년)": 365}
    selected_period_str = st.radio("분석 기간을 선택하세요:", list(period_map.keys()), horizontal=True)
    selected_period = period_map[selected_period_str]

    # 코인 선택 (ENA 추가)
    coin_dict = {
        "비트코인 (BTC)": "KRW-BTC",
        "이더리움 (ETH)": "KRW-ETH",
        "리플 (XRP)": "KRW-XRP",
        "에테나 (ENA)": "ENA/KRW" if False else "KRW-ENA"  # 업비트에서는 실제 마켓 코드가 "KRW-ENA"
    }
    selected_coin = st.selectbox("분석할 코인을 선택하세요:", list(coin_dict.keys()))
    market_code = coin_dict[selected_coin]
    is_ena = selected_coin.startswith("에테나")

    # 데이터 수집 및 분석
    df = get_ohlcv_extended(market_code, total_days=selected_period)
    df = compute_indicators(df, is_ena=is_ena)

    # 시세 차트
    st.subheader(f"📈 {selected_coin} 가격 및 기술적 지표")
    fig, ax = plt.subplots()
    ax.plot(df['날짜'], df['종가'], label='Close')
    if is_ena:
        ax.plot(df['날짜'], df['EMA5'], label='5EMA')
        ax.plot(df['날짜'], df['EMA20_short'], label='20EMA')
    else:
        ax.plot(df['날짜'], df['MA20'], label='MA20')
        ax.plot(df['날짜'], df['MA60'], label='MA60')
        ax.fill_between(df['날짜'], df['Upper'], df['Lower'], alpha=0.2, label='Bollinger Bands')
    ax.legend()
    st.pyplot(fig)

    # 보조 지표 차트
    st.subheader("📉 보조 지표: RSI / MACD / 거래량" + (" / StochRSI" if is_ena else ""))
    if is_ena:
        fig2, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
        axs[0].plot(df['날짜'], df['RSI'], label='RSI')
        axs[0].axhline(70, linestyle='--')
        axs[0].axhline(30, linestyle='--')
        axs[0].legend()
        axs[1].plot(df['날짜'], df['MACD'], label='MACD')
        axs[1].plot(df['날짜'], df['Signal'], label='Signal')
        axs[1].axhline(0, linestyle='--')
        axs[1].legend()
        axs[2].plot(df['날짜'], df['StochRSI_K'], label='StochRSI K')
        axs[2].plot(df['날짜'], df['StochRSI_D'], label='StochRSI D')
        axs[2].legend()
        axs[3].bar(df['날짜'], df['거래량'], label='Volume')
        axs[3].plot(df['날짜'], df['VOL_MA20'], label='Volume Avg')
        axs[3].legend()
        st.pyplot(fig2)
    else:
        fig2, ax2 = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        ax2[0].plot(df['날짜'], df['RSI'], label='RSI')
        ax2[0].axhline(70, linestyle='--')
        ax2[0].axhline(30, linestyle='--')
        ax2[0].legend()
        ax2[1].plot(df['날짜'], df['MACD'], label='MACD')
        ax2[1].plot(df['날짜'], df['Signal'], label='Signal')
        ax2[1].axhline(0, linestyle='--')
        ax2[1].legend()
        ax2[2].bar(df['날짜'], df['거래량'], label='Volume')
        ax2[2].plot(df['날짜'], df['VOL_MA20'], label='Volume Avg')
        ax2[2].legend()
        st.pyplot(fig2)

    # 전략 제안
    st.subheader("💡 전략 제안")
    suggestions = strategy_suggestion(df, is_ena=is_ena)
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
