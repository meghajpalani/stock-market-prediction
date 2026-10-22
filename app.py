import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="📈 Stock Trend Prediction", layout="wide")

st.title("📊 Stock Price Trend Prediction App")
st.write("Upload your stock data (CSV) and get predicted trends using a Random Forest model.")

# File upload
uploaded_file = st.file_uploader("Upload your CSV file (e.g. EW-MAX.csv)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"✅ Loaded {uploaded_file.name}")
    st.dataframe(df.head())

    # Detect columns
    PRICE_COLS = {
        'open': 'Open' if 'Open' in df.columns else df.columns[0],
        'high': 'High' if 'High' in df.columns else (df.columns[1] if len(df.columns)>1 else None),
        'low': 'Low' if 'Low' in df.columns else (df.columns[2] if len(df.columns)>2 else None),
        'close': 'Close' if 'Close' in df.columns else df.columns[-1],
        'volume': 'Volume' if 'Volume' in df.columns else None
    }

    close_col = PRICE_COLS['close']
    st.write(f"**Detected Close column:** `{close_col}`")

    # Feature engineering (same as in your .py file)
    df['return_1d'] = df[close_col].pct_change()
    df['return_3d'] = df[close_col].pct_change(3)
    df['ma_5'] = df[close_col].rolling(window=5).mean()
    df['ma_10'] = df[close_col].rolling(window=10).mean()
    df['std_5'] = df[close_col].rolling(window=5).std()
    df['std_10'] = df[close_col].rolling(window=10).std()
    df['momentum_5'] = df[close_col] - df[close_col].shift(5)

    def compute_rsi(series, period=14):
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -1 * delta.clip(upper=0)
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    df['rsi_14'] = compute_rsi(df[close_col], 14)

    for lag in [1, 2, 3, 5, 10]:
        df[f'lag_{lag}'] = df[close_col].shift(lag)

    df = df.dropna().reset_index(drop=True)

    st.write("### Processed Data Sample")
    st.dataframe(df.head())

    # Load or train model
    model_file = "rf_stock_model.joblib"
    try:
        model = joblib.load(model_file)
        st.success("✅ Loaded trained model successfully!")
    except:
        st.warning("⚠️ No saved model found — training a new one...")
        feature_cols = [c for c in df.columns if c not in ['Date'] and df[c].dtype in [float, int]]
        X = df[feature_cols]
        y = (df[close_col].shift(-1) > df[close_col]).astype(int)[:-1]
        X = X[:-1]
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        joblib.dump(model, model_file)
        st.success("✅ New model trained and saved!")

    # Predictions
    feature_cols = [c for c in df.columns if c not in ['Date'] and df[c].dtype in [float, int]]
    preds = model.predict(df[feature_cols])
    df['Prediction'] = np.where(preds == 1, "📈 Up", "📉 Down")

    st.write("### Predictions")
    st.dataframe(df[[close_col, 'Prediction']].tail(20))

    # Visualization
    st.write("### 📊 Trend Visualization")
    plt.figure(figsize=(12, 4))
    plt.plot(df[close_col], label="Close Price", color="blue")
    plt.title("Stock Price with Predicted Trend")
    plt.xlabel("Index")
    plt.ylabel("Price")
    st.pyplot(plt)

    st.download_button(
        label="⬇️ Download Predictions as CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="predictions.csv",
        mime="text/csv",
    )
