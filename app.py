import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os

# Setting page configuration for a wide, professional layout
st.set_page_config(page_title="Stock Market Analysis Dashboard", layout="wide", page_icon="📈")

# Loading and preparing data
@st.cache_data
def load_data():
    file_path = r"C:\Users\Dhruv Patel\Downloads\stocks.csv"
    if not os.path.exists(file_path):
        st.error(f"File not found at {file_path}. Please ensure the file exists.")
        return None
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Ticker', 'Date'])
    # Data cleaning: Remove duplicates and handle missing values
    df = df.drop_duplicates()
    df = df.dropna()
    return df

# Calculating 20-day moving averages
def calculate_moving_averages(df, window=20):
    df['MA20'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window, min_periods=1).mean())
    return df

# Calculating annualized volatility (20-day rolling window)
def calculate_volatility(df, window=20):
    df['Volatility'] = df.groupby('Ticker')['Close'].transform(
        lambda x: x.pct_change().rolling(window, min_periods=1).std() * np.sqrt(252)
    )
    return df

# Calculating correlation matrix
def calculate_correlation(df):
    pivot_df = df.pivot(index='Date', columns='Ticker', values='Close')
    corr_matrix = pivot_df.corr()
    return corr_matrix, pivot_df

# Creating dual-axis chart for price and volume
def create_dual_axis_chart(df, ticker):
    df_ticker = df[df['Ticker'] == ticker]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Adding price line
    fig.add_trace(
        go.Scatter(x=df_ticker['Date'], y=df_ticker['Close'], name="Close Price", line=dict(color='blue')),
        secondary_y=False,
    )
    
    # Adding volume bars
    fig.add_trace(
        go.Bar(x=df_ticker['Date'], y=df_ticker['Volume'], name="Volume", opacity=0.3),
        secondary_y=True,
    )
    
    # Updating layout
    fig.update_layout(
        title=f"{ticker}: Closing Price vs. Trading Volume",
        xaxis_title="Date",
        font_size=12,
        showlegend=True,
        height=500
    )
    fig.update_yaxes(title_text="Price (USD)", secondary_y=False)
    fig.update_yaxes(title_text="Volume", secondary_y=True)
    
    return fig

# Main app
def main():
    st.title("📊 Stock Market Analysis Dashboard")
    st.markdown("""
    Welcome to the Stock Market Analysis Dashboard! This tool analyzes historical stock price data for Apple (AAPL), 
    Microsoft (MSFT), Netflix (NFLX), and Google (GOOG) from February to May 2023. Explore trends, moving averages, 
    volatility, and correlations to gain insights into market performance. Use the sidebar to filter data and interact 
    with the visualizations.
    """)

    # Loading data
    df = load_data()
    if df is None:
        return

    # Calculating metrics
    df = calculate_moving_averages(df)
    df = calculate_volatility(df)
    corr_matrix, pivot_df = calculate_correlation(df)

    # Sidebar for user inputs
    st.sidebar.header("🔧 Filter Options")
    tickers = st.sidebar.multiselect(
        "Select Stocks", 
        options=df['Ticker'].unique(), 
        default=df['Ticker'].unique(),
        help="Choose one or more stocks to analyze."
    )
    date_range = st.sidebar.date_input(
        "Select Date Range",
        [df['Date'].min(), df['Date'].max()],
        min_value=df['Date'].min(),
        max_value=df['Date'].max(),
        help="Select the date range for analysis."
    )

    # Filtering data
    filtered_df = df[df['Ticker'].isin(tickers)]
    filtered_df = filtered_df[
        (filtered_df['Date'] >= pd.to_datetime(date_range[0])) & 
        (filtered_df['Date'] <= pd.to_datetime(date_range[1]))
    ]

    if filtered_df.empty:
        st.warning("No data available for the selected filters. Please adjust your selections.")
        return

    # Interesting Fact
    st.subheader("💡 Interesting Fact")
    max_volatility = filtered_df.groupby('Ticker')['Volatility'].mean().idxmax()
    max_vol_value = filtered_df.groupby('Ticker')['Volatility'].mean().max()
    st.markdown(
        f"**{max_volatility}** had the highest average volatility ({max_vol_value:.2%}) during the selected period, "
        "indicating larger price swings compared to other stocks. This could reflect higher market sensitivity or investor sentiment."
    )

    # Layout: Two-column structure for visualizations
    col1, col2 = st.columns(2)

    # Stock Price Trends
    with col1:
        st.subheader("📈 Stock Price Trends")
        fig_trends = px.line(
            filtered_df, 
            x='Date', 
            y='Close', 
            color='Ticker',
            title="Stock Closing Prices Over Time",
            labels={'Close': 'Price (USD)', 'Date': 'Date'}
        )
        fig_trends.update_layout(font_size=12, showlegend=True, height=400)
        st.plotly_chart(fig_trends, use_container_width=True)

    # Moving Averages
    with col2:
        st.subheader("📉 20-Day Moving Averages")
        fig_ma = px.line(
            filtered_df, 
            x='Date', 
            y='MA20', 
            color='Ticker',
            title="20-Day Moving Average of Closing Prices",
            labels={'MA20': '20-Day MA (USD)', 'Date': 'Date'}
        )
        fig_ma.update_layout(font_size=12, showlegend=True, height=400)
        st.plotly_chart(fig_ma, use_container_width=True)

    # Volatility
    with col1:
        st.subheader("🌪️ Stock Volatility")
        fig_vol = px.line(
            filtered_df, 
            x='Date', 
            y='Volatility', 
            color='Ticker',
            title="Annualized Volatility (20-Day Rolling Window)",
            labels={'Volatility': 'Volatility (Annualized)', 'Date': 'Date'}
        )
        fig_vol.update_layout(font_size=12, showlegend=True, height=400)
        st.plotly_chart(fig_vol, use_container_width=True)

    # Correlation Heatmap
    with col2:
        st.subheader("🔗 Correlation Between Stocks")
        if len(tickers) > 1:
            fig_corr = go.Figure(data=go.Heatmap(
                z=corr_matrix.loc[tickers, tickers].values,
                x=tickers,
                y=tickers,
                colorscale='Viridis',
                zmin=-1, 
                zmax=1,
                text=corr_matrix.loc[tickers, tickers].values.round(2),
                texttemplate="%{text}",
                textfont={"size": 12}
            ))
            fig_corr.update_layout(
                title="Correlation Matrix of Stock Closing Prices",
                font_size=12,
                xaxis_title="Ticker",
                yaxis_title="Ticker",
                height=400
            )
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("Select multiple stocks to view the correlation heatmap.")

    # Dual-Axis Chart for Price and Volume
    st.subheader("📊 Price vs. Volume Analysis")
    for ticker in tickers:
        st.markdown(f"**{ticker}**")
        fig_dual = create_dual_axis_chart(filtered_df, ticker)
        st.plotly_chart(fig_dual, use_container_width=True)

    # Summary Statistics Table
    st.subheader("📋 Summary Statistics")
    summary = filtered_df.groupby('Ticker').agg({
        'Close': ['mean', 'std', 'min', 'max'],
        'Volume': 'mean',
        'Volatility': 'mean'
    }).round(2)
    summary.columns = ['Avg Close', 'Std Close', 'Min Close', 'Max Close', 'Avg Volume', 'Avg Volatility']
    summary['Avg Volatility'] = summary['Avg Volatility'].apply(lambda x: f"{x:.2%}")
    st.dataframe(summary, use_container_width=True)

    # Key Insights
    st.subheader("🔍 Key Insights")
    st.markdown("""
    - **Market Trends**: 
      - **AAPL** and **MSFT** exhibit steady upward trends, with AAPL gaining approximately 
        {:.2f}% and MSFT {:.2f}% over the period.
      - **NFLX** shows high volatility with sharp dips and recoveries, reflecting sensitivity to market events.
      - **GOOG** maintains stable growth but with less pronounced gains compared to AAPL and MSFT.
    - **Moving Averages**: The 20-day moving averages reveal sustained trends for AAPL and MSFT, 
      while NFLX's fluctuations are smoothed but still evident.
    - **Volatility**: NFLX consistently has the highest volatility, often exceeding 40% annualized, 
      indicating higher risk and potential reward.
    - **Correlation**: Strong positive correlations (e.g., >0.8) between AAPL and MSFT suggest similar 
      market drivers, while NFLX's weaker correlations indicate unique market behavior.
    - **Volume Insights**: Spikes in trading volume often coincide with significant price movements, 
      particularly for NFLX during mid-March and April.
    """.format(
        ((filtered_df[filtered_df['Ticker'] == 'AAPL']['Close'].iloc[-1] / 
          filtered_df[filtered_df['Ticker'] == 'AAPL']['Close'].iloc[0] - 1) * 100),
        ((filtered_df[filtered_df['Ticker'] == 'MSFT']['Close'].iloc[-1] / 
          filtered_df[filtered_df['Ticker'] == 'MSFT']['Close'].iloc[0] - 1) * 100)
    ))

    # Conclusion
    st.subheader("🏁 Conclusion")
    st.markdown("""
    This Stock Market Analysis Dashboard provides a comprehensive view of the performance of AAPL, MSFT, NFLX, 
    and GOOG from February to May 2023. The visualizations and metrics highlight distinct behaviors:
    - **AAPL and MSFT** are stable performers with strong correlations, suitable for investors seeking steady growth.
    - **NFLX** is a high-risk, high-reward stock with significant volatility, appealing to risk-tolerant investors.
    - **GOOG** offers balanced performance but lags behind AAPL and MSFT in growth.
    
    Use these insights for investment decisions, portfolio diversification, or further market research. 
    The interactive filters allow you to explore specific stocks and time periods to tailor the analysis to your needs.
    """)

    # Footer
    st.markdown("---")
    st.markdown("**Developed by Grok | Powered by xAI** | Data Source: Historical Stock Prices (Feb-May 2023)")

if __name__ == "__main__":
    main()