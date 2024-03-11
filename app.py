import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots


@st.cache_data
def collect_sp500_list():
    tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    return tables[0]

@st.cache_data
def collect_basic_info(ticker):
    yf_ticker = yf.Ticker(ticker)
    basic_info = yf_ticker.get_info()
    return basic_info

@st.cache_data
def collect_news(ticker):
    yf_ticker = yf.Ticker(ticker)
    news = yf_ticker.get_news()
    return news

@st.cache_data
def collect_prices_and_returns(ticker, time_period):
    df = yf.download(ticker, period = time_period)
    df.loc[:, "r_t"] = df.loc[:, ["Adj Close"]].pct_change()
    df.dropna(inplace = True)
    return df

@st.cache_data
def collect_market_returns(time_period):
    df_market = yf.download(["^RUA", "^GSPC"], period = time_period)
    market_returns = df_market.loc[:, "Adj Close"].pct_change().dropna()
    market_returns.columns = ["SP 500", "Russell 3000"]
    return market_returns


sp500_list = collect_sp500_list()
tickers = sp500_list.Symbol.tolist()
tickers.sort()


st.title("Stock watcher")
st.markdown("""The following is a *demo* for a streamlit app which
            is made to examine some key characteristics and information
            of a stock market listed company.""")

with st.sidebar:
    st.title("Selections")
    ticker = st.selectbox(
        'Choose a ticker symbol:',
        tickers, key = "ticker_choice")
    time_period = st.selectbox(
        'Choose a historical time period:',
        ["1y", "3mo", "6mo", "3y", "5y"])
    
    

ticker_info_one = sp500_list[sp500_list.Symbol == ticker]
company_name = ticker_info_one["Security"].tolist()[0]
company_gic = ticker_info_one["GICS Sector"].tolist()[0]
company_cik = ticker_info_one["CIK"].tolist()[0]
ticker_info_two = collect_basic_info(ticker)
industry_info = pd.DataFrame([ticker_info_two["industry"], ticker_info_two["sector"], ticker_info_two["fullTimeEmployees"]], index = ["Industry", "Sector", "Employees"], columns = [""])
industry_info.index.name = "Basic info"


st.header(f"{company_name} (CIK: {company_cik})")
st.markdown(ticker_info_two["longBusinessSummary"])
st.dataframe(industry_info, width = 600)

news = collect_news(ticker)
headlines = [fnews["title"] for fnews in news]
urls = [fnews["link"] for fnews in news]
relatedness = ["company only" if len(fnews["relatedTickers"]) == 1 else "multiple companies" for fnews in news]

#news_df = pd.DataFrame(news)
#news_df = news_df.loc[:, ["title", "link"]]
#st.dataframe(news_df)

st.subheader("Current news:")
for related, headline, url in zip(relatedness, headlines, urls):
    st.text("-"*100)
    st.markdown(f"Relates to: {related} - " + headline + f" [(link)]({url}) \n")
st.text("-"*100)

df = collect_prices_and_returns(ticker, time_period)

# Create subplots and mention plot grid size
fig_ohlc = make_subplots(rows=2, cols=1, shared_xaxes=True, 
               vertical_spacing=0.10, subplot_titles=('OHLC', 'Volume'), 
               row_width=[0.2, 0.7])

# Plot OHLC on 1st row
# include candlestick with rangeselector
fig_ohlc.add_trace(go.Candlestick(
    x = df.index,
    open=df['Open'], high=df['High'],
    low=df['Low'], close=df['Close'], increasing_line_color= '#636EFA', decreasing_line_color= '#EF553B'),
    row=1, col=1
)

# Bar trace for volumes on 2nd row without legend
fig_ohlc.add_trace(go.Bar(x = df.index, y=df['Volume'], marker_color = 'rgb(204, 204, 204)', showlegend=False), row=2, col=1)

# Do not show OHLC's rangeslider plot 
fig_ohlc.update(layout_xaxis_rangeslider_visible=False)
fig_ohlc.update_layout(showlegend=False) 

st.subheader(f"OHLC chart")
st.markdown(
    """
        The OHLC chart stand for open-high-low-close. The color indicates on which day the close price is higher than the opening price.
        Blue bars are days with a stock price increase and red bars represent days with a market value decrease. Below you can also observe volume data.
        Volume is the number of trades per day. The higher the volume the more important a corresponding price change. Note that we only see the market development
        by looking at OHLC or simpler price line charts. Usually, it is not possibly to infer the performance, especially, the risk adjusted performance from 
        this type of visualization. This is better than with plots and descriptive metrics from return data at which you can take a look below.
    """
)
st.plotly_chart(fig_ohlc, use_container_width=True)


fig_uni = make_subplots(rows = 2, cols = 1, shared_xaxes = True, subplot_titles = ["Discrete returns", "Absolute discrete returns"])
fig_uni.add_trace(go.Scatter(x = df.index, y = df.r_t, name = r"discrete return", marker_color = '#636EFA'), row = 1, col = 1)
fig_uni.add_trace(go.Scatter(x = df.index, y = df.r_t.rolling(10).mean(), name = "rolling average: t = 20", marker_color = '#EF553B'), row = 1, col = 1)
fig_uni.add_trace(go.Scatter(x = df.index, y = df.r_t.abs(), name = r"abs. discrete return", marker_color = '#636EFA'), row = 2, col = 1)
fig_uni.add_trace(go.Scatter(x = df.index, y = df.r_t.abs().rolling(10).mean(), name = "rolling average: t = 20", marker_color = '#EF553B'), row = 2, col = 1)
fig_uni.update_layout(
    xaxis2=dict(
        rangeslider=dict(visible=True),
        type="date",
        range=[df.index[0], df.index[-1]]
    ),
    showlegend = False
)

st.subheader(f"Univariate analysis")
st.markdown(
    r"""
        Given the (adjusted closing) price $p_t$ of a stock at time $t$. The discrete return $r_t$ is defined as $r_t$:
    """
)
st.latex(r"r_t = \frac{p_t}{p_{t-1}} - 1")
st.markdown(
    """
        Traditionally, one is interested in the average return which is an estimate for its expected value. Moreover, by economic reasoning,
        investments which are more risky should pay higher returns because a rational (risk averse) investor would always invest her money into the 
        investment with lower risk given the same profit. Thus, besides the profit oriented perspective, analyzing returns is also about analyzing the 
        risk of an investment. Risk is often capture by volatility which is the standard deviation of returns. Below you can take a look at some 
        descriptive statistics and visualizations. The red line is the 20 day rolling average and can be seen as an estimate for time varying behavior
        which usually can be found for the mean and volatility of stock returns.
    """
)

returns_descriptive = df.r_t.describe(percentiles=[0.05, 0.50, 0.95]).to_frame().transpose().iloc[:, 1:]
returns_descriptive.index = [""]
st.dataframe(returns_descriptive.round(4), width = 600)
st.plotly_chart(fig_uni, use_container_width=True)


market_returns = collect_market_returns(time_period)
df_all = market_returns.merge(df, left_index = True, right_index = True)

lr_sp = LinearRegression()
lr_sp.fit(df_all.loc[:, ["SP 500"]], df_all.r_t)

lr_rua = LinearRegression()
lr_rua.fit(df_all.loc[:, ["Russell 3000"]], df_all.r_t)


fig_bi = make_subplots(rows = 1, cols = 2, subplot_titles = ["S&P 500", "Russell 3000"])
fig_bi.add_trace(go.Scatter(x = df_all["SP 500"], y = df_all.r_t, mode = "markers", marker_color = '#636EFA'), row = 1, col = 1)
fig_bi.add_trace(go.Scatter(x = df_all["SP 500"], y = lr_sp.predict(df_all.loc[:, ["SP 500"]]), marker_color = '#EF553B'), row = 1, col = 1)

fig_bi.add_trace(go.Scatter(x = df_all["Russell 3000"], y = df_all.r_t, mode = "markers", marker_color = '#636EFA'), row = 1, col = 2)
fig_bi.add_trace(go.Scatter(x = df_all["Russell 3000"], y = lr_rua.predict(df_all.loc[:, ["Russell 3000"]]), marker_color = '#EF553B'), row = 1, col = 2)

fig_bi.update_layout(showlegend = False, title = "Discrete return correlation")
fig_bi["layout"]["xaxis"]["title"] = "^GSPC"
fig_bi["layout"]["yaxis"]["title"] = ticker
fig_bi["layout"]["xaxis2"]["title"] = "^RUA"

corr_sp = np.corrcoef(df_all.loc[:, "SP 500"].values, df_all.r_t.values, rowvar = False)[0, 1]
corr_rua = np.corrcoef(df_all.loc[:, "Russell 3000"].values, df_all.r_t.values, rowvar = False)[0, 1]
df_dependence = pd.DataFrame([[corr_sp, lr_sp.coef_[0]], [corr_rua, lr_rua.coef_[0]]], index = ["S&P 500", "Russell 3000"], columns = ["Correlation", "Beta"])

st.subheader(f"Market dependence")
st.markdown(
    """
        Speaking from a financial markets theory perspective, univariate attributes of returns distributions are less important to the common exposure to
        systematic risk factors. Companies which are more exposed to systematic risk factors tend to behave similar and the more similar companies behave, 
        the less we can reduce the risk in an investment position holding multiple companies. Bravais-Pearson correlation is an estimate to quantifiy
        the linear co-movement of stock returns. It is often used to examine dependencies, however, it may underestimate the co-movement of extreme changes
        in the market price of companies. Another traditional way to examine the systematic exposure of a company's return towards the market. For the sake
        of simplicity we regress the company's return on the S&P 500 and the Russell 3000 below. This only suffices to examine systematic risk exposure,
        given these indices are good approximations of the (unobservable) market portfolio and if the single risk factor is enough to explain all company co-movements.  
    """
)


st.dataframe(df_dependence.round(4), width = 600)
st.plotly_chart(fig_bi)
st.text(f"Estimates are based on {df_all.shape[0]} observations, stock time series includes {df.shape[0]} observations.")