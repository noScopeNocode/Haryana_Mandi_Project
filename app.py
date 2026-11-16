# ── app.py : Haryana Mandi Insights Dashboard ─────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json
import os
import warnings
warnings.filterwarnings('ignore')

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Haryana Mandi Insights",
    page_icon  = "🌾",
    layout     = "wide",
    initial_sidebar_state = "expanded"
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_PATH   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_PATH, 'Data', 'Processed') + os.sep
MODELS_PATH = os.path.join(BASE_PATH, 'Models') + os.sep
# ── Load Data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(
        DATA_PATH + 'Haryana_Mandi_Cleaned.csv',
        parse_dates=['Arrival_Date']
    )
    monthly = pd.read_csv(DATA_PATH + 'monthly_modal_prices.csv')
    monthly['Month'] = pd.to_datetime(monthly['Month'])

    summary = pd.read_csv(DATA_PATH + 'commodity_summary_stats.csv')

    forecast = pd.read_csv(MODELS_PATH + 'forecast_Wheat.csv')
    forecast['Month'] = pd.to_datetime(forecast['Month'])

    model_res = pd.read_csv(MODELS_PATH + 'model_results_Wheat.csv')

    test_preds = pd.read_csv(
        MODELS_PATH + 'test_predictions_Wheat.csv'
    )
    test_preds['Month'] = pd.to_datetime(test_preds['Month'])

    return df, monthly, summary, forecast, model_res, test_preds

@st.cache_resource
def load_meta():
    with open(MODELS_PATH + 'all_model_stats.json', 'r') as f:
        all_stats = json.load(f)
    with open(MODELS_PATH + 'model_features.json', 'r') as f:
        features = json.load(f)
    return all_stats, features

@st.cache_resource
def load_commodity_model(safe_name, model_name):
    safe_model = model_name.replace(' ', '_')
    path       = MODELS_PATH + f'model_{safe_name}_{safe_model}.pkl'
    return joblib.load(path)

df, monthly_df, summary_df, forecast_df, \
    model_results, test_preds = load_data()
all_model_stats, model_features = load_meta()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("Haryana Mandi Insights")
st.sidebar.markdown("Agricultural Price Analysis & Forecasting")
st.sidebar.markdown("---")

tab = st.sidebar.radio("Navigate", [
    "Overview",
    "Price Explorer",
    "Trend Analysis",
    "Market Comparison",
    "Price Forecast",
    "Price Predictor"
])

st.sidebar.markdown("---")
st.sidebar.markdown("**Districts Covered:**")
for d in sorted(df['District'].unique()):
    st.sidebar.markdown(f"- {d}")
st.sidebar.markdown("---")
st.sidebar.caption("Data: 2022-2026 | Haryana Agri Markets")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if tab == "Overview":
    st.title("Haryana Mandi Insights")
    st.markdown(
        "#### Agricultural Price Analysis across "
        "6 Districts of Haryana (2022-2026)"
    )
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records",  f"{len(df):,}")
    col2.metric("Unique Markets", df['Market'].nunique())
    col3.metric("Commodities",    df['Commodity'].nunique())
    col4.metric("Date Range",
                f"{df['Arrival_Date'].min().year}"
                f"-{df['Arrival_Date'].max().year}")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Records per District")
        dist_counts = df['District'].value_counts().reset_index()
        dist_counts.columns = ['District', 'Records']
        fig = px.bar(dist_counts, x='Records', y='District',
                     orientation='h', color='Records',
                     color_continuous_scale='Blues',
                     text='Records')
        fig.update_traces(texttemplate='%{text:,}',
                          textposition='outside')
        fig.update_layout(showlegend=False, height=350,
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Markets per District")
        mkt_dist = df.groupby('District')['Market'].nunique()\
                     .reset_index()\
                     .sort_values('Market', ascending=False)
        mkt_dist.columns = ['District', 'Markets']
        fig = px.bar(mkt_dist, x='Markets', y='District',
                     orientation='h', color='Markets',
                     color_continuous_scale='Greens',
                     text='Markets')
        fig.update_traces(textposition='outside')
        fig.update_layout(showlegend=False, height=350,
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("Top 15 Commodities by Records")
    top15 = df['Commodity'].value_counts().head(15).reset_index()
    top15.columns = ['Commodity', 'Records']
    fig = px.bar(top15, x='Commodity', y='Records',
                 color='Records',
                 color_continuous_scale='Oranges',
                 text='Records')
    fig.update_traces(texttemplate='%{text:,}',
                      textposition='outside')
    fig.update_layout(height=400, coloraxis_showscale=False,
                      xaxis_tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("Commodity Summary Statistics")
    st.dataframe(summary_df.head(20), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PRICE EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif tab == "Price Explorer":
    st.title("Price Explorer")
    st.markdown(
        "Explore modal prices filtered by commodity, "
        "district, and date range."
    )
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        selected_commodity = st.selectbox(
            "Select Commodity",
            sorted(df['Commodity'].unique())
        )
    with col2:
        selected_district = st.multiselect(
            "Select District(s)",
            sorted(df['District'].unique()),
            default=sorted(df['District'].unique())
        )
    with col3:
        date_range = st.date_input(
            "Date Range",
            value=[df['Arrival_Date'].min(),
                   df['Arrival_Date'].max()]
        )

    filtered = df[
        (df['Commodity'] == selected_commodity) &
        (df['District'].isin(selected_district)) &
        (df['Arrival_Date'] >= pd.to_datetime(date_range[0])) &
        (df['Arrival_Date'] <= pd.to_datetime(date_range[1]))
    ]

    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Records",   f"{len(filtered):,}")
    col2.metric("Avg Price",
                f"Rs {filtered['Modal_Price'].mean():,.0f}")
    col3.metric("Max Price",
                f"Rs {filtered['Modal_Price'].max():,.0f}")
    col4.metric("Min Price",
                f"Rs {filtered['Modal_Price'].min():,.0f}")

    st.markdown("---")

    st.subheader(f"{selected_commodity} - Modal Price Over Time")
    daily = filtered.groupby('Arrival_Date')\
                    ['Modal_Price'].mean().reset_index()
    fig = px.line(daily, x='Arrival_Date', y='Modal_Price',
                  labels={
                      'Modal_Price' : 'Avg Modal Price (Rs/Quintal)',
                      'Arrival_Date': 'Date'
                  },
                  color_discrete_sequence=['steelblue'])
    fig.update_traces(line_width=2)
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Price Distribution")
        fig = px.histogram(filtered, x='Modal_Price', nbins=40,
                           color_discrete_sequence=['steelblue'],
                           labels={
                               'Modal_Price':
                               'Modal Price (Rs/Quintal)'
                           })
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Price Box by District")
        fig = px.box(filtered, x='District', y='Modal_Price',
                     color='District',
                     labels={
                         'Modal_Price':
                         'Modal Price (Rs/Quintal)'
                     })
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    with st.expander("View Raw Data"):
        st.dataframe(
            filtered[['Arrival_Date', 'Commodity', 'District',
                       'Market', 'Min_Price', 'Modal_Price',
                       'Max_Price', 'Season']]\
            .sort_values('Arrival_Date', ascending=False),
            use_container_width=True
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — TREND ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif tab == "Trend Analysis":
    st.title("Trend Analysis")
    st.markdown(
        "Yearly trends, monthly seasonality "
        "and agricultural season patterns."
    )
    st.markdown("---")

    top5 = df['Commodity'].value_counts().head(5).index.tolist()

    st.subheader("Yearly Average Modal Price - Top 5 Commodities")
    yearly = df[df['Commodity'].isin(top5)]\
               .groupby(['Year', 'Commodity'])['Modal_Price']\
               .mean().reset_index()
    fig = px.line(yearly, x='Year', y='Modal_Price',
                  color='Commodity', markers=True,
                  labels={
                      'Modal_Price':
                      'Avg Modal Price (Rs/Quintal)'
                  },
                  color_discrete_sequence=
                  px.colors.qualitative.Set2)
    fig.update_traces(line_width=2.5, marker_size=8)
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("Monthly Seasonality Pattern")
    month_labels = {
        1:'Jan',  2:'Feb',  3:'Mar',  4:'Apr',
        5:'May',  6:'Jun',  7:'Jul',  8:'Aug',
        9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'
    }
    selected_comm = st.multiselect(
        "Select Commodities", top5, default=top5[:3]
    )
    monthly_avg = df[df['Commodity'].isin(selected_comm)]\
                    .groupby(['Month', 'Commodity'])\
                    ['Modal_Price'].mean().reset_index()
    fig = px.line(monthly_avg, x='Month', y='Modal_Price',
                  color='Commodity', markers=True,
                  labels={
                      'Modal_Price':
                      'Avg Modal Price (Rs/Quintal)',
                      'Month': 'Month'
                  },
                  color_discrete_sequence=
                  px.colors.qualitative.Set1)
    fig.update_xaxes(
        tickvals=list(range(1, 13)),
        ticktext=list(month_labels.values())
    )
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("Price by Agricultural Season")
    season_data = df[df['Commodity'].isin(top5)]\
                    .groupby(['Season', 'Commodity'])\
                    ['Modal_Price'].mean().reset_index()
    fig = px.bar(season_data, x='Season', y='Modal_Price',
                 color='Commodity', barmode='group',
                 category_orders={
                     'Season': ['Rabi', 'Zaid', 'Kharif']
                 },
                 labels={
                     'Modal_Price':
                     'Avg Modal Price (Rs/Quintal)'
                 },
                 color_discrete_sequence=
                 px.colors.qualitative.Set2)
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MARKET COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
elif tab == "Market Comparison":
    st.title("Market Comparison")
    st.markdown(
        "Compare prices across different mandis "
        "for the same commodity."
    )
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        comp_commodity = st.selectbox(
            "Select Commodity",
            sorted(df['Commodity'].unique()), key='mc'
        )
    with col2:
        comp_district = st.selectbox(
            "Select District",
            ['All'] + sorted(df['District'].unique())
        )

    comp_df = df[df['Commodity'] == comp_commodity]
    if comp_district != 'All':
        comp_df = comp_df[comp_df['District'] == comp_district]

    market_stats = comp_df.groupby('Market')['Modal_Price']\
                           .agg(['mean', 'std', 'count'])\
                           .query('count >= 5')\
                           .sort_values('mean', ascending=False)\
                           .reset_index()
    market_stats.columns = ['Market', 'Avg_Price',
                             'Std_Price', 'Count']
    market_stats['CV_%'] = (
        market_stats['Std_Price'] /
        market_stats['Avg_Price'] * 100
    ).round(2)

    st.markdown("---")

    st.subheader(
        f"Avg Modal Price by Market - {comp_commodity}"
    )
    fig = px.bar(market_stats.head(20),
                 x='Market', y='Avg_Price',
                 color='Avg_Price',
                 color_continuous_scale='RdYlGn',
                 text='Avg_Price',
                 labels={
                     'Avg_Price':
                     'Avg Modal Price (Rs/Quintal)'
                 })
    fig.update_traces(texttemplate='Rs %{text:,.0f}',
                      textposition='outside')
    fig.update_layout(height=430, xaxis_tickangle=-35,
                      coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Price Volatility (CV%)")
        fig = px.bar(
            market_stats.head(15)\
                         .sort_values('CV_%', ascending=False),
            x='CV_%', y='Market', orientation='h',
            color='CV_%', color_continuous_scale='Reds',
            labels={'CV_%': 'Coefficient of Variation (%)'}
        )
        fig.update_layout(height=420,
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("District-wise Avg Price")
        dist_comp = df[df['Commodity'] == comp_commodity]\
                      .groupby('District')['Modal_Price']\
                      .mean().reset_index()\
                      .sort_values('Modal_Price', ascending=False)
        dist_comp.columns = ['District', 'Avg_Price']
        fig = px.bar(dist_comp, x='District', y='Avg_Price',
                     color='Avg_Price',
                     color_continuous_scale='Blues',
                     text='Avg_Price',
                     labels={
                         'Avg_Price':
                         'Avg Modal Price (Rs/Quintal)'
                     })
        fig.update_traces(texttemplate='Rs %{text:,.0f}',
                          textposition='outside')
        fig.update_layout(height=420,
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    with st.expander("Full Market Statistics Table"):
        st.dataframe(market_stats, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — PRICE FORECAST
# ══════════════════════════════════════════════════════════════════════════════
elif tab == "Price Forecast":
    st.title("Price Forecast")
    st.markdown(
        "ML model predictions and 6-month "
        "price forecast for Wheat."
    )
    st.markdown("---")

    st.subheader("Model Performance Comparison")
    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(model_results.reset_index(),
                     x='index', y='RMSE',
                     color='RMSE',
                     color_continuous_scale='Reds',
                     text='RMSE',
                     labels={
                         'index': 'Model',
                         'RMSE' : 'RMSE (Rs/Quintal)'
                     })
        fig.update_traces(texttemplate='Rs %{text:.1f}',
                          textposition='outside')
        fig.update_layout(height=350, title='RMSE by Model',
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(model_results.reset_index(),
                     x='index', y='MAPE_%',
                     color='MAPE_%',
                     color_continuous_scale='Oranges',
                     text='MAPE_%',
                     labels={
                         'index'  : 'Model',
                         'MAPE_%' : 'MAPE (%)'
                     })
        fig.update_traces(texttemplate='%{text:.2f}%',
                          textposition='outside')
        fig.update_layout(height=350, title='MAPE by Model',
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("Actual vs Predicted - Test Set")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=test_preds['Month'],
        y=test_preds['Modal_Price'],
        mode='lines+markers', name='Actual',
        line=dict(color='steelblue', width=2.5),
        marker=dict(size=8)
    ))
    fig.add_trace(go.Scatter(
        x=test_preds['Month'],
        y=test_preds['Predicted'],
        mode='lines+markers', name='Predicted',
        line=dict(color='tomato', width=2.5, dash='dash'),
        marker=dict(size=7, symbol='square')
    ))
    fig.update_layout(
        height=420,
        xaxis_title='Month',
        yaxis_title='Modal Price (Rs/Quintal)',
        legend=dict(orientation='h', y=1.1)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("6-Month Price Forecast - Wheat")
    fig = go.Figure()

    wheat_hist = monthly_df[monthly_df['Commodity'] == 'Wheat']
    fig.add_trace(go.Scatter(
        x=wheat_hist['Month'],
        y=wheat_hist['Modal_Price'],
        mode='lines+markers', name='Historical',
        line=dict(color='steelblue', width=2),
        marker=dict(size=5)
    ))
    fig.add_trace(go.Scatter(
        x=forecast_df['Month'],
        y=forecast_df['Forecasted_Price'],
        mode='lines+markers', name='Forecast',
        line=dict(color='green', width=2.5, dash='dash'),
        marker=dict(size=9, symbol='triangle-up')
    ))
    fig.add_vrect(
        x0=forecast_df['Month'].min(),
        x1=forecast_df['Month'].max(),
        fillcolor='green', opacity=0.07,
        annotation_text='Forecast Zone',
        annotation_position='top left'
    )
    fig.update_layout(
        height=450,
        xaxis_title='Month',
        yaxis_title='Modal Price (Rs/Quintal)',
        legend=dict(orientation='h', y=1.1)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("Forecast Values")
    forecast_display = forecast_df[
        ['Month', 'Forecasted_Price', 'Model']
    ].copy()
    forecast_display['Month'] = \
        forecast_display['Month'].dt.strftime('%b %Y')
    forecast_display['Forecasted_Price'] = \
        forecast_display['Forecasted_Price']\
        .apply(lambda x: f"Rs {x:,.2f}")
    st.dataframe(forecast_display, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — PRICE PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
elif tab == "Price Predictor":
    st.title("Commodity Price Predictor")
    st.markdown(
        "Select a commodity, district, month and model "
        "to predict the expected Modal Price."
    )
    st.markdown("---")

    col1, col2 = st.columns([1, 1])

    # ── Left Column: Inputs ───────────────────────────────────────
    with col1:
        st.subheader("Input Parameters")

        available_commodities = sorted(all_model_stats.keys())
        selected_commodity    = st.selectbox(
            "Select Commodity", available_commodities
        )

        stats = all_model_stats[selected_commodity]

        selected_district = st.selectbox(
            "Select District",
            ['All Districts'] + sorted(df['District'].unique())
        )

        pred_year = st.selectbox(
            "Year",
            options=[2024, 2025, 2026],
            index=2
        )

        month_map = {
            1 : 'January',   2 : 'February',
            3 : 'March',     4 : 'April',
            5 : 'May',       6 : 'June',
            7 : 'July',      8 : 'August',
            9 : 'September', 10: 'October',
            11: 'November',  12: 'December'
        }
        pred_month_num = st.selectbox(
            "Month",
            options=list(month_map.keys()),
            format_func=lambda x: month_map[x]
        )

        model_names    = [
            'Linear Regression', 'Random Forest',
            'Gradient Boosting', 'XGBoost'
        ]
        selected_model = st.selectbox(
            "Select Model", model_names
        )

        best_model_name = stats['best_model']
        best_rmse       = stats['all_rmse'][best_model_name]
        st.info(
            f"Best model for {selected_commodity}: "
            f"**{best_model_name}** "
            f"(RMSE: Rs {best_rmse:,.0f})"
        )

        predict_btn = st.button(
            "Predict Price",
            use_container_width=True,
            type='primary'
        )

    # ── Right Column: Results ──────────────────────────────────────
    with col2:
        st.subheader("Prediction Result")

        if predict_btn:
            # Load selected model
            try:
                model = load_commodity_model(
                    stats['safe_name'], selected_model
                )
            except Exception as e:
                st.error(
                    f"Model file not found for "
                    f"{selected_commodity} — "
                    f"{selected_model}. "
                    f"Error: {e}"
                )
                st.stop()

            # Auto-calculate lags from historical data
            hist = df[df['Commodity'] == selected_commodity]
            if selected_district != 'All Districts':
                hist = hist[
                    hist['District'] == selected_district
                ]

            if len(hist) > 0:
                hist_monthly = hist.groupby(
                    pd.Grouper(
                        key='Arrival_Date', freq='ME'
                    )
                )['Modal_Price'].mean().dropna()

                if len(hist_monthly) >= 6:
                    lag1 = float(hist_monthly.iloc[-1])
                    lag2 = float(hist_monthly.iloc[-2])
                    lag3 = float(hist_monthly.iloc[-3])
                    lag6 = float(hist_monthly.iloc[-6])
                else:
                    lag1 = stats['last_lag1']
                    lag2 = stats['last_lag2']
                    lag3 = stats['last_lag3']
                    lag6 = stats['last_lag6']
            else:
                lag1 = stats['last_lag1']
                lag2 = stats['last_lag2']
                lag3 = stats['last_lag3']
                lag6 = stats['last_lag6']

            quarter     = (pred_month_num - 1) // 3 + 1
            roll_mean_3 = np.mean([lag1, lag2, lag3])
            roll_mean_6 = np.mean(stats['last_values'][-6:])
            roll_std_3  = np.std([lag1, lag2, lag3])
            momentum    = lag1 - lag3

            input_data = pd.DataFrame([{
                'Year'       : pred_year,
                'Month_N'    : pred_month_num,
                'Quarter'    : quarter,
                'Lag_1'      : lag1,
                'Lag_2'      : lag2,
                'Lag_3'      : lag3,
                'Lag_6'      : lag6,
                'Roll_Mean_3': roll_mean_3,
                'Roll_Mean_6': roll_mean_6,
                'Roll_Std_3' : roll_std_3,
                'Momentum'   : momentum
            }])

            predicted_price = model.predict(input_data)[0]
            price_low       = predicted_price * 0.95
            price_high      = predicted_price * 1.05
            avg_price       = stats['avg_price']
            diff_pct        = (
                (predicted_price - avg_price) / avg_price
            ) * 100

            # Prediction card
            st.markdown(f"""
            <div style="background: linear-gradient(
                            135deg, #1a1a2e, #16213e);
                        border-radius: 15px;
                        padding: 30px;
                        text-align: center;
                        border: 1px solid #0f3460;
                        margin-bottom: 20px;">
                <p style="color: #a0aec0;
                           font-size: 14px; margin: 0;">
                    PREDICTED MODAL PRICE
                </p>
                <h1 style="color: #63b3ed;
                            font-size: 56px;
                            margin: 10px 0;">
                    Rs {predicted_price:,.0f}
                </h1>
                <p style="color: #a0aec0;
                           font-size: 13px;">
                    per Quintal — {selected_commodity}
                </p>
                <p style="color: #718096;
                           font-size: 12px;
                           margin: 5px 0 0 0;">
                    {month_map[pred_month_num]} {pred_year}
                    | {selected_district}
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Price range
            st.markdown(f"""
            <div style="background: #1a202c;
                        border-radius: 10px;
                        padding: 15px;
                        text-align: center;
                        margin-bottom: 15px;">
                <p style="color: #a0aec0;
                           margin: 0; font-size: 13px;">
                    Expected Range (95% - 105%)
                </p>
                <p style="color: #68d391;
                           font-size: 20px;
                           margin: 5px 0;
                           font-weight: bold;">
                    Rs {price_low:,.0f}
                    — Rs {price_high:,.0f}
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Market signal
            if diff_pct >= 5:
                signal       = "Above Average — Good time to sell!"
                signal_color = "#68d391"
                bg_color     = "#1a2e1a"
            elif diff_pct <= -5:
                signal       = "Below Average — Consider waiting."
                signal_color = "#fc8181"
                bg_color     = "#2e1a1a"
            else:
                signal       = "Near Average — Stable market."
                signal_color = "#f6e05e"
                bg_color     = "#2e2a1a"

            st.markdown(f"""
            <div style="background: {bg_color};
                        border-radius: 10px;
                        padding: 15px;
                        text-align: center;
                        margin-bottom: 15px;">
                <p style="color: {signal_color};
                           font-size: 16px;
                           margin: 0;
                           font-weight: bold;">
                    {signal}
                </p>
                <p style="color: #a0aec0;
                           font-size: 12px;
                           margin: 5px 0 0 0;">
                    {diff_pct:+.1f}% vs historical average
                    (Rs {avg_price:,.0f})
                </p>
            </div>
            """, unsafe_allow_html=True)

            # All model RMSE comparison
            st.markdown(f"""
            <div style="background: #1a202c;
                        border-radius: 10px;
                        padding: 12px;
                        margin-bottom: 15px;">
                <p style="color: #a0aec0;
                           font-size: 12px;
                           margin: 0 0 8px 0;">
                    Model Performance for {selected_commodity}
                </p>
                {''.join([
                    f'<p style="color: '
                    f'{"#68d391" if m == best_model_name else "#a0aec0"}'
                    f'; font-size: 13px; margin: 3px 0;">'
                    f'{"★ " if m == best_model_name else "  "}'
                    f'{m} — RMSE: Rs {stats["all_rmse"][m]:,.0f}'
                    f'{"  (Best)" if m == best_model_name else ""}'
                    f'{"  (Selected)" if m == selected_model and m != best_model_name else ""}'
                    f'</p>'
                    for m in model_names
                ])}
            </div>
            """, unsafe_allow_html=True)

            # Input summary
            st.markdown("**Input Summary**")
            summary_table = pd.DataFrame({
                'Feature': [
                    'Commodity', 'District', 'Year',
                    'Month', 'Model Used', 'Best Model'
                ],
                'Value': [
                    selected_commodity,
                    selected_district,
                    pred_year,
                    month_map[pred_month_num],
                    selected_model,
                    best_model_name
                ]
            })
            st.dataframe(
                summary_table,
                use_container_width=True,
                hide_index=True
            )

        else:
            st.info(
                "Select inputs on the left "
                "and click Predict Price."
            )