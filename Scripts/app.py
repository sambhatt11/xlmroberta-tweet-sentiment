import streamlit as st
import pandas as pd
import plotly.express as px

# ✅ Set config at the very top
st.set_page_config(page_title="Tweet Sentiment Dashboard", layout="wide", page_icon="📊")

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv("../sentiment_analysis/replies_with_analysis.csv")
    df['tweet_id'] = pd.to_numeric(df['tweet_id'], errors='coerce').astype('Int64')
    df['reply_date'] = pd.to_datetime(df['reply_date']).dt.tz_localize(None)
    return df

# --- Sidebar Filters ---
def sidebar_filters(df):
    st.sidebar.header("🔍 Filter Replies")

    tweet_ids = sorted(df['tweet_id'].dropna().unique())
    selected_tweet_id = st.sidebar.selectbox("Select Tweet ID", ["All"] + [str(t) for t in tweet_ids])

    min_date = df['reply_date'].min().date()
    max_date = df['reply_date'].max().date()
    date_range = st.sidebar.slider("Date Range", min_value=min_date, max_value=max_date, value=(min_date, max_date))

    sentiments = ["All"] + sorted(df['coarse_sentiment'].dropna().unique())
    selected_sentiment = st.sidebar.selectbox("Coarse Sentiment", sentiments)

    fine_sentiments = ["All"] + sorted(df['fine_sentiment'].dropna().unique())
    selected_fine_sentiment = st.sidebar.selectbox("Fine Sentiment", fine_sentiments)

    relevance_labels = ["All"] + sorted(df['relevance_label'].dropna().unique())
    selected_relevance = st.sidebar.selectbox("Relevance Label", relevance_labels)

    filtered_df = df.copy()
    if selected_tweet_id != "All":
        filtered_df = filtered_df[filtered_df['tweet_id'] == int(selected_tweet_id)]
    if selected_sentiment != "All":
        filtered_df = filtered_df[filtered_df['coarse_sentiment'] == selected_sentiment]
    if selected_fine_sentiment != "All":
        filtered_df = filtered_df[filtered_df['fine_sentiment'] == selected_fine_sentiment]
    if selected_relevance != "All":
        filtered_df = filtered_df[filtered_df['relevance_label'] == selected_relevance]
    if date_range:
        start_date, end_date = date_range
        filtered_df = filtered_df[
            (filtered_df['reply_date'] >= pd.to_datetime(start_date)) &
            (filtered_df['reply_date'] <= pd.to_datetime(end_date))
        ]

    return filtered_df, selected_tweet_id

# --- Main Dashboard ---
def main():
    st.title("✨ Tweet Sentiment Analysis Dashboard")

    df = load_data()
    filtered_df, selected_tweet_id = sidebar_filters(df)

    # --- Show original tweet if a specific tweet is selected ---
    if selected_tweet_id != "All":
        original_row = df[df['tweet_id'] == int(selected_tweet_id)][['tweet_id', 'tweet_text']].dropna().head(1)
        if not original_row.empty:
            st.markdown("#### 📝 Original Tweet")
            st.info(original_row.iloc[0]['tweet_text'])

    # --- Color Palettes ---
    coarse_colors = {"Negative": "#E74C3C", "Positive": "#27AE60", "Neutral": "#3498DB"}
    negative_shades_light = ['#f1948a', '#ec7063', '#e74c3c', '#cb4335', '#b03a2e']
    fine_shades_blue = px.colors.sequential.Blues
    neutral_shades_dark = px.colors.sequential.Blues_r

    st.markdown("---")

    # --- Metrics ---
    total = len(filtered_df)
    negative = len(filtered_df[filtered_df['coarse_sentiment'].str.lower() == 'negative'])
    positive = len(filtered_df[filtered_df['coarse_sentiment'].str.lower() == 'positive'])
    neutral = len(filtered_df[filtered_df['coarse_sentiment'].str.lower() == 'neutral'])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Replies", total)
    col2.metric("Negative Replies", negative, f"{negative/total:.1%}" if total else "0%")
    col3.metric("Positive Replies", positive, f"{positive/total:.1%}" if total else "0%")
    col4.metric("Neutral Replies", neutral, f"{neutral/total:.1%}" if total else "0%")

    st.markdown("---")

    # --- Tabs ---
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Visualizations", "📈 Trends", "🚫 Negative Analysis", "📄 Data Preview"])

    with tab1:
        col1, col2 = st.columns(2)

        negative_df = filtered_df[filtered_df['coarse_sentiment'].str.lower() == 'negative']
        neg_fine_counts = negative_df['fine_sentiment'].value_counts().reset_index()
        neg_fine_counts.columns = ['Fine Sentiment', 'Count']

        fig_neg_fine = px.bar(
            neg_fine_counts,
            x='Fine Sentiment',
            y='Count',
            title="Negative Fine Sentiment Breakdown",
            text='Count',
            template="plotly_white",
            color='Fine Sentiment',
            color_discrete_sequence=negative_shades_light
        )
        fig_neg_fine.update_traces(marker_line_width=0.5)
        col1.plotly_chart(fig_neg_fine, use_container_width=True)

        relevance_counts = filtered_df['relevance_label'].value_counts().reset_index()
        relevance_counts.columns = ['Relevance Label', 'Count']

        fig_relevance = px.pie(
            relevance_counts,
            names='Relevance Label',
            values='Count',
            title="Reply Types Distribution",
            color_discrete_sequence=neutral_shades_dark
        )
        col2.plotly_chart(fig_relevance, use_container_width=True)

        stacked_df = (
            filtered_df.groupby(['coarse_sentiment', 'fine_sentiment'])
            .size()
            .reset_index(name='Count')
        )
        fig_stacked = px.bar(
            stacked_df,
            x='coarse_sentiment',
            y='Count',
            color='fine_sentiment',
            title="Coarse vs Fine Sentiment Breakdown",
            template="plotly_white",
            color_discrete_sequence=fine_shades_blue
        )
        fig_stacked.update_traces(width=0.5, marker_line_width=0.2)
        fig_stacked.update_layout(
            xaxis_title="Coarse Sentiment",
            yaxis_title="Count",
            legend_title="Fine Sentiment",
            barmode='stack'
        )
        st.plotly_chart(fig_stacked, use_container_width=True)

    with tab2:
        trend = filtered_df.groupby(
            [pd.Grouper(key='reply_date', freq='D'), 'coarse_sentiment']
        ).size().reset_index(name='Count')

        fig_trend = px.area(
            trend,
            x='reply_date',
            y='Count',
            color='coarse_sentiment',
            title="Sentiment Trend Over Time",
            color_discrete_map=coarse_colors,
            template="plotly_white"
        )
        fig_trend.update_layout(xaxis_title="Date", yaxis_title="Number of Replies")
        st.plotly_chart(fig_trend, use_container_width=True)

    with tab3:
        st.subheader("🚫 Detailed Negative Sentiment Analysis")
        col1, col2 = st.columns(2)

        neg_relevance = negative_df['relevance_label'].value_counts().reset_index()
        neg_relevance.columns = ['Relevance Label', 'Count']

        fig_neg_rel = px.pie(
            neg_relevance,
            names='Relevance Label',
            values='Count',
            title="Negative Replies by Relevance",
            color_discrete_sequence=negative_shades_light
        )
        col1.plotly_chart(fig_neg_rel, use_container_width=True)

        neg_fine_pie = negative_df['fine_sentiment'].value_counts().reset_index()
        neg_fine_pie.columns = ['Fine Sentiment', 'Count']

        fig_neg_fine_pie = px.pie(
            neg_fine_pie,
            names='Fine Sentiment',
            values='Count',
            title="Negative Fine Sentiment Distribution",
            hole=0.4,
            color_discrete_sequence=negative_shades_light
        )
        col2.plotly_chart(fig_neg_fine_pie, use_container_width=True)

        st.dataframe(
            negative_df[['tweet_id', 'reply_date', 'reply_username', 'analyzed_text',
                         'fine_sentiment',  'relevance_label']],
            use_container_width=True
        )

    with tab4:
        st.dataframe(
            filtered_df[['tweet_id', 'reply_date', 'reply_username', 'analyzed_text',
                         'coarse_sentiment', 'fine_sentiment', 'relevance_label']],
            use_container_width=True
        )

    st.info("💡 Tip: Use the sidebar to focus your filters for sharper, clearer sentiment analysis.")

# --- Run ---
if __name__ == "__main__":
    main()
