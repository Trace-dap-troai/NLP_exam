# --------------------------------------------------
# BMW AI CEO Agent
# Executive Intelligence Dashboard
# --------------------------------------------------

import os
import json
import pandas as pd
import streamlit as st
from rag_engine import AICEoAgent, CACHE_FILE

st.set_page_config(
    page_title="BMW AI CEO Agent",
    page_icon="🏛️",
    layout="wide"
)

# --------------------------------------------------
# STYLING
# --------------------------------------------------
st.markdown(
    """
    <style>
    div[data-testid="metric-container"] {
        background-color: #f8f9fa;
        border: 1px solid #dcdcdc;
        padding: 15px;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# LOAD DATA 
# --------------------------------------------------
@st.cache_data(ttl=5)
def load_csv():
    if not os.path.exists("bmw_live_data.csv"):
        return pd.DataFrame()
    df = pd.read_csv("bmw_live_data.csv")
    #df["published_date"] = pd.to_datetime(df["published_date"], errors="coerce", utc=True)
    df["published_date"] = pd.to_datetime(df["published_date"], format="mixed", errors="coerce", utc=True)
    return df

def load_report():
    # Priority 1: Directly read previously cached JSON file (Speed ​​0.01 second)
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            st.warning(f"Failed to read cache: {e}. Regenerating...")

    # Priority 2: If there is no JSON file, then call LLM Gemma 3 to calculate
    agent = AICEoAgent()
    return agent.generate_report()

# --------------------------------------------------
# START
# --------------------------------------------------
df = load_csv()

st.image(
    "https://upload.wikimedia.org/wikipedia/commons/4/44/BMW.svg",
    width=80
)
st.title(" BMW Strategic Intelligence Agent")




st.caption("Powered by Gemma 3 (4B) • ChromaDB • SentenceTransformers")

if df.empty:
    st.error("No data found. Run scraper_pipeline.py first.")
    st.stop()

with st.spinner("Loading strategic intelligence..."):
    report = load_report()

if not report:
    st.error("Unable to load strategic report.")
    st.stop()

# --------------------------------------------------
# SECTION 1: OVERVIEW
# --------------------------------------------------
st.header("Section 1: Company Overview")

col1, col2, col3, col4 = st.columns(4)
last_update = (
    df["pipeline_run_time"].iloc[-1]
    if "pipeline_run_time" in df.columns
    else "N/A"
)

col1.metric("Organization", "BMW Group")
col2.metric("Documents", len(df))
col3.metric("Sources", df["source"].nunique())
col4.metric("Last Update", str(last_update))
st.divider()

# --------------------------------------------------
# SECTION 2: MARKET INTEL
# --------------------------------------------------
st.header("Section 2: Market Intelligence")
left, right = st.columns(2)

with left:
    st.subheader("Competitor Activities")
    for item in report.get("market_intelligence", {}).get("competitor_activities", []):
        st.info(item)

with right:
    st.subheader("Emerging Technologies")
    for item in report.get("market_intelligence", {}).get("emerging_technologies", []):
        st.success(item)

    st.subheader("Industry Trends")
    for trend in report.get("trends", []):
        st.markdown(f"**{trend.get('title','Trend')}**")
        st.write(trend.get("description", ""))
st.divider()

# --------------------------------------------------
# SECTION 3 + 4: OPPORTUNITIES & RISKS
# --------------------------------------------------
left, right = st.columns(2)

with left:
    st.header("Section 3: Opportunity Monitor")
    for opp in report.get("opportunities", []):
        with st.container(border=True):
            st.success(opp.get("title", "Opportunity"))
            st.write(f"**Impact:** {opp.get('impact','N/A')}")
            st.write(f"**Confidence:** {opp.get('confidence','N/A')}")
            st.caption(f"Evidence: {opp.get('evidence','N/A')}")

with right:
    st.header("Section 4: Risk Monitor")
    for risk in report.get("risks", []):
        with st.container(border=True):
            st.error(risk.get("title", "Risk"))
            st.write(f"**Category:** {risk.get('category','N/A')} | **Severity:** {risk.get('severity','N/A')}")
            st.write(f"**Confidence:** {risk.get('confidence','N/A')}")
            st.caption(f"Evidence: {risk.get('evidence','N/A')}")
st.divider()

# --------------------------------------------------
# SECTION 5: SENTIMENT
# --------------------------------------------------
st.header("Section 5: Sentiment Analysis")

timeline_df = df.dropna(subset=["published_date"]).copy()
if not timeline_df.empty:
    cutoff_date = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=90)

    timeline_df = timeline_df[
        timeline_df["published_date"] >= cutoff_date
    ]

    timeline_df["date"] = timeline_df["published_date"].dt.date

    sentiment_trend = (
        timeline_df
        .groupby(["date", "sentiment_label"])
        .size()
        .unstack(fill_value=0)
    )

    chart_left, chart_right = st.columns([2, 1])
    with chart_left:
        st.subheader("Sentiment Trend over Time")
        st.line_chart(sentiment_trend)
    with chart_right:
        st.subheader("Overall Distribution")
        st.bar_chart(df["sentiment_label"].value_counts())
else:
    st.info("No timestamp information available to plot timeline.")
st.divider()

# --------------------------------------------------
# SECTION 6: RECOMMENDATIONS
# --------------------------------------------------
st.header("Section 6: Strategic Recommendations")

for rec in report.get("recommendations", []):
    with st.expander(rec.get("action", "Recommendation"), expanded=True):
        st.write(f"**Priority:** {rec.get('priority','N/A')}")
        st.write(f"**Supporting Evidence:** {rec.get('supporting_evidence','N/A')}")
        st.write(f"**Expected Impact:** {rec.get('expected_impact','N/A')}")
        st.write(f"**Risk Assessment:** {rec.get('risk_assessment','N/A')}")
st.divider()

# --------------------------------------------------
# SECTION 7: CEO BRIEFING
# --------------------------------------------------

st.header("Section 7: CEO Briefing")

briefing = report.get("ceo_briefing", {})

next_steps = briefing.get('what_next', 'N/A')
if isinstance(next_steps, list):
    next_steps_formatted = "\n".join([f"- {item}" for item in next_steps])
else:
    next_steps_formatted = str(next_steps)

st.warning(
f"""
### What happened?
{briefing.get('what_happened','N/A')}

### Why does it matter?
{briefing.get('why_it_matters','N/A')}

### What should management do next?
{next_steps_formatted}

"""
)
#{briefing.get('what_next','N/A')}
# --------------------------------------------------
# INTERACTIVE AI AGENT
# --------------------------------------------------
st.divider()

with st.expander(" Ask the BMW AI Agent"):

    user_question = st.text_input(
        "Ask a strategic question about BMW, competitors, EV trends, or risks:"
    )

    if user_question:

        with st.spinner("Analyzing..."):

            agent = AICEoAgent()
            answer = agent.ask_question(user_question)

        st.write(answer)



# --------------------------------------------------
# SOURCE REFERENCES
# --------------------------------------------------
st.divider()
st.header("Retrieved Evidence Sources")

sources = report.get("retrieved_sources", [])
for src in sources[:10]:
    with st.container(border=True):
        st.write(f"**Source:** {src.get('source', 'Unknown')}")

        url = src.get("url", "")
        if url and url != "N/A":
            st.link_button("Open Article", url)