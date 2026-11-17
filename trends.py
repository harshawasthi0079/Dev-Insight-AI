import plotly.express as px
import streamlit as st

from common import load_context, page_header, render_overview_charts, styled_fig

ctx = load_context()
issue_df = ctx["issue_df"]
page_header(ctx, "TRENDS")
render_overview_charts(ctx)

st.divider()
st.subheader("Created vs. closed issue trends")
if not issue_df.empty:
    created_monthly = issue_df.assign(Month=issue_df["Created"].dt.strftime("%Y-%m")).groupby("Month").size().rename("Created").to_frame()
    closed_rows = issue_df.dropna(subset=["Closed"]).copy()
    if not closed_rows.empty:
        closed_monthly = closed_rows.assign(Month=closed_rows["Closed"].dt.strftime("%Y-%m")).groupby("Month").size().rename("Closed")
        trend_compare = created_monthly.join(closed_monthly, how="outer").fillna(0).reset_index().sort_values("Month")
    else:
        trend_compare = created_monthly.assign(Closed=0).reset_index().sort_values("Month")
    trend_long = trend_compare.melt(id_vars="Month", value_vars=["Created", "Closed"], var_name="Activity", value_name="Issues")
    fig = px.line(trend_long, x="Month", y="Issues", color="Activity", markers=True, title="Monthly issues created vs. closed", color_discrete_map={"Created": "#5eead4", "Closed": "#7c9cff"})
    st.plotly_chart(styled_fig(fig), use_container_width=True)
    st.caption("Closed counts use closed_at dates from the fetched sample; results are limited to up to 100 recent regular issues.")
else:
    st.info("Trend analysis ke liye issue data available nahi hai.")
