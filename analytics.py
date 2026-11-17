import plotly.express as px
import streamlit as st

from common import load_context, page_header, render_overview_charts, render_priority_table, styled_fig

ctx = load_context()
issue_df, open_df = ctx["issue_df"], ctx["open_df"]
page_header(ctx, "ANALYTICS")
render_overview_charts(ctx)

st.divider()
st.subheader("Analytics deep dive")
a_col, b_col = st.columns(2)
with a_col:
    if not issue_df.empty:
        status_summary = issue_df["Status"].value_counts().rename_axis("Status").reset_index(name="Issues")
        fig = px.bar(status_summary, x="Status", y="Issues", color="Status", title="Fetched issue status distribution", color_discrete_map={"open": "#5eead4", "closed": "#7c9cff"})
        st.plotly_chart(styled_fig(fig), use_container_width=True)
with b_col:
    if not open_df.empty:
        label_summary = open_df["Labels"].replace("", "Unlabelled").str.split(", ").explode().value_counts().head(10).rename_axis("Label").reset_index(name="Open issues")
        fig = px.bar(label_summary.sort_values("Open issues"), x="Open issues", y="Label", orientation="h", title="Top labels on open issues", color_discrete_sequence=["#5eead4"])
        st.plotly_chart(styled_fig(fig), use_container_width=True)

if not issue_df.empty:
    st.subheader("Issue data explorer")
    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        status_filter = st.selectbox("Filter by status", ["All", "open", "closed"])
    with filter_col2:
        label_options = sorted({label.strip() for labels in issue_df["Labels"] for label in labels.split(",") if label.strip()})
        label_filter = st.selectbox("Filter by label", ["All labels"] + label_options)
    filtered = issue_df.copy()
    if status_filter != "All":
        filtered = filtered[filtered["Status"] == status_filter]
    if label_filter != "All labels":
        filtered = filtered[filtered["Labels"].str.contains(label_filter, regex=False, na=False)]
    st.dataframe(filtered.sort_values("Created", ascending=False), use_container_width=True, hide_index=True, column_config={"URL": st.column_config.LinkColumn("Issue link")})

render_priority_table(ctx)
