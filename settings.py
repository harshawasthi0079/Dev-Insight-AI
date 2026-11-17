import streamlit as st

from common import get_github_headers, load_context

ctx = load_context()
st.subheader("Settings")
st.write("**Current repository:**", ctx["analysis"]["repo"])
st.write("**Cache:** GitHub API responses cached for 10 minutes.")
st.write("**Authentication:** Optional `GITHUB_TOKEN` environment variable or Streamlit secret.")
st.write("**Contributor data:** GitHub contributors endpoint; cached for 30 minutes.")
st.write("**Engineering activity:** Pull request count estimate and up to 100 commits from the last 30 days.")
st.write("**Issue sample:** Up to 100 recent regular issues; pull requests are excluded.")
st.write("**Project Health Score:** Transparent heuristic indicator, separate from the ML prediction.")
st.write("**Current API token:**", "Configured" if get_github_headers().get("Authorization") else "Not configured (public API rate limits apply)")
if st.button("Clear cached GitHub data"):
    st.cache_data.clear()
    st.success("Cache cleared. Re-submit Analyze to fetch fresh data.")
