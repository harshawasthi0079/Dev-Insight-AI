import streamlit as st

from common import load_context

ctx = load_context()
data, analysis = ctx["data"], ctx["analysis"]
st.subheader("Repository Details")
st.write("**Name:**", data.get("full_name", analysis["repo"]))
st.write("**Description:**", data.get("description") or "No description")
st.write("**Language:**", data.get("language") or "Not specified")
st.write("**Created:**", str(data.get("created_at", ""))[:10])
st.write("**Default branch:**", data.get("default_branch", "Unknown"))
st.link_button("Open repository on GitHub ↗️", data.get("html_url", f"https://github.com/{analysis['repo']}"))
