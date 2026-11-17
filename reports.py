import pandas as pd
import streamlit as st

from common import compute_health, load_context

ctx = load_context(metrics=True, model=True)
data, analysis, issue_df = ctx["data"], ctx["analysis"], ctx["issue_df"]
features, result = ctx["features"], ctx["result"]

st.subheader("Download Analysis Report")
if not issue_df.empty:
    export_df = issue_df.copy()
    for col in ("Created", "Updated"):
        export_df[col] = export_df[col].astype(str)
    st.download_button("⬇️ Download Issues Report (CSV)", export_df.to_csv(index=False).encode("utf-8"), "devinsight_issues_report.csv", "text/csv", use_container_width=True)
    summary = {
        "repository": data.get("full_name", analysis["repo"]),
        "stars": data.get("stargazers_count", 0),
        "forks": data.get("forks_count", 0),
        "github_open_issues_count": data.get("open_issues_count", 0),
        "contributors": ctx["contributor_count"],
        "pull_requests_all_time_estimate": ctx["pull_request_count"],
        "commits_last_30_days_sample_max_100": ctx["recent_commit_count"],
        "fetched_regular_issues": len(issue_df),
        "sample_open_issues": ctx["open_count"],
        "sample_closed_issues": ctx["closed_count"],
        "predicted_risk": result.get("risk") if result else None,
        "model_confidence_percent": result.get("confidence") if result else None,
        "project_health_score_heuristic": round(compute_health(features), 1) if features else None,
        "sample_limit": 100,
    }
    st.download_button("⬇️ Download Repository Summary (JSON)", pd.Series(summary).to_json(indent=2).encode("utf-8"), "devinsight_repository_summary.json", "application/json", use_container_width=True)
    st.dataframe(export_df.sort_values("Created", ascending=False), use_container_width=True, hide_index=True, column_config={"URL": st.column_config.LinkColumn("Issue Link")})
else:
    st.info("No issue data available to export.")
