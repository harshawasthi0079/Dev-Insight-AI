import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import os

from ml_features import build_repository_features
from ml_model import predict_repository_risk


st.set_page_config(
    page_title="DevInsight",
    page_icon="📊",
    layout="wide",
)


def get_github_headers():
    """Build GitHub API headers; works with or without Streamlit secrets."""
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    try:
        token = st.secrets.get("GITHUB_TOKEN", "")
    except Exception:
        token = ""
    token = token or os.getenv("GITHUB_TOKEN", "")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


@st.cache_data(ttl=600, show_spinner=False)
def get_repo_issues(owner, repo):
    """Fetch up to 100 regular GitHub issues, excluding pull requests."""
    url = f"https://api.github.com/repos/{owner}/{repo}/issues"
    all_issues = []
    next_url = url
    params = {
        "state": "all",
        "per_page": 100,
        "sort": "created",
        "direction": "desc",
    }

    headers = get_github_headers()

    while next_url and len(all_issues) < 100:
        try:
            response = requests.get(
                next_url,
                params=params,
                headers=headers,
                timeout=(10, 30),
            )
            response.raise_for_status()
            issues = response.json()
        except (requests.RequestException, ValueError) as error:
            st.warning(
                f"Issues fetch error: {error}. "
                "Fetched data incomplete ho sakta hai."
            )
            break

        params = None

        if not isinstance(issues, list) or not issues:
            break

        regular_issues = [
            issue for issue in issues
            if isinstance(issue, dict)
            and "pull_request" not in issue
        ]

        remaining = 100 - len(all_issues)
        all_issues.extend(regular_issues[:remaining])
        next_url = response.links.get("next", {}).get("url")

    return all_issues


st.title("🚀 DevInsight")
st.subheader("AI-Powered Software Development Analytics")
st.write("Analyze real GitHub repository activity.")

repo_name = st.text_input(
    "GitHub repository (owner/repo)",
    "streamlit/streamlit",
).strip()

if st.button("Analyze Repository", type="primary"):
    parts = repo_name.split("/")

    if len(parts) != 2 or not all(parts):
        st.error("Repository format owner/repo hona chahiye.")
        st.stop()

    owner, repo = parts

    headers = get_github_headers()

    try:
        response = requests.get(
            f"https://api.github.com/repos/{owner}/{repo}",
            headers=headers,
            timeout=(10, 30),
        )

        if response.status_code == 404:
            st.error("Repository nahi mili. owner/repo check karo.")
            st.stop()

        response.raise_for_status()
        data = response.json()

    except (requests.RequestException, ValueError) as error:
        st.error(f"GitHub repository load nahi ho paayi: {error}")
        st.stop()

    # Repository overview
    st.success("Live GitHub repository data loaded successfully!")

    col1, col2, col3 = st.columns(3)
    col1.metric("⭐ Stars", data.get("stargazers_count", 0))
    col2.metric("🐛 Open Issues (GitHub)", data.get("open_issues_count", 0))
    col3.metric("🍴 Forks", data.get("forks_count", 0))

    st.subheader("Repository Details")
    st.write("**Name:**", data.get("full_name", repo_name))
    st.write("**Description:**", data.get("description") or "No description")
    st.write("**Language:**", data.get("language") or "Not specified")
    st.write("**Created:**", data.get("created_at", "")[:10])

    # Fetch issue data
    st.subheader("Issue Analytics")
    issues = get_repo_issues(owner, repo)

    if not issues:
        st.info("Is repository se koi regular issue data nahi mila.")
        st.stop()

    st.caption(f"Regular issues fetched: {len(issues)} (maximum 100)")

    try:
        features = build_repository_features(issues)
    except (ValueError, KeyError, TypeError) as error:
        st.error(f"Repository features calculate nahi ho paaye: {error}")
        st.stop()

    # ML prediction
    st.subheader("🤖 AI Repository Health Prediction")

    try:
        result = predict_repository_risk(features)
        risk = str(result["risk"]).strip()
        risk_normalized = risk.lower()

        pred_col, acc_col, confidence_col = st.columns(3)
        pred_col.metric("Predicted Risk", risk)
        acc_col.metric(
            "Model Test Accuracy",
            f'{result["model_accuracy"]:.2f}%',
        )
        confidence_col.metric(
            "Model Confidence",
            f'{result["confidence"]:.2f}%',
        )

        st.caption(
            "Prototype prediction: model labels rule-generated hain. "
            "Test accuracy real-world project failure prediction "
            "ki accuracy nahi hai."
        )

        if risk_normalized in ("high", "high risk"):
            st.warning(
                "Backlog indicators high hain. Repository ko review karein."
            )
        elif risk_normalized in ("medium", "medium risk"):
            st.info(
                "Backlog indicators moderate hain. Repository monitor karein."
            )
        else:
            st.success(
                "Model ne available categories ke hisaab se low risk diya hai."
            )

        # Calculate explanation values
        old_ratio = float(
            features.get("old_open_issue_ratio", 0) or 0
        )
        backlog_change = float(
            features.get("backlog_change_30d", 0) or 0
        )
        open_issue_count = int(
            features.get("open_issue_count", 0) or 0
        )

        # Why this risk?
        st.subheader("🔎 Why this risk?")

        reasons = []

        if old_ratio >= 0.50:
            reasons.append(
                f"{old_ratio:.0%} of open issues are old. "
                "Review long-pending issues."
            )
        elif old_ratio >= 0.25:
            reasons.append(
                f"{old_ratio:.0%} of open issues are old. "
                "Monitor unresolved issues."
            )

        if backlog_change >= 10:
            reasons.append(
                f"Backlog increased by {backlog_change:.0f} "
                "in the available 30-day issue sample."
            )
        elif backlog_change >= 3:
            reasons.append(
                f"Backlog increased by {backlog_change:.0f} "
                "in the available 30-day issue sample."
            )
        elif backlog_change < 0:
            reasons.append(
                "Backlog decreased in the available 30-day issue sample."
            )

        if open_issue_count >= 10:
            reasons.append(
                f"There are {open_issue_count} open issues in the sample."
            )

        if not reasons:
            reasons.append(
                "No major warning was triggered by these selected indicators. "
                "This does not guarantee that the project is healthy."
            )

        for reason in reasons:
            st.write(f"• {reason}")

        # Recommended actions — displayed for every risk category
        st.subheader("🛠️ Recommended Actions")

        if risk_normalized in ("high", "high risk"):
            st.write("1. Review old unresolved issues.")
            st.write("2. Identify blockers and assign priorities.")
            st.write("3. Compare newly created issues with resolved issues.")
        elif risk_normalized in ("medium", "medium risk"):
            st.write("1. Review pending issues regularly.")
            st.write("2. Track issue creation versus closure.")
            st.write("3. Prioritize important unresolved issues.")
        else:
            st.write("1. Continue monitoring the backlog.")
            st.write("2. Review unresolved issues regularly.")
            st.write("3. Reassess if backlog starts increasing.")
        # Risk Summary Card
        st.subheader("🚦 Project Risk Summary")
        risk_col1, risk_col2, risk_col3 = st.columns(3)
        risk_col1.metric(
            "Predicted Risk",
            result.get("risk", "Unknown").title()
            )
        risk_col2.metric(
            "Model Confidence",
            f"{result.get('confidence', 0):.1f}"
            )
        risk_col3.metric(
            "Open Issues",
            open_issue_count
            )
        if risk_normalized in ("high", "high risk"):
            st.warning(
             "Recommended Action: Review aging issues and identify blockers."
            )
        elif risk_normalized in ("medium", "medium risk"):
            st.info(
             "Recommended Action: Review pending issues and monitor backlog."
            )
        else:
            st.success(
            "Recommended Action: Continue monitoring the issue backlog."
        )
        # Repository Health Summary — displayed for every risk category
        st.subheader("📊 Repository Health Summary")

        health_col1, health_col2, health_col3 = st.columns(3)

        health_col1.metric("Open Issues (sample)", open_issue_count)
        health_col2.metric("Old Open Issue Ratio", f"{old_ratio:.1%}")
        health_col3.metric("Backlog Change (30-day sample)", f"{backlog_change:+.0f}")

        st.caption(
            "These indicators use up to 100 fetched issues. "
            "They are not a verified measure of project failure."
        )

        # Model evaluation
        evaluation = result.get("evaluation", {})

        if evaluation:
            st.subheader("📈 ML Model Evaluation")

            eval_col1, eval_col2, eval_col3 = st.columns(3)
            eval_col1.metric(
                "Macro Precision",
                f'{evaluation["precision_macro"]:.2%}',
            )
            eval_col2.metric(
                "Macro Recall",
                f'{evaluation["recall_macro"]:.2%}',
            )
            eval_col3.metric(
                "Macro F1-score",
                f'{evaluation["f1_macro"]:.2%}',
            )

            labels = evaluation["labels"]
            matrix_df = pd.DataFrame(
                evaluation["confusion_matrix"],
                index=[f"Actual {label}" for label in labels],
                columns=[f"Predicted {label}" for label in labels],
            )

            st.write("**Confusion Matrix**")
            st.dataframe(matrix_df, use_container_width=True)

            st.caption(
                f'Test samples: {evaluation["test_rows"]} | '
                f'Training samples: {evaluation["train_rows"]}'
            )

    except (ValueError, KeyError, TypeError) as error:
        st.error(f"Prediction nahi ho paayi: {error}")

    # Feature metrics
    st.subheader("📊 ML Repository Features")

    col1, col2, col3 = st.columns(3)
    col1.metric("Open Issues (sample)", features["open_issue_count"])
    col2.metric("Closed Issues (sample)", features["closed_issue_count"])
    col3.metric(
        "Old Open Issue Ratio",
        f'{features["old_open_issue_ratio"]:.1%}',
    )

    col4, col5, col6 = st.columns(3)
    col4.metric(
        "Issues Created (30 Days)",
        features["issues_created_last_30d"],
    )
    col5.metric(
        "Issues Closed (30 Days)",
        features["issues_closed_last_30d"],
    )
    col6.metric(
        "Backlog Change (30 Days)",
        features["backlog_change_30d"],
    )

    # Build issue table
    issue_df = pd.DataFrame([
        {
            "Title": issue.get("title", "Untitled"),
            "Status": issue.get("state", "unknown"),
            "Created": issue.get("created_at"),
            "URL": issue.get("html_url", ""),
        }
        for issue in issues
    ])

    issue_df["Created"] = pd.to_datetime(
        issue_df["Created"],
        utc=True,
        errors="coerce",
    )
    issue_df = issue_df.dropna(subset=["Created"])

    open_count = int((issue_df["Status"] == "open").sum())
    closed_count = int((issue_df["Status"] == "closed").sum())

    col1, col2 = st.columns(2)
    col1.metric("Open Issues (fetched)", open_count)
    col2.metric("Closed Issues (fetched)", closed_count)

    # Open vs closed chart
    chart_df = pd.DataFrame({
        "Status": ["Open", "Closed"],
        "Count": [open_count, closed_count],
    })

    fig = px.bar(
        chart_df,
        x="Status",
        y="Count",
        title="Open vs Closed Issues (Fetched Sample)",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Monthly issue trends
    st.subheader("Monthly Issue Trends")

    monthly_df = issue_df.copy()
    monthly_df["Month"] = monthly_df["Created"].dt.strftime("%Y-%m")

    trend_df = (
        monthly_df.groupby("Month")
        .size()
        .reset_index(name="Issues in Fetched Sample")
        .sort_values("Month")
    )

    if not trend_df.empty:
        fig_trend = px.line(
            trend_df,
            x="Month",
            y="Issues in Fetched Sample",
            markers=True,
            title="Issue Creation Month (Fetched Sample)",
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.info("Monthly trend ke liye valid creation dates nahi hain.")

    # Issue aging analysis
    st.subheader("Issue Aging Analysis")

    open_issue_df = issue_df[
        issue_df["Status"] == "open"
    ].copy()

    if not open_issue_df.empty:
        now = pd.Timestamp.now(tz="UTC")
        open_issue_df["Age (Days)"] = (
            now - open_issue_df["Created"]
        ).dt.days

        open_issue_df["Priority"] = open_issue_df["Age (Days)"].apply(
            lambda days: "High Attention"
            if days > 30
            else "Needs Monitoring"
            if days > 7
            else "Recent"
        )

        st.metric(
            "Open Issues Older Than 30 Days",
            int((open_issue_df["Age (Days)"] > 30).sum()),
        )

        st.dataframe(
            open_issue_df[["Title", "Age (Days)", "Priority", "URL"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "URL": st.column_config.LinkColumn("Issue Link"),
            },
        )
    else:
        st.success("Fetched sample mein koi open issue nahi mila.")
    # Priority Issues Dashboard
    st.subheader("🎯 Priority Issues Dashboard")

    priority_df = open_issue_df.copy()

    if not priority_df.empty:
        priority_df["Priority Score"] = (
            priority_df["Age (Days)"].clip(lower=0).clip(upper=90)
        )

        priority_df["Action"] = priority_df["Age (Days)"].apply(
            lambda days: "Review urgently"
            if days > 30
            else "Review soon"
            if days > 7
            else "Monitor"
        )

        priority_df = priority_df.sort_values(
            "Priority Score",
            ascending=False,
        )

        st.caption(
            "Issues are ranked by age only. Age is a review signal, "
            "not proof that an issue is severe."
        )

        st.dataframe(
            priority_df[
                ["Title", "Age (Days)", "Priority", "Action", "URL"]
            ],
            use_container_width=True,
            hide_index=True,
            column_config={
                "URL": st.column_config.LinkColumn("Open Issue"),
                "Priority Score": st.column_config.NumberColumn(
                    "Priority Score",
                    format="%.0f",
                ),
            },
        )

        urgent_count = int(
            (priority_df["Age (Days)"] > 30).sum()
        )
        st.metric("Issues needing review (>30 days)", urgent_count)

    else:
        st.success("No open issues in the fetched sample to prioritize.")
    # Download Analysis Report
    st.subheader("📥 Download Analysis Report")

    report_df = issue_df.copy()

    if not report_df.empty:
        csv_data = report_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇️ Download Issues Report (CSV)",
            data=csv_data,
            file_name="devinsight_issues_report.csv",
            mime="text/csv",
        )
    else:
        st.info("No issue data available to download.")    
    # Older open issues
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=30)
    old_issues_df = issue_df[
        (issue_df["Status"] == "open")
        & (issue_df["Created"] < cutoff)
    ]

    st.subheader("Open Issues Older Than 30 Days")
    st.caption(
        "Analysis maximum 100 fetched regular issues par based hai, "
        "complete repository history par nahi."
    )

    if old_issues_df.empty:
        st.info("Fetched sample mein 30 din se purana open issue nahi mila.")
    else:
        st.warning(
            f"{len(old_issues_df)} fetched open issues 30 din se purane hain."
        )
        st.dataframe(
            old_issues_df[["Title", "Created", "URL"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "URL": st.column_config.LinkColumn("Issue Link"),
            },
        ) 

# GitHub Repository Comparison
st.divider()
st.header("⚖️ Compare Two GitHub Repositories")
compare_repo = st.text_input(
    "Second repository (owner/repo)",
    "pallets/flask",
    key="compare_repo_input",
).strip()
if st.button("Compare Repositories"):
    compare_parts = compare_repo.split("/")
    primary_parts = repo_name.split("/")
    if (
        len(compare_parts) != 2
        or not all(compare_parts)
        or len(primary_parts) != 2
        or not all(primary_parts)
    ):
        st.error("Dono repositories ka format owner/repo hona chahiye.")
    elif compare_repo.lower() == repo_name.lower():
        st.warning("Comparison ke liye koi doosri repository enter karo.")
    else:
        primary_owner, primary_repo = primary_parts
        compare_owner, compare_name = compare_parts
        with st.spinner("Dono repositories analyze ho rahi hain..."):
            try:
                # Re-fetch the first repository here: Streamlit button clicks
                # rerun the script, so variables from the Analyze button may not exist.
                headers = get_github_headers()
                primary_response = requests.get(
                    f"https://api.github.com/repos/{primary_owner}/{primary_repo}",
                    headers=headers,
                    timeout=(10, 30),
                )
                primary_response.raise_for_status()
                primary_data = primary_response.json()
                primary_issues = get_repo_issues(primary_owner, primary_repo)
                if not primary_issues:
                    st.warning("First repository mein regular issues nahi mile.")
                    st.stop()
                primary_features = build_repository_features(primary_issues)
                primary_result = predict_repository_risk(primary_features)

                compare_response = requests.get(
                    f"https://api.github.com/repos/{compare_owner}/{compare_name}",
                    headers=headers,
                    timeout=(10, 30),
                )
                compare_response.raise_for_status()
                compare_data = compare_response.json()
                compare_issues = get_repo_issues(compare_owner, compare_name)
                if not compare_issues:
                    st.warning("Second repository mein regular issues nahi mile.")
                else:
                    compare_features = build_repository_features(compare_issues)
                    compare_result = predict_repository_risk(compare_features)
                    st.subheader("Comparison Results")
                    left, right = st.columns(2)
                    with left:
                        st.markdown(f"**{primary_data.get('full_name', repo_name)}**")
                        st.metric("Predicted Risk", primary_result["risk"])
                        st.metric(
                            "Open Issues (sample)",
                            int(primary_features["open_issue_count"]),
                        )
                        st.metric(
                            "Old Open Issue Ratio",
                            f'{primary_features["old_open_issue_ratio"]:.1%}',
                        )
                        st.metric(
                            "Backlog Change (30 days)",
                            int(primary_features["backlog_change_30d"]),
                        )
                    with right:
                        st.markdown(
                            f"**{compare_data.get('full_name', compare_repo)}**"
                        )
                        st.metric(
                            "Predicted Risk",
                            compare_result["risk"],
                        )
                        st.metric(
                            "Open Issues (sample)",
                            int(compare_features["open_issue_count"]),
                        )
                        st.metric(
                            "Old Open Issue Ratio",
                            f'{compare_features["old_open_issue_ratio"]:.1%}',
                        )
                        st.metric(
                            "Backlog Change (30 days)",
                            int(compare_features["backlog_change_30d"]),
                        )
                    st.caption(
                        "Comparison up to 100 fetched issues per repository "
                        "par based hai. Risk predictions prototype estimates hain."
                    )
            except (requests.RequestException, ValueError, KeyError, TypeError) as error:
                st.error(f"Comparison nahi ho paaya: {error}")