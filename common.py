"""Shared helpers for every DevInsight page: theme, GitHub API, data loading and reusable sections."""
import os
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qs, urlparse

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from ml_features import build_repository_features
from ml_model import predict_repository_risk

TAGLINE = "AI-Powered Software Development Analytics & Project Health Prediction Platform"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root { --bg:#0b1020; --panel:#111a2e; --panel2:#151f35; --line:#25324a; --muted:#91a0b8; --text:#edf3ff; --cyan:#5eead4; --blue:#7c9cff; }
.stApp { background: radial-gradient(ellipse at 75% 0%, #17284a 0%, #0b1020 48%); color:var(--text); font-family:'DM Sans',sans-serif; }
[data-testid="stHeader"] { background:rgba(11,16,32,.75); }
.block-container { padding-top:1.5rem; padding-bottom:2rem; max-width:1600px; }
[data-testid="stSidebar"] { background:#0d1527; border-right:1px solid #25324a; }
[data-testid="stSidebar"] * { color:#dbe6fb; }
h1,h2,h3 { font-family:'Space Grotesk',sans-serif !important; letter-spacing:-.03em; }
h1 { font-size:2.15rem !important; }
[data-testid="stMetric"] { background:linear-gradient(145deg,#151f35,#10182a); border:1px solid #293752; border-radius:16px; padding:18px 20px; min-height:125px; }
[data-testid="stMetricLabel"] { color:#9aabc5 !important; font-size:.88rem !important; }
[data-testid="stMetricValue"] { color:#f2f6ff !important; font-family:'Space Grotesk',sans-serif; }
[data-testid="stPlotlyChart"] { background:#111a2e; border:1px solid #25324a; border-radius:16px; padding:10px; }
.stButton > button, .stDownloadButton > button { border-radius:10px; border:1px solid #415a86; background:linear-gradient(100deg,#4776e6,#6a5cff); color:white; font-weight:700; min-height:2.6rem; }
.stButton > button:hover, .stDownloadButton > button:hover { border-color:#8ba6ff; color:white; }
[data-testid="stTextInput"] input { background:#111a2e; color:#f3f6ff; border:1px solid #34425e; border-radius:10px; }
[data-testid="stDataFrame"] { border:1px solid #293752; border-radius:12px; overflow:hidden; }
hr { border-color:#25324a; }
.hero { background:linear-gradient(115deg,rgba(38,64,112,.92),rgba(17,26,46,.96) 62%,rgba(20,65,77,.78)); border:1px solid #344968; border-radius:20px; padding:24px 28px; margin-bottom:20px; }
.hero-title { font-family:'Space Grotesk',sans-serif; font-size:1.8rem; font-weight:700; color:#f4f7ff; margin:0; }
.hero-sub { color:#b0bfd8; margin-top:6px; font-size:.96rem; }
.ai-badge { display:inline-block; background:rgba(94,234,212,.12); color:#70f0d9; border:1px solid rgba(94,234,212,.35); padding:5px 10px; border-radius:20px; font-size:.76rem; font-weight:700; }
.section-label { color:#91a0b8; text-transform:uppercase; letter-spacing:.12em; font-size:.72rem; font-weight:700; margin-bottom:7px; }
.panel { background:linear-gradient(145deg,#121d32,#10182a); border:1px solid #273650; border-radius:16px; padding:18px 20px; }
.insight { background:linear-gradient(135deg,rgba(34,74,94,.42),rgba(22,31,54,.88)); border:1px solid #2e6572; border-radius:16px; padding:18px 20px; }
.small-muted { color:#91a0b8; font-size:.84rem; }
footer { visibility:hidden; }
@media (max-width: 760px) { .block-container { padding-left:1rem; padding-right:1rem; } .hero { padding:18px; } .hero-title { font-size:1.45rem; } }
</style>
"""


# ------------------------------ Page setup ----------------------------------
def setup():
    """Must be the first Streamlit call in app.py."""
    st.set_page_config(page_title=f"DevInsight | {TAGLINE}", page_icon="📊", layout="wide", initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)


def render_sidebar():
    with st.sidebar:
        st.divider()
        st.markdown("<div class='small-muted'>CONNECTED SOURCE</div>", unsafe_allow_html=True)
        st.markdown("**GitHub REST API**")
        st.caption("Live public repository data")
        st.divider()
        st.caption("DevInsight · Project Health Prediction Platform")


def render_footer():
    st.divider()
    st.markdown(f"<div style='display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap;color:#8191ad;font-size:.8rem'><span>©️ DevInsight · {TAGLINE}</span><span>Live GitHub data · Up to 100 recent regular issues</span></div>", unsafe_allow_html=True)


def render_hero_and_form():
    st.markdown(f"<div class='hero'><div style='display:flex;justify-content:space-between;gap:12px;align-items:flex-start;flex-wrap:wrap'><div><div class='hero-title'>DevInsight</div><div class='hero-sub'>AI-Powered Software Development Analytics &amp; Project Health Prediction Platform</div><div class='hero-sub'>Understand repository health, issue backlog and engineering activity at a glance.</div></div><span class='ai-badge'>✦ POWERED BY AI</span></div></div>", unsafe_allow_html=True)
    with st.form("analyze_form"):
        input_col, button_col = st.columns([5, 1.2], vertical_alignment="bottom")
        with input_col:
            repo_input = st.text_input("GitHub repository", value=st.session_state.get("repo_input", "facebook/react"), placeholder="owner/repo", help="Example: facebook/react")
        with button_col:
            submitted = st.form_submit_button("⚡ Analyze", type="primary", use_container_width=True)
    if submitted:
        st.session_state["repo_input"] = repo_input.strip()
        parts = repo_input.strip().split("/")
        if len(parts) != 2 or not all(parts):
            st.error("Repository format owner/repo hona chahiye, jaise facebook/react.")
        else:
            try:
                with st.spinner("GitHub se repository aur issue data load ho raha hai..."):
                    data = get_repo_data(parts[0], parts[1])
                    issues = get_repo_issues(parts[0], parts[1])
                st.session_state["analysis"] = {"repo": repo_input.strip(), "data": data, "issues": issues}
                st.success(f"{data.get('full_name', repo_input)} successfully analyzed.")
            except requests.HTTPError as e:
                status = e.response.status_code if e.response is not None else "unknown"
                st.error("Repository nahi mili (404)." if status == 404 else f"GitHub API error: HTTP {status}. Token/rate limit check karein.")
            except (requests.RequestException, ValueError) as e:
                st.error(f"GitHub data load nahi ho paaya: {e}")


# ------------------------------ GitHub API ----------------------------------
def get_github_headers():
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    try:
        token = st.secrets.get("GITHUB_TOKEN", "")
    except Exception:
        token = ""
    token = token or os.getenv("GITHUB_TOKEN", "")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _last_page_number(response, payload):
    """Total count from GitHub pagination metadata when per_page=1."""
    last_url = response.links.get("last", {}).get("url")
    if last_url:
        pages = parse_qs(urlparse(last_url).query).get("page", [])
        if pages:
            try:
                return int(pages[0])
            except (TypeError, ValueError):
                pass
    return len(payload)


@st.cache_data(ttl=600, show_spinner=False)
def get_repo_data(owner, repo):
    response = requests.get(f"https://api.github.com/repos/{owner}/{repo}", headers=get_github_headers(), timeout=(10, 30))
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=1800, show_spinner=False)
def get_contributor_count(owner, repo):
    """Contributor count from pagination metadata. None while GitHub is computing (202) or no data (204)."""
    response = requests.get(
        f"https://api.github.com/repos/{owner}/{repo}/contributors",
        params={"per_page": 1, "anon": "true"}, headers=get_github_headers(), timeout=(10, 30),
    )
    if response.status_code in (202, 204):
        return None
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        return None
    return _last_page_number(response, payload)


@st.cache_data(ttl=600, show_spinner=False)
def get_repo_issues(owner, repo):
    url = f"https://api.github.com/repos/{owner}/{repo}/issues"
    collected, next_url = [], url
    params = {"state": "all", "per_page": 100, "sort": "created", "direction": "desc"}
    headers = get_github_headers()
    while next_url and len(collected) < 100:
        response = requests.get(next_url, params=params, headers=headers, timeout=(10, 30))
        response.raise_for_status()
        payload = response.json()
        params = None
        if not isinstance(payload, list) or not payload:
            break
        regular = [x for x in payload if isinstance(x, dict) and "pull_request" not in x]
        collected.extend(regular[: max(0, 100 - len(collected))])
        next_url = response.links.get("next", {}).get("url")
    return collected


@st.cache_data(ttl=600, show_spinner=False)
def get_pull_request_count(owner, repo):
    """Estimate total pull requests using GitHub pagination metadata."""
    response = requests.get(
        f"https://api.github.com/repos/{owner}/{repo}/pulls",
        params={"state": "all", "per_page": 1}, headers=get_github_headers(), timeout=(10, 30),
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        return None
    return _last_page_number(response, payload)


@st.cache_data(ttl=600, show_spinner=False)
def get_recent_commit_count(owner, repo, days=30):
    """Count commits for the last N days, capped at 100 for responsiveness."""
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    response = requests.get(
        f"https://api.github.com/repos/{owner}/{repo}/commits",
        params={"since": since, "per_page": 100}, headers=get_github_headers(), timeout=(10, 30),
    )
    response.raise_for_status()
    payload = response.json()
    return len(payload) if isinstance(payload, list) else None


# ------------------------------ Data helpers --------------------------------
def make_issue_df(issues):
    df = pd.DataFrame([{
        "Title": i.get("title", "Untitled"), "Status": i.get("state", "unknown"),
        "Created": i.get("created_at"), "Updated": i.get("updated_at"),
        "Closed": i.get("closed_at"),
        "URL": i.get("html_url", ""), "Labels": ", ".join(x.get("name", "") for x in i.get("labels", [])),
        "Comments": i.get("comments", 0),
    } for i in issues])
    if df.empty:
        return df
    for col in ("Created", "Updated", "Closed"):
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    return df.dropna(subset=["Created"]).copy()


def styled_fig(fig):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font={"family": "DM Sans, sans-serif", "color": "#dce7fb"},
                      margin={"l": 15, "r": 15, "t": 55, "b": 15}, title={"font": {"size": 16}})
    fig.update_xaxes(gridcolor="#25324a", zerolinecolor="#25324a")
    fig.update_yaxes(gridcolor="#25324a", zerolinecolor="#25324a")
    return fig


def compute_health(features):
    """Transparent heuristic score (0-100), separate from the ML prediction."""
    old_ratio = float(features.get("old_open_issue_ratio", 0) or 0)
    backlog_change = float(features.get("backlog_change_30d", 0) or 0)
    sample_open = int(features.get("open_issue_count", 0) or 0)
    score = 100.0 - min(45.0, old_ratio * 90.0) - min(35.0, max(0.0, backlog_change) * 3.0) - min(20.0, sample_open * 0.5)
    return max(0.0, min(100.0, score))


# ------------------------------ Context loader ------------------------------
def load_context(metrics=False, model=False):
    """Return everything a page needs. Stops the page if no repository was analyzed yet."""
    analysis = st.session_state.get("analysis")
    if not analysis:
        st.info("Dashboard dekhne ke liye repository enter karke **Analyze** par click karein. Example: `facebook/react`.")
        st.markdown("<div class='panel'><div class='section-label'>WHAT YOU GET</div><h3>One workspace for repository health</h3><p class='small-muted'>Track repository metrics, spot aging issues, review AI-assisted risk indicators, and export issue data. Charts appear after live GitHub data is loaded.</p></div>", unsafe_allow_html=True)
        st.stop()

    data, issues = analysis["data"], analysis["issues"]
    issue_df = make_issue_df(issues)
    owner, repo = analysis["repo"].split("/", 1)
    open_df = issue_df[issue_df["Status"] == "open"].copy() if not issue_df.empty else issue_df.copy()
    closed_df = issue_df[issue_df["Status"] == "closed"].copy() if not issue_df.empty else issue_df.copy()
    ctx = {
        "analysis": analysis, "data": data, "issues": issues, "issue_df": issue_df,
        "open_df": open_df, "closed_df": closed_df, "open_count": len(open_df), "closed_count": len(closed_df),
        "owner": owner, "repo": repo,
        "contributor_count": None, "contributor_error": None,
        "pull_request_count": None, "pull_request_error": None,
        "recent_commit_count": None, "commit_error": None,
        "features": {}, "result": {}, "model_error": None,
    }
    if metrics:
        try:
            ctx["contributor_count"] = get_contributor_count(owner, repo)
        except (requests.RequestException, ValueError) as e:
            ctx["contributor_error"] = str(e)
        try:
            ctx["pull_request_count"] = get_pull_request_count(owner, repo)
        except (requests.RequestException, ValueError) as e:
            ctx["pull_request_error"] = str(e)
        try:
            ctx["recent_commit_count"] = get_recent_commit_count(owner, repo, 30)
        except (requests.RequestException, ValueError) as e:
            ctx["commit_error"] = str(e)
    if model and issues:
        try:
            ctx["features"] = build_repository_features(issues)
            ctx["result"] = predict_repository_risk(ctx["features"])
        except (ValueError, KeyError, TypeError) as e:
            ctx["model_error"] = str(e)
    return ctx


def page_header(ctx, title):
    st.markdown(f"<div class='section-label'>{title} · {ctx['data'].get('full_name', ctx['analysis']['repo'])}</div>", unsafe_allow_html=True)


# ------------------------------ Reusable sections ---------------------------
def render_overview(ctx):
    data = ctx["data"]
    page_header(ctx, "REPOSITORY OVERVIEW")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("⭐ Stars", f"{data.get('stargazers_count', 0):,}")
    k2.metric("🐛 Open Issues", f"{data.get('open_issues_count', 0):,}")
    k3.metric("🍴 Forks", f"{data.get('forks_count', 0):,}")
    cc = ctx["contributor_count"]
    k4.metric("👥 Contributors", f"{cc:,}" if cc is not None else "Pending",
              help="Count from the GitHub contributors endpoint; GitHub may temporarily defer contributor statistics.")
    if ctx["contributor_error"]:
        st.caption(f"Contributor count unavailable: {ctx['contributor_error']}")
    elif cc is None:
        st.caption("GitHub contributor statistics are still being calculated or unavailable for this repository.")
    pr, rc = ctx["pull_request_count"], ctx["recent_commit_count"]
    a1, a2 = st.columns(2)
    a1.metric("🔀 Pull Requests (all time)", f"{pr:,}" if pr is not None else "Unavailable")
    a2.metric("💻 Commits (last 30 days)", f"{rc:,}" if rc is not None else "Unavailable")
    if ctx["pull_request_error"]:
        st.caption(f"Pull request metric unavailable: {ctx['pull_request_error']}")
    if ctx["commit_error"]:
        st.caption(f"Commit metric unavailable: {ctx['commit_error']}")
    st.caption(f"{data.get('description') or 'No repository description available.'}  ·  Primary language: {data.get('language') or 'Not specified'}  ·  Issues fetched: {len(ctx['issue_df'])} (maximum 100 regular issues)")


def render_overview_charts(ctx):
    issue_df, open_df = ctx["issue_df"], ctx["open_df"]
    left, right = st.columns([1.1, 1])
    with left:
        st.markdown("<div class='section-label'>ISSUE STATUS</div>", unsafe_allow_html=True)
        status_df = pd.DataFrame({"Status": ["Open", "Closed"], "Count": [ctx["open_count"], ctx["closed_count"]]})
        fig = px.pie(status_df, names="Status", values="Count", hole=.68, color="Status", color_discrete_map={"Open": "#5eead4", "Closed": "#7c9cff"}, title="Open vs Closed Issues")
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(styled_fig(fig), use_container_width=True)
    with right:
        st.markdown("<div class='section-label'>ACTIVITY</div>", unsafe_allow_html=True)
        if not issue_df.empty:
            trend = issue_df.assign(Month=issue_df["Created"].dt.strftime("%Y-%m")).groupby("Month").size().reset_index(name="Issues").sort_values("Month")
            fig = px.line(trend, x="Month", y="Issues", markers=True, title="Monthly Issue Trends", color_discrete_sequence=["#5eead4"])
            st.plotly_chart(styled_fig(fig), use_container_width=True)
        else:
            st.info("Trend chart ke liye issue data available nahi hai.")

    c1, c2 = st.columns([1, 1.2])
    with c1:
        st.markdown("<div class='section-label'>ISSUE AGING</div>", unsafe_allow_html=True)
        if not open_df.empty:
            now = pd.Timestamp.now(tz="UTC")
            aging = open_df.copy()
            aging["Age (Days)"] = (now - aging["Created"]).dt.days.clip(lower=0)
            aging["Age Group"] = pd.cut(aging["Age (Days)"], bins=[-1, 7, 30, 90, float("inf")], labels=["0–7 days", "8–30 days", "31–90 days", "90+ days"])
            age_counts = aging.groupby("Age Group", observed=False).size().reset_index(name="Issues")
            fig = px.bar(age_counts, x="Age Group", y="Issues", title="Open Issue Aging", color="Age Group", color_discrete_sequence=["#5eead4", "#7c9cff", "#fbbf24", "#fb7185"])
            st.plotly_chart(styled_fig(fig), use_container_width=True)
        else:
            st.info("No open issues in fetched sample.")
    with c2:
        st.markdown("<div class='section-label'>ISSUES BY LABEL</div>", unsafe_allow_html=True)
        label_rows = []
        for issue in ctx["issues"]:
            if issue.get("state") == "open":
                for label in issue.get("labels", []):
                    label_rows.append(label.get("name", "Unlabelled"))
        if label_rows:
            label_df = pd.Series(label_rows).value_counts().head(8).rename_axis("Label").reset_index(name="Issues")
            fig = px.pie(label_df, names="Label", values="Issues", hole=.55, title="Open Issues by Label", color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(styled_fig(fig), use_container_width=True)
        else:
            st.info("Fetched open issues par labels nahi mile.")


def render_ai_section(ctx):
    features, result, model_error = ctx["features"], ctx["result"], ctx["model_error"]
    st.markdown("<div class='section-label'>AI-ASSISTED PROJECT HEALTH</div>", unsafe_allow_html=True)
    if result:
        risk = str(result.get("risk", "Unknown"))
        risk_norm = risk.lower()
        old_ratio = float(features.get("old_open_issue_ratio", 0) or 0)
        backlog_change = float(features.get("backlog_change_30d", 0) or 0)
        sample_open = int(features.get("open_issue_count", 0) or 0)
        health_score = compute_health(features)
        a, b, c, d = st.columns(4)
        a.metric("Project Health Score (heuristic)", f"{health_score:.0f}/100")
        b.metric("Predicted Risk", risk)
        c.metric("Model Confidence", f"{float(result.get('confidence', 0)):.1f}%")
        d.metric("Old Open Issue Ratio", f"{old_ratio:.1%}")
        st.caption("Health score is a transparent heuristic based on the sampled backlog and aging indicators; it is not an ML output or a validated project-success probability.")
        if risk_norm in ("high", "high risk"):
            st.warning("Backlog indicators high hain. Aging issues aur blockers review karein.")
        elif risk_norm in ("medium", "medium risk"):
            st.info("Backlog indicators moderate hain. Unresolved issues ko regularly monitor karein.")
        else:
            st.success("Available indicators low risk suggest karte hain; ye healthy project ki guarantee nahi hai.")
        reasons = []
        if old_ratio >= .25: reasons.append(f"{old_ratio:.0%} of sampled open issues are old; review long-pending items.")
        if backlog_change >= 3: reasons.append(f"Sample backlog increased by {backlog_change:.0f} over the model's 30-day estimate.")
        elif backlog_change < 0: reasons.append("Sample backlog decreased in the model's 30-day estimate.")
        if sample_open >= 10: reasons.append(f"{sample_open} open issues were detected in the feature sample.")
        if not reasons: reasons.append("No major warning was triggered by the selected indicators.")
        actions = []
        if old_ratio >= 0.50:
            actions.append("Prioritize a triage session for long-pending issues and close duplicates or stale reports.")
        elif old_ratio >= 0.25:
            actions.append("Review issues older than 30 days and add an owner or next step to each one.")
        if backlog_change > 0:
            actions.append("Check whether incoming issues are outpacing closures; set a weekly backlog target.")
        elif backlog_change < 0:
            actions.append("Keep the current triage cadence and confirm that older issues are also being resolved.")
        if sample_open >= 10:
            actions.append("Group open issues by label and priority, then assign maintainers to the largest groups.")
        if not actions:
            actions.append("Continue weekly issue triage and reassess when the sampled backlog or aging ratio changes.")
        st.markdown("<div class='insight'><h3>✦ AI Insights & Recommended Actions</h3>" + "".join(f"<p>• {r}</p>" for r in reasons) + "<p><b>Recommended next steps</b></p>" + "".join(f"<p>✓ {x}</p>" for x in actions) + "</div>", unsafe_allow_html=True)
        st.caption("Prototype prediction: labels may be rule-generated. Test accuracy is not verified real-world project-failure prediction accuracy.")
        evaluation = result.get("evaluation", {})
        if evaluation:
            st.subheader("Model Evaluation")
            e1, e2, e3 = st.columns(3)
            e1.metric("Macro Precision", f"{evaluation.get('precision_macro', 0):.2%}")
            e2.metric("Macro Recall", f"{evaluation.get('recall_macro', 0):.2%}")
            e3.metric("Macro F1-score", f"{evaluation.get('f1_macro', 0):.2%}")
            labels = evaluation.get("labels", [])
            matrix = evaluation.get("confusion_matrix", [])
            if labels and matrix:
                st.dataframe(pd.DataFrame(matrix, index=[f"Actual {x}" for x in labels], columns=[f"Predicted {x}" for x in labels]), use_container_width=True)
    elif model_error:
        st.warning(f"AI prediction calculate nahi ho paayi: {model_error}")
    else:
        st.info("AI insights ke liye regular issue data required hai.")


def render_priority_table(ctx):
    st.divider()
    st.subheader("🎯 Top Issues Requiring Attention")
    open_df = ctx["open_df"]
    if not open_df.empty:
        now = pd.Timestamp.now(tz="UTC")
        priority = open_df.copy()
        priority["Age (Days)"] = (now - priority["Created"]).dt.days.clip(lower=0)
        priority["Priority"] = priority["Age (Days)"].apply(lambda d: "High attention" if d > 30 else "Monitor" if d > 7 else "Recent")
        priority = priority.sort_values(["Age (Days)", "Comments"], ascending=False).head(10)
        st.caption("Rank is based mainly on age and discussion count; age alone does not prove severity.")
        st.dataframe(priority[["Title", "Age (Days)", "Priority", "Comments", "URL"]], use_container_width=True, hide_index=True, column_config={"URL": st.column_config.LinkColumn("Open issue")})
    else:
        st.success("No open issues in the fetched sample.")


def render_compare(ctx):
    with st.expander("⚖️ Compare with another repository", expanded=False):
        compare_repo = st.text_input("Second repository (owner/repo)", "pallets/flask", key="compare_repo_input")
        if st.button("Compare Repositories"):
            first = ctx["analysis"]["repo"].split("/")
            second = compare_repo.strip().split("/")
            if len(second) != 2 or not all(second):
                st.error("Repository format owner/repo hona chahiye.")
            elif compare_repo.strip().lower() == ctx["analysis"]["repo"].lower():
                st.warning("Comparison ke liye doosri repository enter karo.")
            else:
                try:
                    with st.spinner("Repositories compare ho rahi hain..."):
                        d1 = get_repo_data(first[0], first[1]); i1 = get_repo_issues(first[0], first[1])
                        d2 = get_repo_data(second[0], second[1]); i2 = get_repo_issues(second[0], second[1])
                        f1 = build_repository_features(i1) if i1 else {}; f2 = build_repository_features(i2) if i2 else {}
                        r1 = predict_repository_risk(f1) if f1 else {}; r2 = predict_repository_risk(f2) if f2 else {}
                    compare_df = pd.DataFrame([
                        {"Metric": "Stars", "Repository 1": d1.get("stargazers_count", 0), "Repository 2": d2.get("stargazers_count", 0)},
                        {"Metric": "Forks", "Repository 1": d1.get("forks_count", 0), "Repository 2": d2.get("forks_count", 0)},
                        {"Metric": "Open issues in sample", "Repository 1": f1.get("open_issue_count", 0), "Repository 2": f2.get("open_issue_count", 0)},
                        {"Metric": "Old open issue ratio", "Repository 1": f"{float(f1.get('old_open_issue_ratio', 0)):.1%}", "Repository 2": f"{float(f2.get('old_open_issue_ratio', 0)):.1%}"},
                        {"Metric": "Predicted risk", "Repository 1": r1.get("risk", "No issue sample"), "Repository 2": r2.get("risk", "No issue sample")},
                    ])
                    st.dataframe(compare_df, use_container_width=True, hide_index=True)
                    st.caption("Comparison uses up to 100 regular issues per repository. Predictions are prototype estimates.")
                except (requests.RequestException, ValueError, KeyError, TypeError) as e:
                    st.error(f"Comparison nahi ho paaya: {e}")
