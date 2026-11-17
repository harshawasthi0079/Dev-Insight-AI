import os
import time
import tomllib
from pathlib import Path

import pandas as pd
import requests

from ml_features import build_repository_features

TARGET_REPOSITORIES = 100
ISSUES_PER_REPOSITORY = 100
OUTPUT_FILE = Path(__file__).with_name("repository_dataset.csv")

BASE_URL = "https://api.github.com"

SESSION = requests.Session()
SESSION.headers.update({
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
})


def load_github_token():
    """Read token from environment or Streamlit secrets."""
    token = os.environ.get("GITHUB_TOKEN")

    secrets_file = Path(__file__).parent / ".streamlit" / "secrets.toml"

    if not token and secrets_file.exists():
        with secrets_file.open("rb") as file:
            secrets = tomllib.load(file)
        token = secrets.get("GITHUB_TOKEN")

    if token:
        SESSION.headers["Authorization"] = f"Bearer {token}"


def github_get(url, params=None):
    """Make a GitHub API request with basic retries."""
    for attempt in range(3):
        response = SESSION.get(
            url,
            params=params,
            timeout=(10, 30)
        )

        if response.status_code == 403 and response.headers.get(
            "X-RateLimit-Remaining"
        ) == "0":
            reset_at = int(
                response.headers.get("X-RateLimit-Reset", "0")
            )
            wait_seconds = max(
                reset_at - int(time.time()), 1
            )
            raise RuntimeError(
                f"GitHub rate limit reached. Try again in "
                f"about {wait_seconds} seconds."
            )

        if response.status_code in (500, 502, 503, 504):
            if attempt < 2:
                time.sleep(2 ** attempt)
                continue

        response.raise_for_status()
        return response.json()

    raise RuntimeError("GitHub request failed after retries.")


def find_repositories():
    """Search multiple pages of public repositories."""
    found = []
    seen = set()

    for page in range(1, 11):
        data = github_get(
            f"{BASE_URL}/search/repositories",
            params={
                "q": "is:public archived:false fork:false stars:>10 has:issues",
                "sort": "updated",
                "order": "desc",
                "per_page": 100,
                "page": page,
            },
        )

        items = data.get("items", [])

        if not items:
            break

        for item in items:
            name = item.get("full_name")

            if name and name not in seen:
                found.append(item)
                seen.add(name)

        if len(items) < 100:
            break

    return found


def collect_repository_issues(owner, repo):
    """Fetch up to 100 recently updated regular issues."""
    data = github_get(
        f"{BASE_URL}/repos/{owner}/{repo}/issues",
        params={
            "state": "all",
            "sort": "updated",
            "direction": "desc",
            "per_page": ISSUES_PER_REPOSITORY,
        },
    )

    return [
        item for item in data
        if isinstance(item, dict)
        and "pull_request" not in item
    ]


def main():
    load_github_token()

    # Preserve the current dataset.
    if OUTPUT_FILE.exists():
        existing_df = pd.read_csv(OUTPUT_FILE)
    else:
        existing_df = pd.DataFrame()

    if (
        not existing_df.empty
        and "repository" in existing_df.columns
    ):
        existing_df = existing_df.drop_duplicates(
            subset=["repository"],
            keep="first"
        )
        existing_names = set(
            existing_df["repository"].dropna().astype(str)
        )
    else:
        existing_names = set()

    print(f"Existing repositories: {len(existing_names)}")

    if len(existing_names) >= TARGET_REPOSITORIES:
        print("Target already reached. No new collection needed.")
        return

    print("Searching public GitHub repositories...")
    repositories = find_repositories()

    if not repositories:
        raise RuntimeError("No repositories found.")

    rows = []
    failures = []

    for index, repo_info in enumerate(repositories, start=1):
        if len(existing_names) + len(rows) >= TARGET_REPOSITORIES:
            break

        full_name = repo_info["full_name"]

        if full_name in existing_names:
            continue

        owner, repo = full_name.split("/", 1)
        print(f"[{index}/{len(repositories)}] Collecting {full_name}")

        try:
            issues = collect_repository_issues(owner, repo)

            if not issues:
                print("  Skipped: no regular issues returned.")
                continue

            features = build_repository_features(issues)

            rows.append({
                "repository": full_name,
                "stars": repo_info.get("stargazers_count", 0),
                "language": repo_info.get("language") or "Unknown",
                "fetched_issue_count": len(issues),
                **features,
            })

            # Save progress without losing existing rows.
            combined_df = pd.concat(
                [existing_df, pd.DataFrame(rows)],
                ignore_index=True
            ) if not existing_df.empty else pd.DataFrame(rows)

            combined_df = combined_df.drop_duplicates(
                subset=["repository"],
                keep="first"
            )

            combined_df.to_csv(OUTPUT_FILE, index=False)

            print(
                f"  Saved progress: {len(combined_df)} unique repositories"
            )

        except (
            requests.RequestException,
            ValueError,
            KeyError,
            RuntimeError,
        ) as error:
            failures.append({
                "repository": full_name,
                "error": str(error),
            })
            print(f"  Failed: {error}")

    # Final save.
    if rows:
        combined_df = pd.concat(
            [existing_df, pd.DataFrame(rows)],
            ignore_index=True
        ) if not existing_df.empty else pd.DataFrame(rows)

        combined_df = combined_df.drop_duplicates(
            subset=["repository"],
            keep="first"
        )

        combined_df.to_csv(OUTPUT_FILE, index=False)
    else:
        combined_df = existing_df

    total = (
        combined_df["repository"].nunique()
        if not combined_df.empty and "repository" in combined_df.columns
        else 0
    )

    print(f"\nTotal unique repositories saved: {total}")
    print(f"Dataset location: {OUTPUT_FILE}")
    print("Each row uses up to 100 recent issues.")
    print("Risk labels have NOT been created yet.")

    if total < TARGET_REPOSITORIES:
        print(f"Still need {TARGET_REPOSITORIES - total} repositories.")
        print("Run the script again later to try more repositories.")
    else:
        print("Target of 100 repositories reached!")

    if failures:
        print(f"Failed repositories: {len(failures)}")


if __name__ == "__main__":
    main()