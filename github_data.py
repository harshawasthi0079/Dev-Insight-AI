import requests


def get_repo_data(owner, repo):
    owner = owner.strip()
    repo = repo.strip()

    # Validate repository details
    if not owner or not repo:
        print("Error: Owner and repository are required")
        return None

    url = f"https://api.github.com/repos/{owner}/{repo}"

    headers = {
        "Accept": "application/vnd.github+json"
    }

    try:
        response = requests.get(
            url,
            headers=headers,
            timeout=15
        )

        if response.status_code != 200:
            print("GitHub API Error:", response.status_code)
            print(response.text)
            return None

        data = response.json()

        return data

    except requests.exceptions.RequestException as e:
        print("Connection Error:", e)
        return None


if __name__ == "__main__":

    data = get_repo_data("streamlit", "streamlit")

    if data:
        print("Repository:", data["full_name"])
        print("Stars:", data["stargazers_count"])
        print("Open Issues:", data["open_issues_count"])
        print("Forks:", data["forks_count"])