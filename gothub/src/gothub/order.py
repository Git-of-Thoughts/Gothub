import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from github import Github
from github.Auth import Token
from pydantic.dataclasses import dataclass

from gots.repo_agent import (
    RepoAgent,
    WriteRepoInp,
    WriteRepoOut,
    gots_repo_agent,
)

from .setup_repo import SetupRepoInp, setup_repo
from .write_github import GothubPullRequest, create_pull_request

SETUP_ORDERS_BASE_DIR = "orders/"


@dataclass
class GithubOrderInp:
    username: str
    https_url: str
    github_token: str
    openai_api_key: str
    branch_name: Optional[str] = None
    extra_prompt: Optional[str] = None
    repo_agent: RepoAgent = gots_repo_agent
    tools_selected: Optional[Dict[str, bool]] = None


@dataclass
class GithubOrderOut:
    order_id: str
    pull_requests: list[GothubPullRequest]


def take_order(inp: GithubOrderInp) -> GithubOrderOut:
    match inp:
        case GithubOrderInp(
            username=username,
            https_url=https_url,
            github_token=github_token,
            openai_api_key=openai_api_key,
            branch_name=branch_name,
            extra_prompt=extra_prompt,
            repo_agent=repo_agent,
            tools_selected=tools_selected,
        ):
            pass

    time = datetime.now().strftime("%Y-%m-%d %H_%M_%S_%f")

    setup_dir = SETUP_ORDERS_BASE_DIR + username + "/" + time

    setup_repo_inp = SetupRepoInp(
        setup_dir=setup_dir,
        https_url=https_url,
        github_token=github_token,
        branch_name=branch_name,
    )

    with setup_repo(setup_repo_inp) as repo:
        write_repo_out = repo_agent(
            WriteRepoInp(
                repo=repo,
                openai_api_key=openai_api_key,
                extra_prompt=extra_prompt,
                tools_selected=tools_selected,
            )
        )

        new_branches = write_repo_out.new_branches

        # TODO abstract this away
        for branch in new_branches:
            repo.remote().push(branch)

    # TODO abstract this away
    github = Github(auth=Token(github_token))
    pattern = r"github\.com/([^/]+)/([^/]+)\.git"
    match = re.search(pattern, https_url)
    full_name = match.group(1) + "/" + match.group(2) if match else ""
    github_repo = github.get_repo(full_name)

    new_pull_requests = [
        create_pull_request(
            github_repo,
            branch,
        )
        for branch in new_branches
    ]

    return GithubOrderOut(
        order_id=time,
        pull_requests=new_pull_requests,
    )
