from datetime import datetime
from typing import Callable, Optional

import langchain
from git import Head, Repo
from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
from pydantic import BaseModel
from pydantic.dataclasses import dataclass

from gots.tools.oracle_runner import oracle_runner_factory

from .callbacks.git_callback_handler import GitCallbackHandler
from .tools.scoped_file_tools import build_scoped_file_tools

# keep this true if you want to see the outputs
langchain.debug = True


class WriteRepoInp(BaseModel):
    repo: Repo
    openai_api_key: str
    extra_prompt: Optional[str]

    class Config:
        arbitrary_types_allowed = True


class WriteRepoOut(BaseModel):
    new_branches: list[Head]

    class Config:
        arbitrary_types_allowed = True


RepoAgent = Callable[[WriteRepoInp], WriteRepoOut]


def one_branch_mrkl(inp: WriteRepoInp) -> None:
    match inp:
        case WriteRepoInp(
            repo=repo,
            openai_api_key=openai_api_key,
            extra_prompt=extra_prompt,
        ):
            pass

    tools = [
        *build_scoped_file_tools(repo.working_dir),
        oracle_runner_factory(repo.working_dir + "/.."),
    ]

    llm = ChatOpenAI(
        temperature=0,
        model="gpt-3.5-turbo-0613",
        # model="gpt-4-0613",
        openai_api_key=openai_api_key,
        callbacks=[GitCallbackHandler(repo)],
    )

    mrkl = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        callbacks=[GitCallbackHandler(repo)],
        verbose=True,
    )

    mrkl.run(extra_prompt)


def gots_repo_agent(inp: WriteRepoInp) -> WriteRepoOut:
    """
    ! Should only modify what's permitted by inp
    """
    match inp:
        case WriteRepoInp(
            repo=repo,
            openai_api_key=openai_api_key,
            extra_prompt=extra_prompt,
        ):
            pass

    time = datetime.now().strftime("%Y-%m-%d_%H_%M_%S_%f")
    original_branch = repo.active_branch

    # TODO Create more than one branch
    new_branch_name = "gothub_gots" + time
    new_branch = repo.create_head(new_branch_name)
    new_branch.checkout()

    # Replace this with the actual code
    repo.git.commit("--allow-empty", "-m", "empty commit: start")
    one_branch_mrkl(inp)
    repo.git.commit("--allow-empty", "-m", "empty commit: end")

    original_branch.checkout()

    return WriteRepoOut(
        new_branches=[new_branch],
    )
