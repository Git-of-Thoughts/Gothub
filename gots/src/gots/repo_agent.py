from datetime import datetime
from typing import Callable, Dict, List, Optional

import langchain
from git import Head, Repo
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.schema import AgentAction, AgentFinish, HumanMessage
from pydantic import BaseModel
from pydantic.dataclasses import dataclass

from .callbacks.git_callback_handler import GitCallbackHandler
from .tools.scoped_file_tools import build_scoped_file_tools

# keep this true if you want to see the outputs
langchain.debug = True

filepath = "/Users/wayne/Desktop/Goth Repos/Gothub/gots/src/gots/prompts/pretrain.txt"

with open(filepath, "r") as file:
    content = file.read()


class WriteRepoInp(BaseModel):
    repo: Repo
    openai_api_key: str
    extra_prompt: Optional[str]
    tools_selected: Optional[Dict[str, bool]]

    class Config:
        arbitrary_types_allowed = True


class WriteRepoOut(BaseModel):
    new_branches: list[Head]

    class Config:
        arbitrary_types_allowed = True


RepoAgent = Callable[[WriteRepoInp], WriteRepoOut]


def get_selected_tools(
    tool_selection: Dict[str, bool], tools: Dict[str, Tool]
) -> List[Tool]:
    return [
        tools[tool_name]
        for tool_name, is_selected in tool_selection.items()
        if is_selected
    ]


def one_branch_mrkl(inp: WriteRepoInp) -> None:
    match inp:
        case WriteRepoInp(
            repo=repo,
            openai_api_key=openai_api_key,
            extra_prompt=extra_prompt,
            tools_selected=tools_selected,
        ):
            pass

    if tools_selected:
        tools_dict = build_scoped_file_tools(repo.working_dir)

        tools = get_selected_tools(tools_selected, tools_dict)
    else:
        tools = list(build_scoped_file_tools(repo.working_dir).values())

    llm = ChatOpenAI(
        temperature=0,
        model="gpt-3.5-turbo-0613",
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
