import re
from datetime import datetime
from typing import Callable, List, Optional, Union

import langchain
from git import Head, Repo
from langchain import LLMChain, OpenAI, SerpAPIWrapper
from langchain.agents import (
    AgentExecutor,
    AgentOutputParser,
    AgentType,
    LLMSingleActionAgent,
    Tool,
    initialize_agent,
)
from langchain.chat_models import ChatOpenAI
from langchain.prompts import StringPromptTemplate
from langchain.schema import (
    AgentAction,
    AgentFinish,
    HumanMessage,
    OutputParserException,
)
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

    template = """You are a smart software developer.
        You are working on a project that requires you to understand in
        depth the design and use of this project.
        You should be clear at each step what you are doing,
        and like a human software developer to
        notice what is the git commit message, branch, base,
        and more git related information.
        You should and always report the
        git commit message at steps where you feel like
        you made a major step in the generation process.
        You have access to the following tools:

        {tools}

        Use Strctly the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        Begin! Remember to include git commit message
        when you make a major step in the generation process.
        And Make sure your follow the format provided at every step.

        Question: {input}
        {agent_scratchpad}"""

    prompt = CustomPromptTemplate(
        template=template,
        tools=tools,
        input_variables=["input", "intermediate_steps"],
    )

    output_parser = CustomOutputParser()

    llm = ChatOpenAI(
        temperature=0,
        # model="gpt-3.5-turbo-0613",
        model="gpt-4-0613",
        openai_api_key=openai_api_key,
        callbacks=[GitCallbackHandler(repo)],
    )

    llm_chain = LLMChain(llm=llm, prompt=prompt)

    tool_names = [tool.name for tool in tools]
    agent = LLMSingleActionAgent(
        llm_chain=llm_chain,
        output_parser=output_parser,
        stop=["\nObservation:"],
        allowed_tools=tool_names,
    )

    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=True
    )
    agent_executor.run(extra_prompt)


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


# Set up a prompt template
class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join(
            [f"{tool.name}: {tool.description}" for tool in self.tools]
        )
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)


class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise OutputParserException(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(
            tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output
        )
