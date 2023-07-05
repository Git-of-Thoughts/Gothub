# stdlib imports
from typing import Optional, Callable

# third-party imports
from pydantic.dataclasses import dataclass
from pydantic import BaseModel
from git import Repo, Head


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


RepoWriter = Callable[[WriteRepoInp], WriteRepoOut]