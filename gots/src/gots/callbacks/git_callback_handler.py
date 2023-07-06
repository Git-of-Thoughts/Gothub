import sys
from typing import Any, Dict, List, Union

from git import Repo
from langchain.callbacks.base import BaseCallbackHandler


# First, define custom callback handler implementations
class GitCallbackHandler(BaseCallbackHandler):
    def __init__(self, repo: Repo):
        self.repo = repo

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> Any:
        """Run when tool starts running."""

    def on_llm_start(self, **kwargs: Any) -> Any:
        """Run when tool ends running."""
        if self.repo.is_dirty():
            self.repo.git.add(A=True)  # This will add all files to the staging area
            self.repo.git.commit("-m", "on llm start")

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        **kwargs: Any,
    ) -> Any:
        """Run when chain ends running."""
        if self.repo.is_dirty():
            self.repo.git.add(A=True)  # This will add all files to the staging area
            self.repo.git.commit("-m", "on chain end")

    def on_agent_finish(
        self,
        finish,
        **kwargs: Any,
    ) -> Any:
        """Run on agent end."""
        if self.repo.is_dirty():
            self.repo.git.add(A=True)  # This will add all files to the staging area
            self.repo.git.commit("-m", "on agent finish")
