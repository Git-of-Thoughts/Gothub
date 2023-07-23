import os

from git import Repo

from gots.repo_agent import WriteRepoInp, WriteRepoOut, one_branch_mrkl


def main():
    # assuming WriteRepoInp is a dataclass or similar object
    repo_inp = WriteRepoInp(
        repo=Repo(str(os.getcwd())),  # assuming Repo is the type of your repo object
        openai_api_key="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",
        extra_prompt="help me write a script that uses some gpt4 apis",
    )

    one_branch_mrkl(repo_inp)


if __name__ == "__main__":
    main()
