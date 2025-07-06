from typing import Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from git import Repo
import os

class GitCloneToolInput(BaseModel):
    repo_url: str = Field(..., description="The URL of the GitHub repository to clone.")
    dest_dir: str = Field(..., description="The destination directory to clone the repository into.")

class GitCloneTool(BaseTool):
    name: str = "Git Clone Tool"
    description: str = "Clones a GitHub repository to a specified local directory. Returns the local path."
    args_schema: Type[BaseModel] = GitCloneToolInput

    def _run(self, repo_url: str = None, dest_dir: str = None, **kwargs) -> str:
        # Allow fallback to context/inputs if not directly provided
        if repo_url is None:
            repo_url = kwargs.get('GitHub_repository_URLs')
        if dest_dir is None:
            dest_dir = kwargs.get('destination_dir')
        try:
            if not repo_url or not dest_dir:
                return "Missing repo_url or dest_dir for cloning."
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
            Repo.clone_from(repo_url, dest_dir)
            # Return the local path for downstream tasks
            return dest_dir
        except Exception as e:
            return f"Failed to clone repository: {str(e)}"
