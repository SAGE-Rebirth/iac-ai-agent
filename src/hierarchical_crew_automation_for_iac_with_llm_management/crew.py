from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import GithubSearchTool
from crewai_tools import DirectoryReadTool
from crewai_tools import DirectorySearchTool
from crewai_tools import FileReadTool
from crewai_tools import FileWriterTool
from src.hierarchical_crew_automation_for_iac_with_llm_management.tools.git_clone_tool import GitCloneTool
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Debug prints for model and API key
# print("[DEBUG] Using Hugging Face model:", "huggingface/meta-llama-3-8b-instruct")
# print("[DEBUG] HUGGINGFACE_API_KEY present:", bool(os.getenv("HUGGINGFACE_API_KEY")))

# LLM declaration for Gemini (per CrewAI docs: model="gemini/gemini-2.0-flash")
llm_gemini = LLM(model="gemini/gemini-2.0-flash",temperature=0.7, api_key=os.getenv("GEMINI_API_KEY"), max_tokens=4096,)

# Hugging Face LLM declaration
# llm_hf = LLM(
#     model="huggingface/meta-llama/Meta-Llama-3.1-8B-Instruct",  # You can change to any supported Hugging Face model
#     temperature=0.7,
#     api_key=os.getenv("HF_TOKEN"),
# )

@CrewBase
class HierarchicalCrewAutomationForIacWithLlmManagement():
    """HierarchicalCrewAutomationForIacWithLlmManagement crew"""

    @agent
    def RepoAgent(self) -> Agent:
        return Agent(
            config=self.agents_config['RepoAgent'],
            tools=[
                GitCloneTool(),
                GithubSearchTool(
                    gh_token=os.getenv("GITHUB_API_KEY"),
                    content_types=["code", "repo", "pr", "issue"]
                )
            ],
            llm=llm_gemini,  # Use Gemini LLM for RepoAgent
            allow_delegation=True,
        )

    @agent
    def AnalyzerAgent(self) -> Agent:
        return Agent(
            config=self.agents_config['AnalyzerAgent'],
            tools=[
                DirectoryReadTool(),
                DirectorySearchTool(),
                FileReadTool()
            ],
            llm=llm_gemini,  # Use Gemini LLM for AnalyzerAgent
            allow_delegation=True,
        )

    @agent
    def TerraformAgent(self) -> Agent:
        return Agent(
            config=self.agents_config['TerraformAgent'],
            tools=[FileWriterTool()],
            llm=llm_gemini,  # Use Gemini LLM for TerraformAgent
            allow_delegation=True,
        )

    @agent
    def PipelineAgent(self) -> Agent:
        return Agent(
            config=self.agents_config['PipelineAgent'],
            tools=[FileWriterTool()],
            llm=llm_gemini,  # Use Gemini LLM for PipelineAgent
            allow_delegation=True,
        )

    @agent
    def ManagerLLMAgent(self) -> Agent:
        return Agent(
            config=self.agents_config['ManagerLLMAgent'],
            tools=[],
            llm=llm_gemini,  # Use Gemini LLM for ManagerLLMAgent
            allow_delegation=True,
        )


    @task
    def pull_repositories_task(self) -> Task:
        return Task(
            config=self.tasks_config['pull_repositories_task'],
        )

    @task
    def read_directories_task(self) -> Task:
        return Task(
            config=self.tasks_config['read_directories_task'],
        )

    @task
    def search_files_task(self) -> Task:
        return Task(
            config=self.tasks_config['search_files_task'],
        )

    @task
    def analyze_files_task(self) -> Task:
        return Task(
            config=self.tasks_config['analyze_files_task'],
        )

    @task
    def generate_terraform_files_task(self) -> Task:
        return Task(
            config=self.tasks_config['generate_terraform_files_task'],
        )

    @task
    def create_ci_cd_pipelines_task(self) -> Task:
        return Task(
            config=self.tasks_config['create_ci_cd_pipelines_task'],
        )

    @task
    def monitor_and_suggest_task(self) -> Task:
        return Task(
            config=self.tasks_config['monitor_and_suggest_task'],
        )


    @crew
    def crew(self) -> Crew:
        """Creates the HierarchicalCrewAutomationForIacWithLlmManagement crew"""
        # Exclude the manager agent from the agents list
        agents = [
            self.RepoAgent(),
            self.AnalyzerAgent(),
            self.TerraformAgent(),
            self.PipelineAgent(),
        ]
        return Crew(
            agents=agents,
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.hierarchical,
            verbose=True,
            manager_agent=self.ManagerLLMAgent(),
            # memory=True,  # Enable memory for the crew
        )
