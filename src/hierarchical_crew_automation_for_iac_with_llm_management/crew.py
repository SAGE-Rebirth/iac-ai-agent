from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import GithubSearchTool
from crewai_tools import DirectoryReadTool
from crewai_tools import DirectorySearchTool
from crewai_tools import FileReadTool
from crewai_tools import FileWriterTool
import os

# LLM declaration for Gemini
llm_gemini = LLM(
    model="gemini/gemini-2.0-flash",
    temperature=0.7,
    api_key=os.getenv("GEMINI_API_KEY")
)

@CrewBase
class HierarchicalCrewAutomationForIacWithLlmManagementCrew():
    """HierarchicalCrewAutomationForIacWithLlmManagement crew"""

    @agent
    def RepoAgent(self) -> Agent:
        return Agent(
            config=self.agents_config['RepoAgent'],
            tools=[],
            llm=llm_gemini,
            allow_delegation=True,
        )

    @agent
    def AnalyzerAgent(self) -> Agent:
        return Agent(
            config=self.agents_config['AnalyzerAgent'],
            tools=[],
            llm=llm_gemini,
            allow_delegation=True,
        )

    @agent
    def TerraformAgent(self) -> Agent:
        return Agent(
            config=self.agents_config['TerraformAgent'],
            tools=[],
            llm=llm_gemini,
            allow_delegation=True,
        )

    @agent
    def PipelineAgent(self) -> Agent:
        return Agent(
            config=self.agents_config['PipelineAgent'],
            tools=[],
            llm=llm_gemini,
            allow_delegation=True,
        )

    @agent
    def ManagerLLMAgent(self) -> Agent:
        return Agent(
            config=self.agents_config['ManagerLLMAgent'],
            tools=[],
            llm=llm_gemini,
            allow_delegation=True,
        )


    @task
    def pull_repositories_task(self) -> Task:
        return Task(
            config=self.tasks_config['pull_repositories_task'],
            tools=[GithubSearchTool(
                gh_token=os.getenv("GITHUB_API_KEY"),
                content_types=["code", "repo", "pr", "issue"]
            )],
        )

    @task
    def read_directories_task(self) -> Task:
        return Task(
            config=self.tasks_config['read_directories_task'],
            tools=[DirectoryReadTool()],
        )

    @task
    def search_files_task(self) -> Task:
        return Task(
            config=self.tasks_config['search_files_task'],
            tools=[DirectorySearchTool()],
        )

    @task
    def analyze_files_task(self) -> Task:
        return Task(
            config=self.tasks_config['analyze_files_task'],
            tools=[FileReadTool()],
        )

    @task
    def generate_terraform_files_task(self) -> Task:
        return Task(
            config=self.tasks_config['generate_terraform_files_task'],
            tools=[FileWriterTool()],
        )

    @task
    def create_ci_cd_pipelines_task(self) -> Task:
        return Task(
            config=self.tasks_config['create_ci_cd_pipelines_task'],
            tools=[FileWriterTool()],
        )

    @task
    def monitor_and_suggest_task(self) -> Task:
        return Task(
            config=self.tasks_config['monitor_and_suggest_task'],
            tools=[],
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
            memory=True,  # Enable memory for the crew
        )
