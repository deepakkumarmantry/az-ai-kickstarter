import os
import glob
import yaml
import logging
from typing import List, Optional, Any

from semantic_kernel.kernel import Kernel
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.open_ai import AzureChatPromptExecutionSettings
from semantic_kernel.functions import KernelArguments
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior

class AgentManager:
    """
    Manages the loading and creation of agents from YAML definitions.
    """
    
    def __init__(self, kernel: Kernel, service_id: str = "executor"):
        """
        Initialize the agent manager.
        
        Args:
            kernel: The Semantic Kernel instance
            service_id: The default service ID to use for creating agents
        """
        self.kernel = kernel
        self.service_id = service_id
        self.logger = logging.getLogger(__name__)
        self.critics = []
        self.agents = []
        
    def load_agents_from_directory(self, directory_path: str, reasoning_effort: Optional[float] = None) -> List[ChatCompletionAgent]:
        """
        Load all agent definitions from YAML files in the specified directory.
        
        Args:
            directory_path: Path to directory containing agent YAML definitions
            reasoning_effort: Optional reasoning effort parameter for OpenAI models
            
        Returns:
            List of created agent instances
        """
        self.logger.info(f"Loading agents from directory: {directory_path}")
        
        # Get all YAML files in the specified directory
        yaml_files = glob.glob(os.path.join(directory_path, "*.yaml"))
        yaml_files.extend(glob.glob(os.path.join(directory_path, "**/*.yaml"), recursive=True))
        
        if not yaml_files:
            self.logger.warning(f"No YAML files found in {directory_path}")
            return []
        
        self.logger.info(f"Found {len(yaml_files)} YAML files")
        
        # Create agents from each YAML file
        for file_path in yaml_files:
            try:
                agent = self.create_agent_from_yaml(file_path, reasoning_effort)
                self.agents.append(agent)
                
                # Check if this is a critic agent based on YAML definition
                with open(file_path, 'r', encoding='utf-8') as file:
                    definition = yaml.safe_load(file)
                    if definition.get('is_critic', False):
                        self.critics.append(agent)
                        self.logger.info(f"Added {agent.name} as a critic agent")
                
            except Exception as e:
                self.logger.error(f"Error creating agent from {file_path}: {str(e)}")
        
        self.logger.info(f"Successfully loaded {len(self.agents)} agents")
        return self.agents
    
    def create_agent_from_yaml(self, definition_file_path: str, reasoning_effort: Optional[float] = None) -> ChatCompletionAgent:
        """
        Creates a ChatCompletionAgent from a YAML definition file.
        
        Args:
            definition_file_path: Path to the YAML file containing agent definition
            reasoning_effort: Optional reasoning effort parameter for OpenAI models
            
        Returns:
            ChatCompletionAgent: Configured agent instance
        """
        self.logger.debug(f"Creating agent from {definition_file_path}")
        
        with open(definition_file_path, 'r', encoding='utf-8') as file:
            definition = yaml.safe_load(file)
            
        # Determine service ID - use agent-specific if specified, otherwise default
        service_id = definition.get('service_id', self.service_id)
            
        settings = AzureChatPromptExecutionSettings(
                temperature=definition.get('temperature', 0.5),
                function_choice_behavior=FunctionChoiceBehavior.Auto(
                    filters={"included_plugins": definition.get('included_plugins', [])}
                ))

        # Reasoning model specifics
        model_id = self.kernel.get_service(service_id=service_id).ai_model_id
        if model_id and model_id.lower().startswith("o"):
            settings.temperature = None
            settings.reasoning_effort = reasoning_effort
            
        agent = ChatCompletionAgent(
            service=self.kernel.get_service(service_id=service_id),
            kernel=self.kernel,
            arguments=KernelArguments(settings=settings),
            name=definition['name'],
            description=definition['description'],
            instructions=definition['instructions']
        )
        
        self.logger.debug(f"Created agent: {agent.name}")
        return agent
    
    def get_critics(self) -> List[ChatCompletionAgent]:
        """
        Get all critic agents.
        
        Returns:
            List of critic agents
        """
        return self.critics
    
    def get_all_agents(self) -> List[ChatCompletionAgent]:
        """
        Get all loaded agents.
        
        Returns:
            List of all loaded agents
        """
        return self.agents
    
    def get_agent_by_name(self, name: str) -> Optional[ChatCompletionAgent]:
        """
        Get an agent by its name.
        
        Args:
            name: The name of the agent to find
            
        Returns:
            The agent with the specified name, or None if not found
        """
        for agent in self.agents:
            if agent.name == name:
                return agent
        return None