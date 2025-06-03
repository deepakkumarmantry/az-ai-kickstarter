import os
import re
import json
import logging
from typing import ClassVar
import datetime
from utils.util import describe_next_action
from patterns.agent_manager import AgentManager

from semantic_kernel.kernel import Kernel
from semantic_kernel.agents import AgentGroupChat
from semantic_kernel.exceptions.agent_exceptions import AgentChatException
from semantic_kernel.agents.strategies.termination.termination_strategy import (
    TerminationStrategy,
)
from semantic_kernel.agents.strategies import KernelFunctionSelectionStrategy
from semantic_kernel.connectors.ai.open_ai import AzureChatPromptExecutionSettings

from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole
from semantic_kernel.core_plugins.time_plugin import TimePlugin
from semantic_kernel.functions import (
    KernelPlugin,
    KernelFunctionFromPrompt,
    KernelArguments,
)

from semantic_kernel.connectors.ai.azure_ai_inference import (
    AzureAIInferenceChatCompletion,
)
from azure.ai.inference.aio import ChatCompletionsClient
from azure.identity.aio import DefaultAzureCredential

from semantic_kernel.contents import ChatHistory
from semantic_kernel.agents.strategies import (
    SequentialSelectionStrategy,
    DefaultTerminationStrategy,
)

from semantic_kernel.agents.strategies.selection.selection_strategy import (
    SelectionStrategy,
)

from opentelemetry.trace import get_tracer

from pydantic import Field
########################################


# This pattern demonstrates how a debate between equally skilled models
# can deliver an outcome that exceeds the capability of the model if
# the task is handled as a single request-response in its entirety.
# We focus each agent on the subset of the whole task and thus
# get better results.
class DebateOrchestrator:
    """
    Orchestrates a debate between AI agents to produce higher quality responses.

    This class sets up and manages a conversation between Writer and Critic agents using
    Semantic Kernel's Agent Group Chat functionality. The debate pattern improves response
    quality by allowing specialized agents to focus on different aspects of the task.
    """

    # --------------------------------------------
    # Constructor
    # --------------------------------------------
    def __init__(self):
        """
        Creates the DebateOrchestrator with necessary services and kernel configurations.

        Sets up Azure OpenAI connections for both executor and utility models,
        configures Semantic Kernel, and prepares execution settings for the agents.
        """

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.info("Semantic Orchestrator Handler init")

        self.logger.info("Creating - %s", os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"))

        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        executor_deployment_name = os.getenv("EXECUTOR_AZURE_OPENAI_DEPLOYMENT_NAME")
        utility_deployment_name = os.getenv("UTILITY_AZURE_OPENAI_DEPLOYMENT_NAME")

        credential = DefaultAzureCredential()

        # Multi model setup - a service is an LLM in SK terms
        # Executor - gpt-4o
        # Utility  - gpt-4o-mini
        executor_service = AzureAIInferenceChatCompletion(
            ai_model_id="executor",
            service_id="executor",
            client=ChatCompletionsClient(
                endpoint=f"{str(endpoint).strip('/')}/openai/deployments/{executor_deployment_name}",
                api_version=api_version,
                credential=credential,
                credential_scopes=["https://cognitiveservices.azure.com/.default"],
            ),
        )

        utility_service = AzureAIInferenceChatCompletion(
            ai_model_id="utility",
            service_id="utility",
            client=ChatCompletionsClient(
                endpoint=f"{str(endpoint).strip('/')}/openai/deployments/{utility_deployment_name}",
                api_version=api_version,
                credential=credential,
                credential_scopes=["https://cognitiveservices.azure.com/.default"],
            ),
        )

        self.kernel = Kernel(
            services=[executor_service, utility_service],
            plugins=[
                KernelPlugin.from_object(
                    plugin_instance=TimePlugin(), plugin_name="time"
                )
            ],
        )

        self.settings_executor = AzureChatPromptExecutionSettings(
            service_id="executor", temperature=0
        )
        self.settings_utility = AzureChatPromptExecutionSettings(
            service_id="utility", temperature=0
        )

        self.resourceGroup = os.getenv("AZURE_RESOURCE_GROUP")

        # Create the agent manager
        self.agent_manager = AgentManager(self.kernel, service_id="executor")
 

 
    # --------------------------------------------
    # Create Agent Group Chat
    # --------------------------------------------
    def create_agent_group_chat(
        self, agents_directory="agents/cosmos", maximum_iterations=10
    ):
        """
        Creates and configures an agent group chat with Writer and Critic agents.

        Returns:
            AgentGroupChat: A configured group chat with specialized agents,
                           selection strategy and termination strategy.
        """

        self.logger.debug("Creating chat")

        # Load all agents from directory
        agents = self.agent_manager.load_agents_from_directory(agents_directory)

        if not agents:
            raise ValueError(f"No agents found in {agents_directory}")

        # Get critics for termination strategy
        critics = self.agent_manager.get_critics()
        if not critics:
            self.logger.warning(
                "No critic agents found. Using default termination strategy."
            )
            # Find any agent named "Critic-Team" if is_critic wasn't specified
            for agent in agents:
                if "critic" in agent.name.lower():
                    critics.append(agent)
                    self.logger.info(f"Using {agent.name} as critic based on name")

        # Create agent group chat with all loaded agents
        agent_group_chat = AgentGroupChat(
            agents=agents,
            selection_strategy=SequentialSelectionStrategy(),
            termination_strategy=DefaultTerminationStrategy(maximum_iterations=10),
            # selection_strategy=self.create_selection_strategy(agents),
            # termination_strategy=self.create_termination_strategy(
            #     agents=critics
            #     if critics
            #     else [agents[-1]],  # Use critics or last agent if no critics
            #     maximum_iterations=maximum_iterations,
            # ),
        )

        return agent_group_chat

    # --------------------------------------------
    # Speaker Selection Strategy
    # --------------------------------------------
    def create_selection_strategy(self, agents):
        """
        Creates a strategy to determine which Cosmos DB specialist agent speaks next in the conversation.

        Uses the executor model to analyze conversation context and select the most
        appropriate next speaker based on the conversation history and specific Cosmos DB domains.

        Args:
            agents: List of all available agents

        Returns:
            KernelFunctionSelectionStrategy: A strategy for selecting the next speaker.
        """
        default_agent = agents[0]  # Use first agent as default

        # Create a well-formatted definition of each agent with clear responsibilities
        agent_definitions = "\n\n".join(
            [
                f"{agent.name}:\n- Description: {agent.description}\n- When to select: {self._get_selection_criteria(agent.name)}"
                for agent in agents
            ]
        )

        selection_function = KernelFunctionFromPrompt(
            function_name="CosmosDBSpeakerSelector",
            prompt_execution_settings=self.settings_executor,
            prompt=rf"""
                You are the next speaker selector for a conversation between Azure Cosmos DB specialist agents.
                Your task is to analyze the conversation history and determine which specialist should speak next.
                
                # SELECTION RULES
                1. Select the agent whose expertise best addresses the current topic or question
                2. If a question is specifically directed to a particular agent, select that agent
                3. Ensure balanced participation - avoid having the same agent speak multiple times in a row unless necessary
                4. If the conversation seems to be reaching a conclusion, select an agent who can provide comprehensive closing thoughts
                5. Use the "When to select" guidelines for each agent to help make your decision
                
                # IMPORTANT INSTRUCTIONS
                - You MUST return ONLY ONE agent name from the list of available agents below
                - The agent name must be returned exactly as written (case-sensitive, no abbreviations)
                - Return ONLY the agent name with no additional text or explanation
                
                # AVAILABLE AGENTS
                {agent_definitions}
                
                # CONVERSATION HISTORY
                {{{{$history}}}}
                
                Based on the above, which agent should speak next?
            """,
        )

        # Result parser function to extract agent name from output
        def parse_selection_output(output):
            self.logger.info("------- Speaker selection output: %s", output)

            if output.value is not None:
                # Extract just the agent name, ignoring any explanations or extra text
                content = output.value[0].content.strip()

                # If the output contains multiple lines, take just the first line
                if "\n" in content:
                    content = content.split("\n")[0]

                # Check if the output contains one of the valid agent names
                for agent in agents:
                    if agent.name in content:
                        self.logger.info(f"Selected agent: {agent.name}")
                        return agent.name

                # If no valid agent name found, return the default
                self.logger.warning(
                    f"No valid agent found in output, using default: {default_agent.name}"
                )
                return default_agent.name
            else:
                self.logger.warning(
                    f"Empty selection output, using default: {default_agent.name}"
                )
                return default_agent.name

        return KernelFunctionSelectionStrategy(
            kernel=self.kernel,
            function=selection_function,
            result_parser=parse_selection_output,
            agent_variable_name="agents",
            history_variable_name="history",
        )
    

    def _get_selection_criteria(self, agent_name):
        """Helper method to provide selection criteria based on agent name"""

        criteria = {
            "CosmosUseCaseFit": "Select when evaluating if Cosmos DB is right for a specific scenario, comparing with alternatives, or determining appropriate API choice",
            "CosmosPricing": "Select when discussing costs, RU calculation, throughput provisioning, or optimizing for cost efficiency",
            "CosmosDataModel": "Select when designing document schemas, choosing partition keys, or discussing data modeling patterns and trade-offs",
            "CosmosIntegration": "Select when covering integration with other Azure services, SDKs, change feed patterns, or deployment architectures",
            "CosmosPerformanceTuning": "Select when addressing performance optimization, query efficiency, indexing strategies, or RU consumption patterns",
            "CosmosSecurity": "Select when discussing security configurations, authentication, authorization, encryption, or compliance requirements",
            "CosmosReliability": "Select when covering high availability, disaster recovery, multi-region deployments, or consistency level selection",
            "CosmosMigration": "Select when planning migrations from other databases, discussing schema transformation, or data validation strategies",
            "CosmosTroubleshooting": "Select when resolving specific errors, connectivity issues, throttling problems, or other operational challenges",
            "CosmosMonitoring": "Select when setting up monitoring, interpreting metrics, configuring alerts, or implementing operational dashboards",
            "CosmosAPISpecialist": "Select when discussing specific API implementations, compatibility issues, or API-specific optimizations",
            "Critic-Team": "Select after major solution components have been proposed to evaluate completeness and identify gaps",
        }

        return criteria.get(
            agent_name,
            "Select when topics related to this agent's expertise are being discussed",
        )

    def create_termination_strategy(self, agents, maximum_iterations=10):
        """
        Creates a strategy to determine when the agent conversation should end.

        The strategy terminates the conversation when the Critic agent evaluates that all
        requirements have been addressed or when maximum iterations are reached.

        Args:
            agents: List of critic agents that can trigger termination evaluation.
            maximum_iterations: Maximum number of conversation turns before forced termination.

        Returns:
            CompletionTerminationStrategy: A strategy for determining when to end the conversation.
        """

        class CompletionTerminationStrategy(TerminationStrategy):
            logger: ClassVar[logging.Logger] = logging.getLogger(__name__)

            iteration: int = Field(default=0)
            kernel: ClassVar[Kernel] = self.kernel

            # Function to evaluate if conversation should terminate based on completeness
            termination_function: ClassVar[KernelFunctionFromPrompt] = (
                KernelFunctionFromPrompt(
                    function_name="TerminationEvaluator",
                    prompt_execution_settings=self.settings_utility,
                    prompt=rf"""
                    You are evaluating whether a conversation between specialized Azure Cosmos DB agents has reached a 
                    satisfactory conclusion.
                    
                    Review the conversation and evaluate if all key requirements have been addressed across these potential areas:
                    1. Use case suitability evaluation
                    2. Pricing and cost optimization strategies
                    3. Data model design and schema optimization
                    4. Integration with other services
                    5. Performance optimization recommendations
                    6. Security and compliance requirements
                    7. Reliability and disaster recovery planning
                    8. Migration strategy (if applicable)
                    9. Troubleshooting guidance (if applicable)
                    10. Monitoring configuration (if applicable)
                    11. API-specific implementation details (if applicable)
                    
                    Note that not all areas need to be addressed - only those relevant to the user's query.
                    
                    First, provide a score from 0-10 indicating how completely requirements have been met.
                    Then provide your assessment in this format:
                    
                    SCORE: [number between 0-10]
                    ASSESSMENT: [brief explanation]
                    DECISION: [TERMINATE or CONTINUE]
                    
                    Example:
                    SCORE: 8.5
                    ASSESSMENT: All major requirements addressed with specific recommendations
                    DECISION: TERMINATE
                    
                    {{{{$conversation}}}}
                """,
                )
            )

            async def should_agent_terminate(self, agent, history):
                """Determine if the conversation should terminate based on critic evaluation."""

                self.iteration += 1
                self.logger.info(
                    f"Iteration: {self.iteration} of {self.maximum_iterations}"
                )

                # Force termination if maximum iterations reached
                if self.iteration >= self.maximum_iterations:
                    self.logger.info(
                        f"Maximum iterations ({self.maximum_iterations}) reached. Forcing termination."
                    )
                    return True

                # If this is the first iteration, don't terminate yet
                if self.iteration <= 1:
                    return False

                # Get the full conversation history to evaluate
                conversation = "\n\n".join(
                    [f"{msg.role}: {msg.content}" for msg in history]
                )

                # Check for consensus indicators in the last message
                last_message = history[-1].content
                consensus_indicators = [
                    "all requirements have been addressed",
                    "we have reached a complete solution",
                    "final recommendation:",
                    "agreed solution:",
                    "we've covered all aspects",
                ]

                if any(
                    indicator.lower() in last_message.lower()
                    for indicator in consensus_indicators
                ):
                    self.logger.info(
                        f"Consensus indicator detected in message. Terminating."
                    )
                    return True

                # Use the critic agent to evaluate the conversation
                arguments = KernelArguments()
                arguments["conversation"] = conversation
                evaluation = await self.kernel.invoke(
                    function=self.termination_function, arguments=arguments
                )
                self.logger.info(f"Critic Evaluation: {evaluation}")

                # Extract the score from the evaluation
                try:
                    score_match = re.search(r"SCORE:\s*(\d+\.?\d*)", str(evaluation))
                    if score_match:
                        score = float(score_match.group(1))
                        self.logger.info(f"Extracted evaluation score: {score}")

                        # Check if score meets termination threshold (8.0 or higher)
                        should_terminate = score >= 8.0
                    else:
                        self.logger.warning(
                            "Could not extract score from evaluation. Continuing conversation."
                        )
                        should_terminate = False

                    # Also check for explicit DECISION: TERMINATE
                    decision_match = re.search(
                        r"DECISION:\s*(TERMINATE|CONTINUE)", str(evaluation)
                    )
                    if decision_match and decision_match.group(1) == "TERMINATE":
                        self.logger.info("Critic explicitly recommended termination.")
                        should_terminate = True

                except ValueError as e:
                    self.logger.error(f"Error parsing evaluation score: {e}")
                    should_terminate = False

                self.logger.info(f"Should terminate: {should_terminate}")
                return should_terminate

        return CompletionTerminationStrategy(
            agents=agents, maximum_iterations=maximum_iterations
        )
 
    async def process_conversation(
        self, user_id, conversation_messages, maximum_iterations=10
    ):
        """
        Processes a conversation by orchestrating interactions between Cosmos DB specialist agents.

        Manages the entire conversation flow from initialization to response collection, uses OpenTelemetry
        for tracing, and provides status updates throughout the conversation.

        Args:
            user_id: Unique identifier for the user, used in session tracking.
            conversation_messages: List of dictionaries with role, name and content
                                representing the conversation history.
            maximum_iterations: Maximum number of conversation turns.

        Yields:
            Status updates during processing and the final response in JSON format.
        """
        try:
            # Create the agent group chat with specialized Cosmos DB agents
            self.agent_group_chat = self.create_agent_group_chat(
                agents_directory="agents/cosmos", maximum_iterations=maximum_iterations
            )

            # Extract user query
            user_query = None
            for msg in conversation_messages:
                if msg.get("role") == "user":
                    user_query = msg.get("content")

            if not user_query:
                self.logger.warning("No user query found in conversation messages")
                user_query = "Tell me about Azure Cosmos DB"

            # Format user message for add_chat_messages
            user_messages = [
                ChatMessageContent(
                    role=AuthorRole(m.get("role")),
                    name=m.get("name"),
                    content=m.get("content"),
                )
                for m in conversation_messages
                if m.get("role") == "user"
            ]

            # If we have any user messages, add them to the chat
            if user_messages:
                try:
                    await self.agent_group_chat.add_chat_messages(user_messages)
                    self.logger.info(
                        f"Added {len(user_messages)} user messages to chat"
                    )
                except Exception as e:
                    self.logger.warning(f"Error adding chat messages: {e}")

            # Setup OpenTelemetry tracing
            tracer = get_tracer(__name__)

            # Create a unique session ID for tracing purposes
            current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            session_id = f"{user_id}-{current_time}"

            # Track all messages exchanged during this conversation
            messages = []

            # List to store all debate messages for the response
            debate_transcript = []

            # Start the traced conversation session
            with tracer.start_as_current_span(session_id):
                # Initial status message
                yield "Evaluating your Cosmos DB requirements..."

                # Process each message in the conversation
                async for agent_message in self.agent_group_chat.invoke():
                    # Log the message
                    self.logger.info("Agent: %s", agent_message.to_dict())

                    # Add to messages collection
                    message_dict = agent_message.to_dict()
                    messages.append(message_dict)

                    # Add to debate transcript
                    debate_transcript.append(message_dict)

                    # Generate descriptive status for the client
                    next_action = await describe_next_action(
                        self.kernel, self.settings_utility, messages
                    )
                    self.logger.info("%s", next_action)

                    # Yield status update
                    yield f"{next_action}"

            # Get the final conversation
            response = list(
                reversed(
                    [item async for item in self.agent_group_chat.get_chat_messages()]
                )
            )

            # Find response from the last Cosmos DB specialist agent to speak
            # Create a list of all Cosmos agent names by checking agent prefixes
            cosmos_agents = [
                agent.name
                for agent in self.agent_manager.get_all_agents()
                if agent.name.startswith("Cosmos")
            ]

            specialist_responses = [r for r in response if r.name in cosmos_agents]

            if specialist_responses:
                # Use the most recent specialist response
                final_response = specialist_responses[0].to_dict()
            else:
                # Fallback to the last assistant message
                assistant_messages = [
                    r.to_dict() for r in response if r.role == "assistant"
                ]
                if assistant_messages:
                    final_response = assistant_messages[0]
                else:
                    # Ultimate fallback if no messages found
                    final_response = {
                        "role": "assistant",
                        "content": "I wasn't able to generate a complete response. Please try again with more specific requirements about Azure Cosmos DB.",
                    }

            # Add the transcript to the final response
            final_response["debate_transcript"] = debate_transcript

            # Final message is formatted as JSON to indicate the final response
            yield json.dumps(final_response)

        except Exception as e:
            # Log the error
            self.logger.error(f"Error in process_conversation: {str(e)}", exc_info=True)

            # Return a user-friendly error message
            error_response = {
                "role": "assistant",
                "content": "I encountered an issue while processing your request. Please try again with a more specific question about Azure Cosmos DB.",
                "error": str(e),
            }
            yield json.dumps(error_response)
 