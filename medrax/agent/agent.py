import operator
from typing import List, Dict, Any, TypedDict, Annotated, Optional
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage, HumanMessage
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AnyMessage, SystemMessage
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool

_ = load_dotenv()


class AgentState(TypedDict):
    """
    A TypedDict representing the state of an agent.

    Attributes:
        messages (Annotated[List[AnyMessage], operator.add]): A list of messages
            representing the conversation history. The operator.add annotation
            indicates that new messages should be appended to this list.
    """

    messages: Annotated[List[AnyMessage], operator.add]


class Agent:
    """
    A class representing an agent that processes requests and executes tools based on
    language model responses with parallel tool execution capabilities.

    Attributes:
        model (BaseLanguageModel): The language model used for processing.
        tool_node (ToolNode): The parallel tool execution node.
        checkpointer (Any): Manages and persists the agent's state.
        system_prompt (str): The system instructions for the agent.
        workflow (StateGraph): The compiled workflow for the agent's processing.
    """

    def __init__(
        self,
        model: BaseLanguageModel,
        tools: List[BaseTool],
        checkpointer: Any = None,
        system_prompt: str = "",
    ):
        """
        Initialize the Agent.

        Args:
            model (BaseLanguageModel): The language model to use.
            tools (List[BaseTool]): A list of available tools.
            checkpointer (Any, optional): State persistence manager. Defaults to None.
            system_prompt (str, optional): System instructions. Defaults to "".
        """
        self.system_prompt = system_prompt

        # Create the parallel tool execution node
        self.tool_node = ToolNode(tools)

        # Define the agent workflow with parallel tool execution
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", self.process_request)
        workflow.add_node("tools", self.tool_node)
        workflow.add_conditional_edges("agent", self.has_tool_calls, {True: "tools", False: END})
        workflow.add_edge("tools", "agent")
        workflow.set_entry_point("agent")

        self.workflow = workflow.compile(checkpointer=checkpointer)
        self.model = model.bind_tools(tools)

    def process_request(self, state: AgentState) -> Dict[str, List[AnyMessage]]:
        """
        Process the request using the language model.

        Args:
            state (AgentState): The current state of the agent.

        Returns:
            Dict[str, List[AnyMessage]]: A dictionary containing the model's response.
        """
        messages = state["messages"]
    
        # Only add system prompt if it's not already present (i.e., first call in this conversation)
        # This avoids redundantly sending the system prompt on every model invocation
        if self.system_prompt and (len(messages) == 0 or not isinstance(messages[0], SystemMessage)):
            messages = [SystemMessage(content=self.system_prompt)] + messages
        
        # Check if we just executed tools by checking if the last message is a ToolMessage
        # This indicates we're in the process node immediately after the execute node
        has_tool_results = len(messages) > 0 and isinstance(messages[-1], ToolMessage)

        # If we have tool results, add explicit instruction to continue reasoning
        # This is especially important for models like Gemini that may stop without generating output
        # Use HumanMessage instead of SystemMessage as it's more compatible with all models
        # The prompt allows the model to call more tools OR provide final answer, giving it flexibility
        if has_tool_results:
            synthesis_prompt = HumanMessage(
                content="Review the tool results above. If you need more information, you can call additional tools. "
                "Otherwise, provide your complete final answer synthesizing all the information."
            )
            messages = messages + [synthesis_prompt]
        
        response = self.model.invoke(messages)
        return {"messages": [response]}

    def has_tool_calls(self, state: AgentState) -> bool:
        """
        Check if the response contains any tool calls.

        Args:
            state (AgentState): The current state of the agent.

        Returns:
            bool: True if tool calls exist, False otherwise.
        """
        response = state["messages"][-1]
        return len(response.tool_calls) > 0
