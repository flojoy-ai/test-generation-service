__all__ = ["gpt_agent_exec"]

import json
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain import hub
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_xml_agent, create_openai_tools_agent
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.agents import AgentFinish
from langchain.agents.output_parsers.openai_tools import OpenAIToolAgentAction
from dotenv import load_dotenv

load_dotenv()

"""
CREATE VECTORE STORE CLIENT FOR DOCUMENT RETRIEVAL
"""
embeddings = OpenAIEmbeddings()
docsearch = PineconeVectorStore(
    index_name="hardware-libraries-embedding", embedding=embeddings
)


"""
CREATE PROMPT TEMPLATE FOR ANTHROPIC AGENT
"""
prompt = ChatPromptTemplate.from_template(
    """
    You are a world class coder. You generate python assertion tests only. Help the user answer any questions.

    You have access to the following tools:

    {tools}

    In order to use a tool, you can use <tool></tool> and <tool_input></tool_input> tags. You will then get back a response in the form <observation></observation>
    For example, if you have a tool called 'search' that could run a google search, in order to search for the weather in SF you would respond:

    <tool>search</tool><tool_input>weather in SF</tool_input>
    <observation>64 degrees</observation>

    When you are done, respond with a final answer between <final_answer></final_answer>. For example:

    <final_answer>The weather in SF is 64 degrees</final_answer>

    If you generate code and user input is required, create placeholders only where a user input is necessary. The placeholder is of the form $$$Placeholder Description$$$. Basically,
    it contains text that describes what the value is used for an it's wrapped with $$$.

    For example:
    <example>
    user_age = $$$The age of the user to be printed$$$
    </example>

    Another example:
    <example>
    import some_connection_library
    device_name = "$$$The name of the device to connect to using some_connection_library$$$"
    some_connection_library.connect(device_name)
    </example>

    Remember, you generate python assertion tests only. In other words, python tests that test stuff using the "assert" keyword.
    <example>
    user = $$$User name$$$
    assert user == "Bob" //test if user's name is Bob 
    </example>

    Begin!

    Previous Conversation:
    {chat_history}

    Question: {input}
    {agent_scratchpad}
    """
)
prompt = prompt.partial(chat_history="")

"""
CREATE TOOLS 
"""
# Use claude 3 for code generation and have it available as a tool
retriever = docsearch.as_retriever(search_kwargs={"k": 10})
retriever_tool = create_retriever_tool(
    retriever=retriever,
    name="hardware-libraries-retriever",
    description="USE ONLY IF you really need to retrieve relevant documentation for hardware libraries. These include: tektronix, pyserial, pyvisa, nidaqmx",
)

tools = [retriever_tool]

llm = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0)  # type: ignore

agent = create_xml_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools)  # type: ignore


class CodeGenerateArgsSchema(BaseModel):
    prompt: str = Field(
        description="The prompt to generate code from",
    )


@tool("code-generation-tool", args_schema=CodeGenerateArgsSchema)
def generate_code(prompt: str):
    """Use this tool to generate code to the given user prompt or input"""
    return agent_executor.invoke({"input": prompt})


# define the schema for the final output
@tool
class Response(BaseModel):
    """Finally, use this tool as response to return to the user"""

    code: str = Field(description="This field contains the code snippet")
    reasoning: str = Field(
        description="This field contains the reasoning behind the code"
    )


# use gpt-3 to run the agent and decide which tool to use
gpt_tools = [generate_code, Response]


"""
CREATE GPT AGENT TO DECIDE WHICH STEPS TO TAKE TO COMPLETE USER PROMPT
"""
gpt_llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
gpt_prompt = (
    hub.pull("hwchase17/openai-functions-agent")
    + "use the Response tool if output with code"
)


# hacky, but no other way. Return model output as the final output from the agent
def parser(output):
    if (
        isinstance(output, list)
        and isinstance(output[0], OpenAIToolAgentAction)
        and output[0].tool == "Response"
    ):
        output = output[0]
        if isinstance(output.tool_input, str):  # if it's a string, convert to json
            output.tool_input = json.loads(output.tool_input)
        return AgentFinish(return_values=output.tool_input, log=str(output.tool))
    return output


agent = create_openai_tools_agent(gpt_llm, gpt_tools, gpt_prompt) | parser

gpt_agent_exec = AgentExecutor(agent=agent, tools=gpt_tools, verbose=True)  # type: ignore
