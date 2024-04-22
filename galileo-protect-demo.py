from langchain_community.tools import (
    Tool,
    DuckDuckGoSearchRun,
    ArxivQueryRun,
    WikipediaQueryRun,
)
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.callbacks import StreamlitCallbackHandler
import os
import streamlit as st
import galileo_protect as gp
from galileo_observe import GalileoObserveCallback


# A hack to "clear" the previous result when submitting a new prompt. This avoids
# the "previous run's text is grayed-out but visible during rerun" Streamlit behavior.
class DirtyState:
    NOT_DIRTY = "NOT_DIRTY"
    DIRTY = "DIRTY"
    UNHANDLED_SUBMIT = "UNHANDLED_SUBMIT"


def get_dirty_state() -> str:
    return st.session_state.get("dirty_state", DirtyState.NOT_DIRTY)


def set_dirty_state(state: str) -> None:
    st.session_state["dirty_state"] = state


def with_clear_container(submit_clicked):
    if get_dirty_state() == DirtyState.DIRTY:
        if submit_clicked:
            set_dirty_state(DirtyState.UNHANDLED_SUBMIT)
            st.rerun()
        else:
            set_dirty_state(DirtyState.NOT_DIRTY)

    if submit_clicked or get_dirty_state() == DirtyState.UNHANDLED_SUBMIT:
        set_dirty_state(DirtyState.DIRTY)
        return True

    return False


monitor_handler = GalileoObserveCallback(project_name='demo-galileo-protect')

# metrics = [
#    Scorers.context_adherence,
#    Scorers.completeness_gpt,
#    Scorers.prompt_perplexity,
#    Scorers.pii,
#    Scorers.chunk_attribution_utilization_gpt
#    ]
#
# pq.login("https://console.demo.rungalileo.io")
# If you don't have your GALILEO_USERNAME and GALILEO_PASSWORD exported, login


# galileo_handler = pq.GalileoPromptCallback(
#     project_name='sg_chatdemo_1', scorers=metrics, run_name='run_sn3'
#     # Make sure the run_name is unique across runs
# )

st.set_page_config(
    page_title="Galileo Chat Bot",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="collapsed",
)

"# 🔭 Galileo Chat Bot"
user_openai_api_key = os.environ["OPENAI_API_KEY"]
# Looks for openai_api_key

if user_openai_api_key:
    openai_api_key = user_openai_api_key
    enable_custom = True
else:
    openai_api_key = "not_supplied"
    enable_custom = False

search = DuckDuckGoSearchRun()
arxiv = ArxivQueryRun()
wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

llm = ChatOpenAI(temperature=0, openai_api_key=os.environ["OPENAI_API_KEY"])

tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events.",
    ),
    Tool(
        name="Arxiv",
        func=arxiv.run,
        description="useful when you need an answer about encyclopedic general knowledge",
    ),
    Tool(
        name="Wikipedia",
        func=wiki.run,
        description="useful when you need an answer about encyclopedic general knowledge",
    )
]
agent = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True)

with st.form(key="form"):
    user_input = ""

    if enable_custom:
        user_input = st.text_input(
            "This is an AI search agent that has access to the open web. Tell this agent what do you want to know and it will find the answers for you to its best ability."
        )
    submit_clicked = st.form_submit_button("Submit")

output_container = st.empty()

if with_clear_container(submit_clicked):
    output_container = output_container.container()
    output_container.chat_message("user").write(user_input)

    answer_container = output_container.chat_message("assistant", avatar="🔭")
    st_callback = StreamlitCallbackHandler(answer_container)

    prompt = "Answer the user's question using the tools provided. For successful task completion: Consider user's question and determine which search tool is best suited based on its capabilities. Be helpful and honest. Question: {input}"
    input = user_input

    answer = agent.invoke(prompt.format(input=input), callbacks=[st_callback,monitor_handler])
    
    answer_container.write(f"**Response from the model:**")
    answer_container.write(answer['output'])

    response = gp.invoke(
        payload=answer,
        prioritized_rulesets=[
            {
                "rules": [
                    {
                        "metric": "pii",
                        "operator": "contains",
                        "target_value": "address",
                    },
                ],
                "action": {
                    "type": "OVERRIDE",
                    "choices": [
                        "Personal address detected in the model output. Sorry, I cannot answer that question."
                    ],
                },
            },
            {
                "rules": [
                    {
                        "metric": "input_toxicity",
                        "operator": "gte",
                        "target_value": 0.9,
                    },
                ],
                "action": {
                    "type": "OVERRIDE",
                    "choices": [
                        "Toxicity detected in the user's prompt. Sorry, I cannot answer that question."
                    ],
                },
            },
            {
                "rules": [
                    {
                        "metric": "prompt_injection",
                        "operator": "eq",
                        "target_value": "impersonation",
                    },
                ],
                "action": {
                    "type": "OVERRIDE",
                    "choices": [
                        "Prompt injection detected in the user's prompt. Sorry, I cannot answer that question."
                    ],
                },
            },
            {
                "rules": [
                    {
                        "metric": "prompt_injection",
                        "operator": "eq",
                        "target_value": "new_context",
                    },
                ],
                "action": {
                    "type": "OVERRIDE",
                    "choices": [
                        "Prompt injection detected in the user's prompt. Sorry, I cannot answer that question."
                    ],
                },
            },
        ],
        stage_id="b67f362f-126e-45c6-a78f-35ce85098a79",
        timeout=10,
    )
    answer_container.write(f"**Response from Galileo Protect:**")
    answer_container.write(response.text)

    st.button('Integration details')
    st.write(response.model_dump())
