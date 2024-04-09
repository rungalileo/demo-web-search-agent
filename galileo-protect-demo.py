import os

import galileo_protect as gp
import streamlit as st
from langchain.agents import AgentType, initialize_agent
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_community.tools import (
    ArxivQueryRun,
    DuckDuckGoSearchRun,
    Tool,
    WikipediaQueryRun,
)
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_openai import ChatOpenAI


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


st.set_page_config(
    page_title="Galileo Chat Bot",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="collapsed",
)

"# 🔭 Galileo Chat Bot"
search = DuckDuckGoSearchRun()
arxiv = ArxivQueryRun()
wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

llm = ChatOpenAI(temperature=0, openai_api_key=os.environ["OPENAI_API_KEY"])
prompt_template = "Write an essay for the topic provided by the user with the help of following content: {content}"
essay = LLMChain(llm=llm, prompt=PromptTemplate.from_template(prompt_template))

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
    ),
    Tool.from_function(
        func=essay.run,
        name="Essay",
        description="useful when you need to write an essay",
    ),
]
agent = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True)

with st.form(key="form"):
    user_input = ""

    user_input = st.text_input(
        "This is an AI search agent that has access to publications in Arxiv.org and Wikipedia.org. Tell this agent what do you want to know and it will find the answers for you to its best ability."
    )
    submit_clicked = st.form_submit_button("Submit")

output_container = st.empty()

if with_clear_container(submit_clicked):
    output_container = output_container.container()
    output_container.chat_message("user").write(user_input)

    answer_container = output_container.chat_message("assistant", avatar="🔭")
    st_callback = StreamlitCallbackHandler(answer_container)

    prompt = "Answer the user's question in helpful ways. Be helpful and honest. Question: {input}"
    input = user_input

    answer = agent.invoke(prompt.format(input=input), callbacks=[st_callback])

    answer_container.write(answer)

    response = gp.invoke(
        payload=answer,
        prioritized_rulesets=[
            {
                "rules": [
                    {
                        "metric": "pii",
                        "operator": "contains",
                        "target_value": "ssn",
                    },
                ],
                "action": {
                    "type": "OVERRIDE",
                    "choices": ["Sorry, I cannot answer that question."],
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
                    "choices": ["Sorry, I cannot answer that question."],
                },
            },
        ],
        timeout=10,
    )
    answer_container.write("Response from Galileo Protect:")
    answer_container.write(response.model_dump())
