import os
from typing import List

import streamlit as st
from galileo_observe import GalileoObserveCallback
from galileo_protect import OverrideAction, Ruleset
from galileo_protect.langchain import ProtectParser, ProtectTool
from langchain.schema.document import Document
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore


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
    page_title="Galileo's Chatbot",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="collapsed",
)

"# 🔭 Galileo's Chatbot"

user_openai_api_key = os.environ["OPENAI_API_KEY"]
if user_openai_api_key:
    openai_api_key = user_openai_api_key
    enable_custom = True
else:
    openai_api_key = "not_supplied"
    enable_custom = False


defined_rulesets = [
    Ruleset(
        rules=[
            {
                "metric": "input_toxicity",
                "operator": "gte",
                "target_value": 0.9,
            },
        ],
        action=OverrideAction(choices=["Sorry, toxicity detected in the user input. I cannot answer that question."]),
    ),
    Ruleset(
        rules=[
            {
                "metric": "prompt_injection",
                "operator": "eq",
                "target_value": "impersonation",
            },
        ],
        action=OverrideAction(
            choices=["Sorry, prompt injection detected in the user input. I cannot answer that question."]
        ),
    ),
    Ruleset(
        rules=[
            {
                "metric": "prompt_injection",
                "operator": "eq",
                "target_value": "new_context",
            },
        ],
        action=OverrideAction(
            choices=["Sorry, prompt injection detected in the user input. I cannot answer that question."]
        ),
    ),
]

output_rulesets = [
    Ruleset(
        rules=[
            {
                "metric": "adherence_nli",
                "operator": "lt",
                "target_value": 0.5,
            },
        ],
        action=OverrideAction(
            choices=["Sorry, hallucination detected in the model output. I cannot answer that question."]
        ),
    ),
    Ruleset(
        rules=[
            {
                "metric": "pii",
                "operator": "contains",
                "target_value": "address",
            },
        ],
        action=OverrideAction(
            choices=["Sorry, personal address detected in the model output. I cannot answer that question."]
        ),
    ),
]


gp_tool = ProtectTool(stage_id=os.environ["GALILEO_STAGE_ID"], prioritized_rulesets=defined_rulesets)

gp_tool_output = ProtectTool(stage_id=os.environ["GALILEO_STAGE_ID"], timeout=15, prioritized_rulesets=output_rulesets)


embeddings = OpenAIEmbeddings()
vectordb = PineconeVectorStore(index_name="galileo-demo", embedding=embeddings, namespace="sp500-qa-demo")
llm = ChatOpenAI(temperature=0, api_key=os.environ["OPENAI_API_KEY"])
# monitor_handler = GalileoObserveCallback(project_name='demo-galileo-protect')
monitor_handler = GalileoObserveCallback(project_name="observe-with-protect")


def format_docs(docs: List[Document]) -> str:
    return "\n\n".join([d.page_content for d in docs])


retriever = vectordb.as_retriever()

template = """You are a helpful assistant. Given the context below, please answer the following questions:

    {context}

    Question: {question}
    """
prompt = ChatPromptTemplate.from_template(template)
model = ChatOpenAI(name="gpt-3.5-turbo", temperature=0)

rag_chain_original = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | model | StrOutputParser()
)

gp_output_parser = ProtectParser(chain=StrOutputParser())

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | {"output": model | StrOutputParser(), "input": lambda x: x.to_string()}
    | gp_tool_output
    | gp_output_parser.parser
)

gp_exec = ProtectParser(chain=rag_chain, echo_output=True)

gp_chain = gp_tool | gp_exec.parser

with st.form(key="form"):
    user_input = ""

    if enable_custom:
        user_input = st.text_input("This is a helpful assistant. What do you want to know?")
    submit_clicked = st.form_submit_button("Submit")

output_container = st.empty()

if with_clear_container(submit_clicked):
    output_container = output_container.container()
    output_container.chat_message("user").write(user_input)

    answer_container = output_container.chat_message("assistant", avatar="🔭")
    st_callback = StreamlitCallbackHandler(answer_container)

    model_answer = rag_chain_original.invoke(user_input)
    gp_answer = gp_chain.invoke(user_input, config=dict(callbacks=[monitor_handler]))

    answer_container.write("**Response from the model:**")
    answer_container.write(model_answer)

    answer_container.write("**Response from Galileo Protect:**")
    answer_container.write(gp_answer)
