## Importing necessary library
import os.path
import openai
### remove this pa
from credential import API_KEY

openai.api_key = API_KEY
from llama_index.core.schema import Document
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    ServiceContext,
    set_global_service_context,
)
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llmsherpa.readers import LayoutPDFReader
import streamlit as st


## Generating a specific llm api request for llm_follow_up_question with higher temperature
llm_follow_up_question = OpenAI(model="gpt-3.5-turbo", temperature=0.9, max_tokens=50)
embed_model = OpenAIEmbedding()
service_context_follow_up = ServiceContext.from_defaults(
    chunk_size=256, llm=llm_follow_up_question, embed_model=embed_model
)


## Building basic streamlit ui
st.set_page_config(
    page_title="Chat with the Chatbot",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)
st.title("Yash-Anchaliya-Info")
if "messages" not in st.session_state.keys():  # Initialize the chat messages history
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me a question about Yash-Anchaliya Resume !",
        }
    ]


## load data and model for answer generation
@st.cache_resource(show_spinner=False)
def load_data_and_model():

    with st.spinner(
        text="Loading and indexing the Streamlit docs – hang tight! This should take 1-2 minutes."
    ):
        llm_answer = OpenAI(model="gpt-3.5-turbo", temperature=0.0, max_tokens=200)
        service_context = ServiceContext.from_defaults(
            chunk_size=256, llm=llm_answer, embed_model=embed_model
        )

        set_global_service_context(service_context)
        ## Creating an persistant index storage 
        ##------------------------ Your path should be added------------------------------- maske sure storage is there##
        ####-------------------------f"{path}/storage"--------------------------
        if not os.path.exists("/Users/yashanchaliya/Desktop/onebyzero/storage"):
            llmsherpa_api_url = "https://readers.llmsherpa.com/api/document/developer/parseDocument?renderFormat=all"
            pdf_url = "Yash-Anchaliya.pdf"
            pdf_reader = LayoutPDFReader(llmsherpa_api_url)
            doc = pdf_reader.read_pdf(pdf_url)
            index = VectorStoreIndex([])
            for chunk in doc.chunks():
                index.insert(Document(text=chunk.to_context_text(), extra_info={}))
            # store it for later
            index.storage_context.persist()
        else:
            # load the existing index
            storage_context = StorageContext.from_defaults(
        ##------------------------ Your path should be added-------------------------------##
                persist_dir="/Users/yashanchaliya/Desktop/onebyzero/storage/"
            )
            index = load_index_from_storage(storage_context)

    return index


index = load_data_and_model()



if "chat_engine" not in st.session_state.keys():  
    # Initialize the chat engine for quesstion answering and query engine for follow-up question
    st.session_state.chat_engine_1 = index.as_chat_engine(
        chat_mode="context", verbose=True, similarity_top_k=3
    )
    st.session_state.query_engine = index.as_query_engine(
        ServiceContext=service_context_follow_up
    )
if prompt := st.chat_input(
    "Your question"
):  # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
# Display the prior chat messages
for message in st.session_state.messages:  
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Clearing chat-history
def clear_chat_history():
    st.session_state.chat_engine_1.reset()
    st.session_state.messages = [
        {"role": "assistant", "content": "How may I assist you today?"}
    ]


st.sidebar.button("Clear Chat History", on_click=clear_chat_history)

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            ###Given a user prompt It will generate answers
            response = st.session_state.chat_engine_1.chat(prompt)
            ## Displaying this on app
            st.write(response.response)

            ### Given this prompt I want to generate follow-up question
            follow_up_question = st.session_state.query_engine.query(
                f"Generate followup question based upom {prompt} "
            )
            
            ### changing the widget so that question can be viewed in different color
            colored_response = (
                f'<span style="color:#FF0000;">{follow_up_question.response}</span>'
            )
            # Display the colored response
            st.write(colored_response, unsafe_allow_html=True)
            message = {"role": "assistant", "content": response.response}
            
            ### append all the answers
            st.session_state.messages.append(message)  # Add response to message history
