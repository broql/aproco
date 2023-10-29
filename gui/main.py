import os
import datetime

import streamlit as st
import pandas as pd

from PIL import Image
from dotenv import load_dotenv

from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate

load_dotenv()

regulations = ['EU', 'FDA', 'PICS']


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


def save_query(file='data/qa.csv', feedback=None):
    row = pd.DataFrame({
        "Query": query,
        "Answer": response['answer'],
        "Sources": ', '.join([doc.metadata['source'].split('/')[-1].replace('.pdf', '') for doc in response['source_documents']]),
        "Timestamp": datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
        "Feedback": feedback
    }, index=[0])

    if not os.path.exists(file):
        row.to_csv(file, index=False, sep=';')
    else:
        row.to_csv(file, mode='a', header=False, index=False, sep=';')


def reset_view():
    st.session_state.query = ''
    st.session_state.answer = ''
    st.session_state.sources = []


def on_like():
    save_query(feedback=1)
    reset_view()


def on_dislike():
    save_query(feedback=-1)
    reset_view()


st.markdown("""
<style>
body {
color: #244250;
font-family: Roboto;
font-size: 16px;
font-style: normal;
font-weight: 400;
line-height: normal;
}

[data-testid="block-container"] {
border-radius: 10px;
position: relative;
top: 50px;
padding: 3rem;
}

[data-testid="stSidebar"] [data-testid="block-container"] {
padding: 0;
}

.stMarkdown h1 {            
color: #244250;
text-align: center;
font-size: 40px;
font-style: normal;
font-weight: 900;
line-height: normal;
}
            
.stMarkdown h1:after {              
content: "beta";
border-radius: 5px;
background: #C02025;
position: absolute;
color: white;
display: inline-flex;
padding: 4px 8px;
justify-content: center;
align-items: center;
font-size: 16px;
font-weight: normal;     
top: 10px;
right: 80px;                           
}
            
.stMarkdown h2 {
padding: 0;
color: #244250;
text-align: center;
font-size: 24px;
font-style: normal;
font-weight: 600;
line-height: normal;                  
}            

[data-testid="stVerticalBlock"] div:nth-child(5) > [data-testid="stVerticalBlock"] .stMarkdown h3,
[data-testid="stVerticalBlock"] div:nth-child(4) > [data-testid="stVerticalBlock"] .stMarkdown h3 {
padding: 0;
color: #244250;
text-align: center;
font-size: 16px;
font-style: normal;
font-weight: 400;
line-height: normal;    
margin-bottom: 20px;          
}                      

[data-testid="stVerticalBlock"] div:nth-child(5) > [data-testid="stVerticalBlock"],
[data-testid="stVerticalBlock"] div:nth-child(4) > [data-testid="stVerticalBlock"] {
border-radius: 10px;
background: rgba(214, 231, 231, 0.54);
padding: 20px;
margin: 20px 0;
}

[data-testid="stVerticalBlock"] div:nth-child(5) > [data-testid="stVerticalBlock"] [data-baseweb="input"] > [data-baseweb="base-input"],                        
[data-testid="stVerticalBlock"] div:nth-child(4) > [data-testid="stVerticalBlock"] [data-baseweb="input"] > [data-baseweb="base-input"] {
background: white;            
}

[data-testid="stVerticalBlock"] > div:nth-child(5) button[kind="secondary"],
[data-testid="stVerticalBlock"] > div:nth-child(4) button[kind="secondary"] {
border-radius: 10px;
background: #34BFC3;   
color: white;                              
}            
            
[data-testid="stExpander"] [data-baseweb="accordion"] {
border: 0;            
}            

[data-testid="stAppViewContainer"] > section > [data-testid="block-container"] > div > [data-testid="stVerticalBlock"] > div:nth-child(2) {
position: fixed;
left: 30px;
top: 10px;
}
                      
[data-testid="stAppViewContainer"] > section > [data-testid="block-container"] > div > [data-testid="stVerticalBlock"] > div:nth-child(2) img {
position: absolute;
top: 40px;           
}            

[data-testid="stAppViewContainer"] > section > [data-testid="block-container"] > div > [data-testid="stVerticalBlock"] > div:nth-child(2) > div > div:nth-child(1) [data-testid="stImage"]:after {                        
position: absolute;
content: "&";
left: 54px;
top: 59px;
display: inline-flex;              
}
                       
[data-testid="stAppViewContainer"] > section > [data-testid="block-container"] > div > [data-testid="stVerticalBlock"] > div:nth-child(2) > div > div:nth-child(2) {            
left: 75px;            
}

@media screen and (max-width: 460px) {
[data-testid="stVerticalBlock"] div:nth-child(5) > [data-testid="stVerticalBlock"] .stMarkdown h3 span,
[data-testid="stVerticalBlock"] div:nth-child(4) > [data-testid="stVerticalBlock"] .stMarkdown h3 span {
margin-left: 10px;
font-size: 14px;
}
            
.stMarkdown h2 span {
margin-left: 10px;
font-size: 24px;           
}       

.stMarkdown h1:after {     
top: 70px;
right: 8px;
}                  
}
            
[data-testid="stSidebar"] {
background: #D6E7E7;
}

</style>
""", unsafe_allow_html=True)

with st.container():
    st.image(Image.open('gui/images/nd.png'), width=50)
    st.image(Image.open('gui/images/aproco.png'), width=120)

chat_name = st.sidebar.selectbox(
    'Regulations', regulations, key='chat_name')
chat_name += '_Documents'

st.title("Knowledge Assistant")

with st.container():
    st.header("Hi! Ask me a question!")
    st.subheader("I'm still learning but I'll try to do my best!")

    col1, col2 = st.columns([0.88, 0.12])

    with col1:
        query = st.text_input(
            label='Query', label_visibility='collapsed', key='query')

    with col2:
        ask_button = st.button("Send")

documents_path = dict()
for regulation in regulations:
    key = f"{regulation}_Documents"
    documents_path[key] = f"data/documents/{key}/"

with open('prompt.txt', 'r') as f:
    system_template = f.read()

user_template = r"""
{question}
"""
messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template(user_template)
]

PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(messages)

if not os.path.exists(f"data/db/{chat_name}"):
    documents = []

    for root, dirs, files in os.walk(documents_path[chat_name]):
        for file in files:
            if file.endswith('.pdf'):
                loader = PyPDFLoader(os.path.join(root, file))
                documents.extend(loader.load())

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    print("Creating new vectorstore")
    vectorstore = Chroma.from_documents(texts, OpenAIEmbeddings(
        deployment='embeddings', chunk_size=16), collection_name=chat_name, persist_directory=f"data/db/{chat_name}")
    vectorstore.persist()
else:
    print("Loading existing vectorstore")
    vectorstore = Chroma(collection_name=chat_name, embedding_function=OpenAIEmbeddings(
        deployment='embeddings', chunk_size=16), persist_directory=f"data/db/{chat_name}")

if query:
    st.markdown("### Answer")

    chat_box = st.empty()
    stream_handler = StreamHandler(chat_box)

    qa = ConversationalRetrievalChain.from_llm(
        AzureChatOpenAI(deployment_name='llm', model_name="gpt-4",
                        temperature=0, streaming=True, callbacks=[stream_handler]),
        vectorstore.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True,
        verbose=False,
        combine_docs_chain_kwargs={'prompt': PROMPT_TEMPLATE})

    response = qa({'question': query, 'chat_history': []})

    st.session_state.answer = response['answer']
    st.session_state.sources = [doc.metadata['source'].split(
        '/')[-1].replace('.pdf', '') for doc in response['source_documents']]

    feedback = st.columns([0.86, 0.07, 0.07])
    like = feedback[1].button("👍", on_click=on_like)
    dislike = feedback[2].button("👎", on_click=on_dislike)

    st.markdown("### Sources")

    for doc in response['source_documents']:
        st.markdown(
            f"- {doc.metadata['source'].split('/')[-1].replace('.pdf', '')}")

    save_query()


expander = st.sidebar.expander(
    "All source documents in the database", expanded=False)

list_of_files = []
for root, dirs, files in os.walk(documents_path[chat_name]):
    for f in files:
        if f.endswith('.pdf'):
            list_of_files.append(f.replace('.pdf', '').replace('_', ' '))

list_of_files.sort()
for file in list_of_files:
    expander.markdown(f"- {file}")
