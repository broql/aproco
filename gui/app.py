import json
import os
import streamlit as st
import datetime

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


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


def on_like():
    with open('data/qa.json', 'a') as f:
        qa_json = {
            'question': st.session_state.query,
            'answer': st.session_state.answer,
            'sources': st.session_state.sources,
            'feedback': 1,
            'timestamp': datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        }
        f.write(json.dumps(qa_json))
        f.write('\n')

    st.session_state.query = ''
    st.session_state.answer = ''
    st.session_state.sources = []

    st.success('Thank you for your feedback!')


def on_dislike():
    with open('data/qa.json', 'a') as f:
        qa_json = {
            'question': st.session_state.query,
            'answer': st.session_state.answer,
            'sources': st.session_state.sources,
            'feedback': -1,
            'timestamp': datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        }
        f.write(json.dumps(qa_json))
        f.write('\n')

    st.session_state.query = ''
    st.session_state.answer = ''
    st.session_state.sources = []

    st.success('Thank you for your feedback!')


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

[data-testid="stVerticalBlock"] div:nth-child(4) > [data-testid="stVerticalBlock"] .stMarkdown h3,
[data-testid="stVerticalBlock"] div:nth-child(3) > [data-testid="stVerticalBlock"] .stMarkdown h3 {
padding: 0;
color: #244250;
text-align: center;
font-size: 16px;
font-style: normal;
font-weight: 400;
line-height: normal;    
margin-bottom: 20px;          
}           

[data-testid="stVerticalBlock"] div:nth-child(4) > [data-testid="stVerticalBlock"],
[data-testid="stVerticalBlock"] div:nth-child(3) > [data-testid="stVerticalBlock"] {
border-radius: 10px;
background: rgba(214, 231, 231, 0.54);
padding: 20px;
margin: 20px 0;
}

[data-testid="stVerticalBlock"] div:nth-child(4) > [data-testid="stVerticalBlock"] [data-baseweb="input"] > [data-baseweb="base-input"],                        
[data-testid="stVerticalBlock"] div:nth-child(3) > [data-testid="stVerticalBlock"] [data-baseweb="input"] > [data-baseweb="base-input"] {
background: white;            
}

[data-testid="stVerticalBlock"] > div:nth-child(4) button[kind="secondary"],
[data-testid="stVerticalBlock"] > div:nth-child(3) button[kind="secondary"] {
border-radius: 10px;
background: #34BFC3;   
color: white;                              
}            
            
[data-testid="stExpander"] [data-baseweb="accordion"] {
border: 0;            
}            

</style>
""", unsafe_allow_html=True)

# st.image(Image.open('gui/images/aproco.png'), width=150)
# st.image(Image.open('gui/images/nd.png'), width=80)


# st.markdown("# Knowledge Assistant")

# st.expander('User Manual', expanded=False).markdown("""
# **Welcome to our test site!**

# The project aim is to test the possibility of an AI driven assistant for knowledge management in the ever changing regulations environment.

# We have uploaded a list of documents from FDA and EU regulations (list available underneath) for it to digest and be able to respond to questions related to these topics.

# Please feel free to ask any questions, but given it's a proof of concept, it's not perfect - partially for a reason. We want to see how you interact with the model and adjust accordingly.

# There are some tips on how to get more relevant information:
# - Give context. For example:
#     - “What is the requirement for cleaning procedure?”
#         - This question might not get the best results, as we have purposefully uploaded more documents that cover different areas, to see how it's gonna work.
#     - “What is the requirement for cleaning procedure in [some situation] according to FDA regulations?”
#         - This question should get much better results.
# - In the near future, the model will try to determine what it might need to ask to get a better understanding of what's expected from it, but it's not there yet.
# - The more specific you are, the better the result.
# - If the model cannot find a specific answer in source documents, it should say it doesn't know - not hallucinate. In that case, please try to rephrase the question.
# - Other general tips for using ChatGPT or similar tools apply here as well, as it's also based on LLM (Large Language Model) and works in a similar way.

# Please keep in mind that we keep a log of questions and answers for evaluation purposes, as the tool is under development and we want to keep improving it and test on the same questions, in case it produces a bad result. These are in no way associated with a user/computer/IP.

# In case of any questions, please contact us directly: nd@aproco.io
# """)

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

chat_name = "no-dev"
documents_path = "data/documents/nd/"
system_template = r"""
Use the following pieces of context to answer the question below. If you don't find the information required to answer in the context, just say you don't know - don't try to make up an answer.
--- context ---
{context}
--- context ---
"""
user_template = r""" 
{question}
"""
messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template(user_template)
]

PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(messages)

# vectorstore
if not os.path.exists(f"data/db/{chat_name}"):
    documents = []
    for filename in os.listdir(documents_path):
        if filename.endswith('.pdf'):
            loader = PyPDFLoader(os.path.join(documents_path, filename))
            documents.extend(loader.load())

    # splitter
    # TODO: use better splitter
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
    doYouLikeMyAnswer = feedback[0].text("Do you like my answer?")
    like = feedback[1].button("👍", on_click=on_like)
    dislike = feedback[2].button("👎", on_click=on_dislike)

    st.markdown("### Sources")

    for doc in response['source_documents']:
        st.markdown(
            f"- {doc.metadata['source'].split('/')[-1].replace('.pdf', '')}")

    with open('data/qa.json', 'a') as f:
        qa_json = {
            'question': query,
            'answer': response['answer'],
            'sources': [doc.metadata['source'].split('/')[-1].replace('.pdf', '') for doc in response['source_documents']],
            'timestamp': datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        }
        f.write(json.dumps(qa_json))
        f.write('\n')


st.markdown("""---""")
expander = st.expander("All source documents in the database", expanded=False)

list_of_files = []
for f in os.listdir(documents_path):
    if f.endswith('.pdf'):
        list_of_files.append(f.replace('.pdf', ''))

list_of_files.sort()
for file in list_of_files:
    expander.markdown(f"- {file}")
