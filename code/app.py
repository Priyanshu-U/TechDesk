from langchain import PromptTemplate, HuggingFaceHub, LLMChain
from langchain import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import streamlit as st
from styles import css, bot_template, user_template
from PyPDF2 import PdfReader


def get_text(pdf):
    text = ""
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def get_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-xl')
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = OpenAI(openai_api_key="<>")
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm,
                                                               memory=memory,
                                                               retriever=vectorstore.as_retriever())

    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="Buy the electronics you DESERVE, get expert advice for your use case Here.",
                       page_icon="💻")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with our Expert Here!")
    col1, col2, col3 = st.columns(3)
    col2.image("Resources/Images/training.png", width=250)
    user_question = st.text_input("Ask for buying advice here!")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Start Conversation")
        pdf_docs = open('Manuals/guide1.pdf', mode='rb')
        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text = get_text(pdf_docs)
                text_chunks = get_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)

if __name__ == '__main__':
    main()
