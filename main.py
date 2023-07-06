import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from htmlTemplates import bot_template, user_template, css


def get_pdf_text(pdf_docs):
    text = ""
    for pdf_doc in pdf_docs:
        pdf_reader = PdfReader(pdf_doc)
        for pdf_page in pdf_reader.pages:
            text += pdf_page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_text(text=text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorStore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorStore


def get_conversation_chain(vectorStore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorStore.as_retriever(), memory=memory
    )
    return conversation_chain


def handle_user_question(question):
    response = st.session_state.conversation({"question": question})
    st.write(response)


def main():
    load_dotenv()

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about documents")
    if user_question:
        handle_user_question(user_question)
    st.write(css, unsafe_allow_html=True)
    st.write(user_template, unsafe_allow_html=True)
    st.write(bot_template, unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("your documents")
        pdf_docs = st.file_uploader("upload your pdfs", accept_multiple_files=True)
        if st.button("process"):
            with st.spinner("processing..."):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs=pdf_docs)

                # get the text chunks
                chunks = get_text_chunks(text=raw_text)

                # st.write(chunks)

                # create vector store
                vectorStore = get_vector_store(chunks)

                # create conversation chat
                st.session_state.conversation = get_conversation_chain(vectorStore)


if __name__ == "__main__":
    main()
