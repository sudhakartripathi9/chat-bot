import tqdm as notebook_tqdm
import streamlit as st
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI # for load the google gemini model

from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

load_dotenv()
genai.configure(api_key=os.getenv('GOOGLE_AIP_KEY'))

def get_pdf_text(pdfs):
    text = ''
    for pdf in pdfs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text = text + page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50
    )
    chunks = text_splitter.split_text(text)

    return chunks

def get_vector(text_chunks):
    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectors = FAISS.from_texts(text_chunks,embedding=embedding)
    vectors.save_local('faiss_index')
    return vectors


def get_conversational_chain():
    prompt_template = '''
    Answer the question from provided context, make sure to give proper correct answer, don't give wrong answer.
    if answer is not available for given question just give the output "Answer is not available for given input \n\n"
    Context : \n {context}? \n
    Question : \n {question} \n

    Answer : 
    '''

    model = ChatGoogleGenerativeAI(model='gemini-1.5-flash',temperature=0.3)
    prompt = PromptTemplate(template=prompt_template,input_variables=['context','question'])

    chain = load_qa_chain(model,chain_type = "stuff",prompt=prompt)
    return chain


def user_input(new_question):
    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local('faiss_index',embedding,allow_dangerous_deserialization = True)
    docs = vector_store.similarity_search(new_question)
    chain = get_conversational_chain()

    response = chain(
        {'input_documents':docs,'question':new_question,'context':docs},
        return_only_outputs=True
        )
    
    return response['output_text']

print(user_input('which model analized in rice dicease detector application?'))


def main():
    st.set_page_config("Chat PDF")
    st.header('Chat with our given PDF')

    question = st.text_input('Ask question from uploaded PDF file')
    bt = st.button('Enter')
    if question:
        if bt:
            output = user_input(question)
            st.write(output)

    with st.sidebar:
        st.title('Menu')
        pdf_file = st.file_uploader('Upload all PDF files',accept_multiple_files=True)
        button = st.button('Submit')
        if button and pdf_file:
            st.spinner('processiong......')
            raw_file = get_pdf_text(pdf_file)
            chunks = get_text_chunks(raw_file)
            vectors = get_vector(chunks)
            st.success('Done',icon="✅")

        elif button and not pdf_file:
            st.write('Error! upload the files and try again')

    



if __name__ == "__main__":
    main()

