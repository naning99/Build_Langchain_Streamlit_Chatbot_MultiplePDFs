# coding=utf-8

raw_text = """
About Us: We are a Global Group AI team that empowers organization with citizen data scientists and enforces governance to ensure responsible and ethical use of AI. Our team is responsible for developing advanced analytics and machine learning models, and delivering data-driven insights to business leaders across the organization.
Job Description: As a Junior Data Scientist, you will be responsible for creating and implementing complex machine learning models, performing data analysis, and driving insights from large datasets. You will work with stakeholders across the organization to understand their business needs and translate them into data-driven solutions. You will also work closely with citizen data scientists to help them leverage data and build models for their business problems.
Key Responsibilities:
• Develop and implement machine learning models to solve complex business problems, using techniques such as linear regression, logistic regression, decision trees, and other supervised and unsupervised learning models.
• Perform data analysis and create insights from large datasets.
• Work with stakeholders across the organization to understand business needs and translate
them into data-driven solutions.
• Collaborate with citizen data scientists to empower them to leverage data and build models
for their business problems.
• Enforce governance to ensure responsible and ethical use of data.
• Keep up-to-date with the latest developments in machine learning and data science.
Requirements:
• Bachelor's or Master's degree in Computer Science, Statistics, Mathematics, Actuarial Science or related fields.
• At least 1-4 years of experience in data science or related field.
• Strong knowledge of machine learning algorithms and statistical modelling techniques,
especially using Scikit-learn.
• Proficiency in Python, PySpark and SQL.
• Experience with data visualization and reporting tools (e.g. Power BI, QlikSense).
• Excellent communication and problem-solving skills.
• Experience with GenAI solutions and prompt engineering
Nice-to-haves:
• Familiarity with actuarial methods and models.
• Experience in MLOps and data pipelines eg, Bitbucket, Artifactory, Jenkins
• Familiarity with deep learning frameworks (e.g. TensorFlow, PyTorch).
• Experience with cloud-based services (e.g. AWS, Azure, GCP).
• Knowledge of big data technologies (e.g. Hadoop, Spark).
If you're interested in this position, please send your resume and cover letter to [Global Group AI team email address]. We look forward to hearing from you!

"""

import streamlit as st 
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS 
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplate import css, bot_template, user_template
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain import hub
from langchain_core.output_parsers import StrOutputParser

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text   

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size = 800,
        chunk_overlap = 200,
        length_function = len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorestore(text_chunks):
    embeddings = OpenAIEmbeddings(base_url="https://api.bianxie.ai/v1")
    #embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorsore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorsore

# def get_conversation_chain(vectorstore, user_question):
#     llm = ChatOpenAI(base_url="https://api.bianxie.ai/v1")
#     memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
#     conversation_chain = ConversationalRetrievalChain.from_llm(
#         llm=llm,
#         retriever=vectorstore.as_retriever(),
#         memory=memory
#     )
#     response = conversation_chain.invoke({'question': user_question})
#     return response


def get_conversation_chain(vectorstore, user_question):
    llm = ChatOpenAI(base_url = "https://api.bianxie.ai/v1")
    conversation_chain = RunnableWithMessageHistory(
        llm = llm,
        input_messages_key="question",
        history_messages_key="history"
    )
    response = conversation_chain.invoke({'question': user_question})
    return response
       
load_dotenv()
text_chunks = get_text_chunks(raw_text)
vectorstore = get_vectorestore(text_chunks)
user_question = "what is this pdf about?"
#response = get_conversation_chain(vectorstore, user_question)
#print(response)

prompt = hub.pull("rlm/rag-prompt")
print(prompt)
print("\n\n")

llm = ChatOpenAI(base_url="https://api.bianxie.ai/v1")

rag_chain = (
    {"context": vectorstore.as_retriever(), "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

response = rag_chain.invoke(user_question)
print(response)

    