import streamlit as st
from dotenv import load_dotenv
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import google.generativeai as genai
from PyPDF2 import PdfReader
from huggingface_hub import login
import cassio
import os
import time
from langchain.llms.base import LLM
from typing import Any, List, Optional, Dict
from pydantic import Field, BaseModel
# import gdown

class GeminiLLM(LLM, BaseModel):
    model_name: str = Field(default="gemini-1.5-flash", description="The name of the Gemini model")
    model: Optional[Any] = Field(None, description="The GenerativeModel instance")

    def __init__(self, model_name: str, **data):
        super().__init__(model_name=model_name, **data)
        self.model = genai.GenerativeModel(model_name=self.model_name)

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self.model.generate_content(prompt)
        return response.text

    @property
    def _llm_type(self) -> str:
        return "gemini"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"model_name": self.model_name}
    
load_dotenv(Path(".env"))

st.set_page_config(page_title="Chat with Placy", layout="wide", )
st.title("Chat with Placy..!!")

# pdfreader = PdfReader("All About Shrirang.pdf")
pdfreader = PdfReader("A STUDENT GUIDE.pdf")


if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

if "faiss_vector_index" not in st.session_state:
    st.session_state.faiss_vector_index = None


with st.spinner("Loading"):
    if pdfreader:
        raw_text = ''
        for page in pdfreader.pages:
            content = page.extract_text()
            if content:
                raw_text += content
        
        if not st.session_state.pdf_processed:
            genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
            login(token="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")
            llm=genai.GenerativeModel(model_name='gemini-1.5-flash')

            embedding_function = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
            faiss_vector_store = FAISS.from_texts([raw_text], embedding_function)
            # result = faiss_vector_store.similarity_search("who is Shrirang",k=3)
            # print("Result1:", result[2].page_content)

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=200,
            )

            texts = text_splitter.split_text(raw_text)
            faiss_vector_store.add_texts(texts[:50])

            st.session_state.faiss_vector_index = VectorStoreIndexWrapper(vectorstore=faiss_vector_store)
            st.session_state.pdf_processed =True

st.sidebar.markdown("## **Welcome to Placy the IICxSPC Chatbot**")
st.sidebar.markdown('##### Here you can ask all your queries regarding placements and internships')
st.sidebar.markdown('##### This chatbot is specifically build for students who have queries regarding placement, internships and related activities')
st.sidebar.markdown(' If anything goes wrong do hard refresh by using **Shift** + **F5** key')

def typing_animation(text, speed):
            for char in text:
                yield char
                time.sleep(speed)

if "intro_displayed" not in st.session_state:
    st.session_state.intro_displayed = True
    intro = "Hello, I am Placy, a  IIC x SPC Chatbot"
    intro2= "You can chat with Placy"
    st.write_stream(typing_animation(intro,0.02))
    st.write_stream(typing_animation(intro2,0.02))

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

#initialised prePrompt_selected
if "prePrompt_selected" not in st.session_state:
    st.session_state.prePrompt_selected = False

if "btn_selected" not in st.session_state:
    st.session_state.btn_selected = True

#defined callback fn
def btn_callback():
    st.session_state.prePrompted_selected = False
    st.session_state.btn_selected=False

prePrompt = None
# if st.session_state.btn_selected:
    
#     with st.expander("What can you ask?"):
#         col1, col2,col3=st.columns(3, gap="small")
#         row2_col1, row2_col2, row2_col3, row2_col4 = st.columns(4, gap="small")
#         row3_col1, row3_col2, row3_col3, row3_col4 = st.columns(4, gap="small")
#         row4_col1, row4_col2, row4_col3, row4_col4 = st.columns(4, gap="small")
#         with row2_col1:
#             button_a = st.button('# Tell me in detail about Shrirang')
#         with row2_col2:    
#             button_b = st.button('# What education Shrirang have completed?')
#         with row2_col3:  
#             button_c = st.button('# Why should I hire Shrirang for AI role?')
#         with row2_col4:  
#             button_d = st.button('# List down Genrative AI projects shrirang have done')
#         with row3_col1:  
#             button_h = st.button('# List down all the skills Shrirang has')
#         with row3_col2:  
#             button_e = st.button('# What is Shrirang\'s GitHub id?')
#         with row3_col3:  
#             button_f = st.button('# What is Shrirang\'s LinkedIn id?')
#         with row3_col4:  
#             button_g = st.button('# List down Machine Learning projects shrirang have done')
#         with row4_col2:  
#             button_i = st.button('# What all position of responsibility Shrirang have took?')
#         with row4_col3:  
#             button_j = st.button('# What all hobbies Shrirang have?')
#         with col1:
#             button_x= st.button('# x',on_click=btn_callback ,  type='primary', key='close_btn')
           

#     if button_a:
#         st.session_state.prePrompt_selected = True
#         prePrompt = 'Tell me in detail about Shrirang'   

#     if button_b:
#         st.session_state.prePrompt_selected = True
#         prePrompt = 'What education Shrirang have completed? answer in points' 

#     if button_c:
#         st.session_state.prePrompt_selected = True
#         prePrompt = 'Analyse and answer Why a recruiter should hire Shrirang for AI role? answer in points'  
    
#     if button_d:
#         st.session_state.prePrompt_selected = True
#         prePrompt = 'List down Machine Learning projects shrirang have done? answer in points'  
    
#     if button_e:
#         st.session_state.prePrompt_selected = True
#         prePrompt = 'What is Shrirang\'s GitHub id?' 
        
#     if button_f:
#         st.session_state.prePrompt_selected = True
#         prePrompt = 'What is Shrirang\'s LinkedIn id?'  
    
#     if button_g:
#         st.session_state.prePrompt_selected = True
#         prePrompt = 'List down Machine Learning projects shrirang have done? answer in points'
    
#     if button_h:
#         st.session_state.prePrompt_selected = True
#         prePrompt = 'List down all the skills Shrirang has? answer in points'

#     if button_i:
#         st.session_state.prePrompt_selected = True
#         prePrompt = 'What all position of responsibility Shrirang have took?'

#     if button_j:
#         st.session_state.prePrompt_selected = True
#         prePrompt = 'What all hobbies Shrirang have? answer in points'


# if st.session_state.prePrompt_selected and prePrompt is not None:
# if st.session_state.prePrompt_selected and prePrompt is not None:
    
    # query_text = prePrompt.strip() 
    # gemini_llm = GeminiLLM(model_name='gemini-1.5-flash')
    # if st.session_state.faiss_vector_index is not None:
    #     answer = st.session_state.faiss_vector_index.query(query_text, llm=gemini_llm).strip()
    #     typing_speed = 0.02
    #     if "context" or "no" in answer:
    #         with st.chat_message("assistant"):
    #             st.write_stream(typing_animation(answer, typing_speed))
    #     else:        
    #         with st.chat_message("assistant"):
    #             st.write_stream(typing_animation(answer,typing_speed))
                
    #     st.session_state.messages.append({"role": "assistant", "content": answer})


prompt = st.chat_input("Chat with Shrirang...")
 
if prompt:
    with st.chat_message('user'):
        st.markdown(prompt)
    
    st.session_state.messages.append({"role": "user", "content": prompt})

    query_text = prompt.strip()
    gemini_llm = GeminiLLM(model_name='gemini-1.5-flash')
    if st.session_state.faiss_vector_index is not None:
        
        answer = st.session_state.faiss_vector_index.query(query_text, llm=gemini_llm).strip()
        
        typing_speed = 0.02
        if "context" or "no" in answer:
            with st.chat_message("assistant"):
                st.write_stream(typing_animation(answer, typing_speed))
        else:
            with st.chat_message("assistant"):
                st.write_stream(typing_animation(answer,typing_speed))
        
        st.session_state.messages.append({"role": "assistant", "content": answer})
    else:
        st.error("Database not initialized. Kindly reload and upload the PDF first.")
