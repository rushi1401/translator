# import streamlit as st
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# import google.generativeai as genai
# from langchain.vectorstores import FAISS
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate
# import os

# # Directly configure the API key
# api_key = "AIzaSyBaNyjY1c6NqYc8JwR82NkUrGRjg5pO5bY"
# genai.configure(api_key=api_key)

# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#     return text

# def get_text_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
#     chunks = text_splitter.split_text(text)
#     return chunks

# def get_vector_store(text_chunks):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
#     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#     if not os.path.exists("faiss_index"):
#         os.makedirs("faiss_index")
#     vector_store.save_local("faiss_index")

# def get_conversational_chain():
#     prompt_template = """
#     Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
#     provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
#     Context:\n {context}?\n
#     Question: \n{question}\n

#     Answer:
#     """

#     model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=api_key)

#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#     chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

#     return chain

# def user_input(user_question):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    
#     # Load the vector store with dangerous deserialization allowed
#     new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
#     docs = new_db.similarity_search(user_question)

#     chain = get_conversational_chain()
    
#     response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

#     st.write("Reply: ", response["output_text"])

# def main():
#     st.set_page_config("Chat PDF")
#     st.header("Chat with PDF using Gemini💁")

#     user_question = st.text_input("Ask a Question from the PDF Files")

#     if user_question:
#         user_input(user_question)

#     with st.sidebar:
#         st.title("Menu:")
#         pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
#         if st.button("Submit & Process"):
#             with st.spinner("Processing..."):
#                 raw_text = get_pdf_text(pdf_docs)
#                 text_chunks = get_text_chunks(raw_text)
#                 get_vector_store(text_chunks)
#                 st.success("Done")

# if __name__ == "__main__":
#     main()
# import streamlit as st
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# import google.generativeai as genai
# from langchain.vectorstores import FAISS
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains import LLMChain
# from langchain.prompts import PromptTemplate
# import os
# from docx import Document

# # Directly configure the API key
# api_key = "AIzaSyBaNyjY1c6NqYc8JwR82NkUrGRjg5pO5bY"
# genai.configure(api_key=api_key)

# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#     return text

# def get_text_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
#     chunks = text_splitter.split_text(text)
#     return chunks

# def get_vector_store(text_chunks):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
#     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#     if not os.path.exists("faiss_index"):
#         os.makedirs("faiss_index")
#     vector_store.save_local("faiss_index")

# def get_translation_chain(target_language):
#     prompt_template = f"""
#     Translate the following text to {target_language}. Make sure to keep the meaning and context intact. Do not provide any additional information.\n\n
#     Text:\n {{text}}\n

#     Translation:
#     """

#     model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=api_key)

#     prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
#     chain = LLMChain(llm=model, prompt=prompt)

#     return chain

# def translate_document(target_language):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    
#     new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
#     docs = new_db.similarity_search("Translate the document")

#     chain = get_translation_chain(target_language)

#     translations = []
#     for doc in docs:
#         response = chain({"text": doc.page_content}, return_only_outputs=True)
        
#         # Debugging: Print the response object to understand its structure
#         print(response)
        
#         if "output_text" in response:
#             translations.append(response["output_text"])
#         else:
#             st.error(f"Unexpected response format: {response}")

#     translated_text = "\n\n".join(translations)
#     return translated_text

# def save_translation_as_docx(translated_text, filename="translated_document.docx"):
#     doc = Document()
#     doc.add_heading('Translated Document', 0)
#     doc.add_paragraph(translated_text)
#     doc.save(filename)
#     return filename

# def main():
#     st.set_page_config("Translate PDF")
#     st.header("Translate PDF using Gemini💁")

#     # Language selection
#     languages = {
#         "Hindi": "Hindi",
#         "Marathi": "Marathi",
#         "Tamil": "Tamil",
#         "Telugu": "Telugu",
#         "Kannada": "Kannada",
#         "Malayalam": "Malayalam",
#         "Arabic": "Arabic"
#     }
#     target_language = st.selectbox("Select the target language for translation:", list(languages.keys()))

#     if st.button("Translate Document"):
#         translated_text = translate_document(languages[target_language])
#         if translated_text:
#             filename = save_translation_as_docx(translated_text)
#             with open(filename, "rb") as file:
#                 st.download_button(
#                     label="Download Translated Document",
#                     data=file,
#                     file_name=filename,
#                     mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
#                 )

#     with st.sidebar:
#         st.title("Menu:")
#         pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
#         if st.button("Submit & Process"):
#             with st.spinner("Processing..."):
#                 raw_text = get_pdf_text(pdf_docs)
#                 text_chunks = get_text_chunks(raw_text)
#                 get_vector_store(text_chunks)
#                 st.success("Done")

# if __name__ == "__main__":
#     main()
# import streamlit as st
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# import google.generativeai as genai
# from langchain.vectorstores import FAISS
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains import LLMChain
# from langchain.prompts import PromptTemplate
# import os
# from docx import Document

# # Directly configure the API key
# api_key = "AIzaSyBaNyjY1c6NqYc8JwR82NkUrGRjg5pO5bY"
# genai.configure(api_key=api_key)

# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#     return text

# def get_text_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
#     chunks = text_splitter.split_text(text)
#     return chunks

# def get_vector_store(text_chunks):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
#     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#     if not os.path.exists("faiss_index"):
#         os.makedirs("faiss_index")
#     vector_store.save_local("faiss_index")

# def get_translation_chain(target_language):
#     prompt_template = f"""
#     Translate the following text to {target_language}. Make sure to keep the meaning and context intact. Do not provide any additional information.\n\n
#     Text:\n {{text}}\n

#     Translation:
#     """

#     model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=api_key)

#     prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
#     chain = LLMChain(llm=model, prompt=prompt)

#     return chain

# def translate_document(target_language):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    
#     new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
#     docs = new_db.similarity_search("Translate the document")

#     chain = get_translation_chain(target_language)

#     translations = []
#     for doc in docs:
#         response = chain({"text": doc.page_content}, return_only_outputs=True)
        
#         if "output_text" in response:
#             translations.append(response["output_text"])
#         else:
#             st.error(f"Unexpected response format: {response}")

#     translated_text = "\n\n".join(translations)
#     return translated_text

# def save_translation_as_docx(translated_text, filename="translated_document.docx"):
#     doc = Document()
#     doc.add_heading('Translated Document', 0)
#     doc.add_paragraph(translated_text)
#     doc.save(filename)
#     return filename

# def save_translation_as_txt(translated_text, filename="translated_document.txt"):
#     with open(filename, "w", encoding="utf-8") as file:
#         file.write(translated_text)
#     return filename

# def main():
#     st.set_page_config("Translate PDF")
#     st.header("Translate PDF using Gemini💁")

#     # Language selection
#     languages = {
#         "Hindi": "Hindi",
#         "Marathi": "Marathi",
#         "Tamil": "Tamil",
#         "Telugu": "Telugu",
#         "Kannada": "Kannada",
#         "Malayalam": "Malayalam",
#         "Arabic": "Arabic"
#     }
#     target_language = st.selectbox("Select the target language for translation:", list(languages.keys()))

#     if st.button("Translate Document"):
#         translated_text = translate_document(languages[target_language])
#         if translated_text:
#             docx_filename = save_translation_as_docx(translated_text)
#             txt_filename = save_translation_as_txt(translated_text)

#             with open(docx_filename, "rb") as file:
#                 st.download_button(
#                     label="Download Translated Document (.docx)",
#                     data=file,
#                     file_name=docx_filename,
#                     mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
#                 )

#             with open(txt_filename, "rb") as file:
#                 st.download_button(
#                     label="Download Translated Document (.txt)",
#                     data=file,
#                     file_name=txt_filename,
#                     mime="text/plain"
#                 )

#     with st.sidebar:
#         st.title("Menu:")
#         pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
#         if st.button("Submit & Process"):
#             with st.spinner("Processing..."):
#                 raw_text = get_pdf_text(pdf_docs)
#                 text_chunks = get_text_chunks(raw_text)
#                 get_vector_store(text_chunks)
#                 st.success("Done")

# if __name__ == "__main__":
#     main()