import os
import streamlit as st  
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from groq import Groq
from langchain_core.messages import HumanMessage, AIMessage
from langchain_qdrant import QdrantVectorStore
from htmlTemplates import css, bot_template, user_template

# Set page configuration
st.set_page_config(page_title="Lawgic.ai", page_icon=":books:", layout="wide")

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(
    api_key=GROQ_API_KEY,
)

# Load PDF text
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_chunks_text(raw_text):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunk = splitter.split_text(raw_text)
    return chunk
    
def get_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectorstore = QdrantVectorStore.from_texts(texts=chunks, embedding=embeddings, url=os.getenv("QDRANT_HOST"), api_key=os.getenv("QDRANT_API_KEY"), collection_name="beginning")
    return vectorstore

# def get_default_vectorstore():
#         vector_db = Qdrant(
#         collection="BNS",
#         url=os.getenv("QDRANT_HOST"),
#         api_key=os.getenv("QDRANT_API_KEY"),
#         embedder=GeminiEmbedder(dimensions=768),)

#         knowledge_base = PDFUrlKnowledgeBase(
#         urls=["https://file.notion.so/f/f/362c1af9-a522-4624-b1c8-d53897a332aa/2b8b4f79-316f-4c49-8a82-479ab1548016/Bharatiya_Nyaya_Sanhita_2023.pdf?table=block&id=1a37d635-716e-801b-8e08-c5e011d9630c&spaceId=362c1af9-a522-4624-b1c8-d53897a332aa&expirationTimestamp=1740348000000&signature=AwxfeweDtlexR3QDyz_C6h0IJ0YnLkvrie146CNFFi8&downloadName=Bharatiya_Nyaya_Sanhita%2C_2023.pdf"],
#         vector_db=vector_db,
#         embedder=GeminiEmbedder(dimensions=768)
#     )
           
# Multichat-PDF
# def get_response(query, vectorstore):
#     docs = vectorstore.similarity_search(query, k=5)
#     docs_string = ""
#     for doc in docs:
#         docs_string += doc.page_content+ "\n"
#     chat_completion = client.chat.completions.create(
#     messages=[
#         {
#             "role":"user",
#             "content": f"Take a look at the following documents: {docs_string} and do not reply to anything unrelated to the documents. Also return the part, chapter or specific clause that is relevant to the query."
#         },
#         {
#             "role": "user",
#             "content": f"{query}",
#         }
#     ],
#     model="llama3-70b-8192",
#     )
#     return chat_completion.choices[0].message.content

#Both 
def get_response(query, vectorstore=None):
    if vectorstore:
        docs = vectorstore.similarity_search(query, k=5)
        docs_string = ""
        for doc in docs:
            docs_string += doc.page_content + "\n"
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": f"Take a look at the following documents: {docs_string} and do not reply to anything unrelated to the documents. Also return the part, chapter or specific clause that is relevant to the query with respect to Bharatiya Nagarik Suraksha Sanhita (BNSS), Bharatiya Nyaya Sanhita (BNS) and Bharatiya Sakshya Adhiniyam (BSA)."
                },
                {
                    "role": "user",
                    "content": f"{query}",
                }
            ],
            model="llama3-70b-8192",
        )
        return chat_completion.choices[0].message.content
    else:
        # Handle chat without vectorstore
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": f"Take a look at the following query: {query} and do not reply to anything unrelated to Indian law which is based on Bharatiya Nagarik Suraksha Sanhita (BNSS), Bharatiya Nyaya Sanhita (BNS) and Bharatiya Sakshya Adhiniyam (BSA). Also return the part, chapter or specific clause that is relevant to the query."
                },
                {
                    "role": "user",
                    "content": f"{query}",
                }
            ],
            model="llama3-70b-8192",
        )
        return chat_completion.choices[0].message.content

# Function to render messages
def render_message(template, message):
    return template.replace("{{MSG}}", message)


def main():
    load_dotenv()
    
    st.header("Lawgic.ai - Legal Text Generation Tool :books:")    
    st.sidebar.title("Capstone Project")

    # Ensure session state keys exist
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None  # Initialize as None to avoid KeyError

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your documents here", type=['pdf', 'docx', 'txt'], accept_multiple_files=True)
        if st.button("Embed Documents"):
            with st.spinner("Processing..."):   
                # Get PDF text
                raw_text = get_pdf_text(pdf_docs)
                # Get chunks of text
                chunks = get_chunks_text(raw_text)   
                # Create vector store                                 
                st.session_state.vectorstore = get_vectorstore(chunks) 
                st.write("Success!") 

    # Add CSS to Streamlit app
    st.markdown(css, unsafe_allow_html=True)

    # Display chat history
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            st.markdown(render_message(user_template, message.content), unsafe_allow_html=True)
        else:
            st.markdown(render_message(bot_template, message.content), unsafe_allow_html=True)

    # Input for new query
    query = st.chat_input("Ask me anything")
    if query is not None and query != "":
        st.session_state.chat_history.append(HumanMessage(query))
        st.markdown(render_message(user_template, query), unsafe_allow_html=True)
        
        ai_response = get_response(query, st.session_state.vectorstore)

        st.markdown(render_message(bot_template, ai_response), unsafe_allow_html=True)
        st.session_state.chat_history.append(AIMessage(ai_response))


if __name__ == '__main__':
    main()


