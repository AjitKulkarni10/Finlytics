# -----------------------------
# General Imports
# -----------------------------
import os
import streamlit as st
from utils import *
from streamlit_lottie import st_lottie
from dotenv import load_dotenv
import asyncio
import torch
import re

# Load environment variables
load_dotenv()

# -----------------------------
# LlamaIndex & LangChain Imports
# -----------------------------
from llama_index.core import VectorStoreIndex, download_loader, Settings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai

# -----------------------------
# Configure Gemini API
# -----------------------------
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# -----------------------------
# Ensure asyncio loop
# -----------------------------
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# -----------------------------
# System Prompt for LLM
# -----------------------------
system_prompt = """
[INST]
<>
    You are a professional financial analyst assistant.
    Analyze financial documents carefully and provide **clear, structured, data-backed insights**.

    - Summarize the document with sections: Revenue, Expenses, Profitability, Cash Flow, Assets & Liabilities, Growth Drivers, Risks, Market Outlook.
    - Highlight **positive signals** and **red flags**.
    - Use bullet points or subheadings for clarity.
    - Ground answers in the document; if info is missing, explicitly state it.
    - Keep tone professional and concise.
<>
[/INST]
"""

# -----------------------------
# Models Setup
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# HuggingFace embeddings (GPU if available)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": device}
)

# Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    google_api_key=os.getenv('GOOGLE_API_KEY'),
    system_instruction=system_prompt
)

# Apply LLM and embeddings to LlamaIndex
Settings.llm = llm
Settings.embed_model = embeddings

# -----------------------------
# Streamlit UI
# -----------------------------
lottie_file = load_lottieurl()
st.set_page_config(page_title="Finlytics")
st_lottie(lottie_file, height=175, quality="medium")
st.title("**Finlytics: Document Analysis**")

# -----------------------------
# Session State Initialization
# -----------------------------
if "uploaded" not in st.session_state:
    st.session_state["uploaded"] = False
    st.session_state["filename"] = None
    st.session_state["initial_response"] = None
    st.session_state["query_engine"] = None
    if "pdf_indices" not in st.session_state:
        st.session_state["pdf_indices"] = {}  # filename → index
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

# -----------------------------
# Reset Function
# -----------------------------
def reset():
    st.session_state["uploaded"] = False
    st.session_state["filename"] = None
    st.session_state["initial_response"] = None
    st.session_state["query_engine"] = None
    st.session_state["pdf_indices"] = {}
    st.session_state["chat_history"] = []


st.button("Reset All", on_click=reset)

# -----------------------------
# PDF Upload & Indexing
# -----------------------------
uploaded_files = st.file_uploader(
    "Upload PDF documents",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    PyMuPDFReader = download_loader("PyMuPDFReader")
    loader = PyMuPDFReader()

    for file in uploaded_files:
        if file.name not in st.session_state["pdf_indices"]:
            # Save file
            SAVE_DIR = "./statements"
            os.makedirs(SAVE_DIR, exist_ok=True)
            path = os.path.join(SAVE_DIR, file.name)
            with open(path, "wb") as f:
                f.write(file.getbuffer())

            # Load and index
            documents = loader.load(file_path=path, metadata=True)
            index = VectorStoreIndex.from_documents(documents)
            st.session_state["pdf_indices"][file.name] = index

    st.success(f"Uploaded {len(uploaded_files)} PDFs successfully!")

# -----------------------------
# Display Initial Analysis & Chat for Multiple PDFs
# -----------------------------
if st.session_state["pdf_indices"]:
    st.markdown("### Uploaded PDFs")
    for fname in st.session_state["pdf_indices"]:
        st.write(f"- {fname}")

    # Initialize container for initial responses if not exists
    if "initial_responses" not in st.session_state:
        st.session_state["initial_responses"] = {}

    # Generate and display initial summary for each PDF
    for fname, idx in st.session_state["pdf_indices"].items():
        if fname not in st.session_state["initial_responses"]:
            initial_prompt = """
            Provide a comprehensive and structured overview of this financial document.
            Include: Executive Summary, Revenue & Sales, Profitability, Expenses, Balance Sheet Highlights, Cash Flow, Growth Drivers, Risks, Future Outlook.
            Use bullet points and sections. Mention missing info explicitly.
            """
            with st.spinner(f"Generating initial analysis for {fname}..."):
                response = idx.as_query_engine().query(initial_prompt)
                st.session_state["initial_responses"][fname] = response.response

        # Display formatted initial analysis
        st.markdown(f"### Initial Financial Overview: {fname}")
        formatted_response = re.sub(r"(\$\d+(?:,\d+)?(?:\.\d+)?)", r"**\1**", st.session_state["initial_responses"][fname])
        formatted_response = formatted_response.replace("\n", "  \n")
        with st.expander("Show Detailed Report"):
            st.markdown(formatted_response)
        st.write("---")


# -----------------------------
# Cross-PDF Analysis
# -----------------------------
if len(st.session_state["pdf_indices"]) > 1:
    st.markdown("## Cross-PDF Analysis")

    # Combine all PDF content references (you could also just pass index/query_engine)
    cross_prompt = """
    You are a professional financial analyst assistant. 

    Based on the uploaded financial documents, provide a **combined analysis**:
    1. Highlight trends or patterns that are common across multiple PDFs.
    2. Note differences or outliers between the PDFs.
    3. Identify recurring risks, opportunities, or financial signals.
    4. Suggest aggregated insights for investors or management.
    5. Use bullet points, subheadings, and clear structured formatting.
    6. Mention if certain information is missing in some PDFs.

    Reference the PDFs by filename if needed.
    """

    # Query each PDF index and combine their content
    cross_combined_response = ""
    for fname, idx in st.session_state["pdf_indices"].items():
        resp = idx.as_query_engine().query(cross_prompt)
        pdf_resp = resp.response
        pdf_resp = pdf_resp.replace("\n", "  \n")  # preserve line breaks
        cross_combined_response += f"### Insights from {fname}\n{pdf_resp}\n\n"

    # Display cross-PDF analysis in an expander
    with st.expander("Show Cross-PDF Analysis"):
        st.markdown(cross_combined_response)

# -----------------------------
# Sentiment Analysis on All PDFs
# -----------------------------
if st.session_state["pdf_indices"]:
    st.markdown("## Sentiment Analysis")

    sentiment_prompt = """
    Perform a financial sentiment analysis on all uploaded documents.

    1. **Overall Sentiment**: Positive / Neutral / Negative.
    2. **Revenue & Growth Sentiment**: Optimistic, cautious, or negative.
    3. **Profitability Sentiment**: Tone regarding margins, net income, costs.
    4. **Overall Tonality**
    6. **Highlighted Phrases**: Extract notable positive and negative statements.
    7. **Final Summary**: 3-5 bullet points giving the big picture.

    Analyze sentiment across all uploaded PDFs together, and highlight if there are differences
    between documents (e.g., one optimistic vs. another cautious).
    """

    # Combine all PDFs into one query context
    all_texts = ""
    for fname, idx in st.session_state["pdf_indices"].items():
        resp = idx.as_query_engine().query("Summarize the main content in detail.")
        all_texts += f"\n\n### From {fname}:\n{resp.response}"

    # Run sentiment analysis
    any_idx = list(st.session_state["pdf_indices"].values())[0]  # use one engine
    sentiment_resp = any_idx.as_query_engine().query(sentiment_prompt + all_texts)

    # Display sentiment
    st.markdown(sentiment_resp.response)


# -----------------------------
# Chatbot for follow-up questions
# -----------------------------

if st.session_state["pdf_indices"]:
    # Display chat history
    for user_msg, bot_msg in st.session_state["chat_history"]:
        st.markdown(f"**You:** {user_msg}")
        st.markdown(f"**DocCheck:** {bot_msg}")
        st.write("---")

    # User input
    user_input = st.text_input("Ask a question about all uploaded documents:", key="multi_chat_input")

    if user_input:
        follow_up_prompt = f"""
            The user asked: "{user_input}"

            You are a professional financial analyst assistant. Using the uploaded financial documents, provide a **comprehensive, structured, and insightful response**. Follow these instructions:

            1. **Reference each PDF separately**. Start each section with the filename.  
            2. **Provide an Executive Summary**: Key takeaways and overall financial health.  
            3. **Revenue Analysis**: Comment on trends, growth/decline, segment contributions , costs and changes.  
            4. **Balance Sheet Highlights**: Assets, liabilities, cash position, and debt analysis.  
            5. **Key Ratios** 
            6. **Risks & Challenges**
            7. **Future Outlook / Management Guidance**
            8. **Provide quantitative data wherever available** and explain its **implications for investors or stakeholders**.  
            9. **Use clear bullet points, subheadings, and sections**.  
            10. If information is missing from a document, explicitly state: *"Information not available in this document."*  
            11. Ensure clarity and readability, as if preparing a report for investors.

            Respond with all available insights from the uploaded PDFs, keeping each PDF's findings separate and labeled.
        """


        # Query all PDFs and combine answers
        combined_response = ""
        for fname, idx in st.session_state["pdf_indices"].items():
            resp = idx.as_query_engine().query(follow_up_prompt)
            pdf_resp = resp.response
            # Format numbers & line breaks
            pdf_resp = re.sub(r"(\$\d+(?:,\d+)?(?:\.\d+)?)", r"**\1**", pdf_resp)
            pdf_resp = pdf_resp.replace("\n", "  \n")
            combined_response += f"### From {fname}\n{pdf_resp}\n\n"

        print(combined_response)

        # Store in chat history
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []
        st.session_state["chat_history"].append((user_input, combined_response))

        # Display immediately
        st.markdown(f"**You:** {user_input}")
        with st.expander("DocCheck Response"):
            st.markdown(combined_response)
        st.write("---")

