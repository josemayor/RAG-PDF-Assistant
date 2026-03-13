import streamlit as st
import time
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

st.set_page_config(page_title="RAG PDF - Attention is All You Need", layout="centered")
st.title("RAG-PDF Assistant")
st.caption(
    "Advanced RAG system for querying PDFs using FAISS, semantic embeddings and LLM reasoning. "
    "Built with LangChain, Streamlit and OpenAI. Includes retrieval metrics and evaluation pipeline."
)

PDF_PATH = "data/attention_is_all_you_need.pdf"


# === VECTOR STORE (se guarda en disco) ===

# Explicación de esta sección:
# - Carga el PDF usando PyPDFLoader
# - Divide el texto en chunks con RecursiveCharacterTextSplitter
# - Genera embeddings con HuggingFaceEmbeddings (modelo all-MiniLM-L6-v2)
# - Almacena los embeddings en FAISS (en disco para evitar reprocesar)

if "vectorstore" not in st.session_state:
    with st.spinner("Processing PDF (first time only)..."):
        if os.path.exists("faiss_index"):
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            vectorstore = FAISS.load_local(
                "faiss_index", embeddings, allow_dangerous_deserialization=True
            )
        else:
            loader = PyPDFLoader(PDF_PATH)
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            )
            splits = text_splitter.split_documents(docs)

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            vectorstore = FAISS.from_documents(splits, embeddings)
            vectorstore.save_local("faiss_index")

        st.session_state.vectorstore = vectorstore
        st.session_state.retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

retriever = st.session_state.retriever


# === LLM OpenAI ===

# Explicación de esta sección:
# - Se inicializa un ChatOpenAI apuntando a LM-STUDIO (modelo dolphin3.0-llama3.1-8b)
# - Se configura para streaming (mostrar token a token) y temperatura baja (respuestas precisas)
# - Se guarda en session_state para evitar reinicializar en cada interacción

if "llm" not in st.session_state:
    llm = ChatOpenAI(
        model="dolphin3.0-llama3.1-8b",
        base_url="http://localhost:1234/v1",  # Puerto para LM STUDIO
        api_key="lm-studio",  # dummy key, LM-STUDIO no la valida
        temperature=0.15,  # Bajo para respuestas muy precisas
        max_tokens=2048,
        streaming=True,  # Streamlit muestra token a token
    )
    st.session_state.llm = llm
llm = st.session_state.llm


# === PROMPT + CHAIN ===

# Explicación de esta sección:
# - Se define un prompt para reescribir la pregunta y mejorar la recuperación de documentos

query_rewriter_prompt = ChatPromptTemplate.from_template(
    """
Rewrite the user question to improve document retrieval.
Question:
{question}"""
)

prompt_template = ChatPromptTemplate.from_template(
    """Responde SOLO usando el contexto. Sé preciso y cita la página.
Contexto:
{context}

Pregunta: {question}
Respuesta:"""
)


def format_docs(docs):
    return "\n\n".join(
        [
            f"Fuente (pág. {doc.metadata.get('page', 'N/A')}):\n{doc.page_content}"
            for doc in docs
        ]
    )


# Chain LCEL
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt_template
    | llm
    | StrOutputParser()
)


# === CHAT ===

# Explicación de esta sección:
# - Se inicializa el chat con un mensaje de bienvenida del asistente
# - Se muestra el historial de mensajes (usuario y asistente)
# - Al enviar una pregunta, se reescribe para mejorar la recuperación, se ejecuta
#   el RAG, se muestra la respuesta y el tiempo de respuesta
# - También se muestran las fuentes recuperadas para transparencia
# - El historial se guarda en session_state para mantenerlo entre interacciones

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "¡Hola! Pregúntame cualquier cosa sobre el paper 'Attention is All You Need'.",
        }
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Escribe tu pregunta..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        start_time = time.perf_counter()
        with st.spinner("Buscando en el paper..."):
            # Se reescribe la pregunta para mejorar la recuperación
            # de documentos relevantes
            rewritten_query = llm.invoke(
                query_rewriter_prompt.format_prompt(question=user_input)
            ).content

            response = rag_chain.invoke(rewritten_query)
            elapsed = time.perf_counter() - start_time

            st.markdown(response)
            st.caption(f"Time: {elapsed:.2f} segs")

            # Funciona el RAG? Se muestran las fuentes recuperadas
            docs = st.session_state.retriever.invoke(rewritten_query)
            st.caption(f"Retrieved {len(docs)} sources")
            with st.expander("Cited sources"):
                st.markdown(format_docs(docs))

        st.session_state.messages.append({"role": "assistant", "content": response})