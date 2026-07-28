import streamlit as st
import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from agent_logic import (
    inicializar_almacen_vectorial,
    inicializar_llm,
    crear_cadena_rag,
    compilar_agente_operaciones
)

st.set_page_config(page_title="Agente Mercado Central 24h", page_icon="🤖")
st.title("🤖 Agente de Soporte - Mercado Central 24h")

@st.cache_resource
def iniciar_agente():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("No se encontró la API Key de Gemini.")
        return None
    
    # Carga EXCLUSIVAMENTE los PDFs e ignora cualquier otro archivo (como el .xlsx)
    loader = PyPDFDirectoryLoader("documentos/")
    chunks = loader.load()

    vectorstore = inicializar_almacen_vectorial(chunks, api_key)
    llm = inicializar_llm(api_key)
    rag_chain = crear_cadena_rag(vectorstore, llm)
    agente = compilar_agente_operaciones(rag_chain, llm)
    return agente

agente = iniciar_agente()

# Historial de chat
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Entrada de usuario
if prompt := st.chat_input("Escribe tu consulta sobre las políticas de la empresa..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    if agente:
        with st.chat_message("assistant"):
            with st.spinner("Procesando respuesta..."):
                try:
                    respuesta = agente.invoke({"consulta": prompt})
                    texto_resp = respuesta.get("respuesta", "Sin respuesta.")
                    st.write(texto_resp)
                    st.session_state.messages.append({"role": "assistant", "content": texto_resp})
                except Exception as e:
                    st.error(f"Error al procesar la consulta: {e}")
