import streamlit as st
import os

from agent_logic import (
    cargar_politicas_empresa,
    inicializar_almacen_vectorial,
    inicializar_llm,
    crear_cadena_rag,
    compilar_agente_operaciones
)

st.set_page_config(
    page_title="Agente Mercado Central 24h",
    page_icon="🤖"
)

st.title("🤖 Agente de Soporte - Mercado Central 24h")


@st.cache_resource
def iniciar_agente():
    # Obtener API Key
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    if not api_key:
        st.error("No se encontró la API Key de Gemini.")
        return None

    # Cargar y dividir documentos
    chunks = cargar_politicas_empresa("documentos/")

    # Crear retriever
    retriever = inicializar_almacen_vectorial(
        chunks,
        api_key
    )

    # Inicializar modelo
    llm = inicializar_llm(api_key)

    # Crear cadena RAG
    document_chain = crear_cadena_rag(llm)

    # Compilar agente
    agente = compilar_agente_operaciones(
        llm,
        retriever,
        document_chain
    )

    return agente


agente = iniciar_agente()


# ==========================
# Historial del chat
# ==========================

if "messages" not in st.session_state:
    st.session_state.messages = []

for mensaje in st.session_state.messages:
    with st.chat_message(mensaje["role"]):
        st.write(mensaje["content"])


# ==========================
# Entrada del usuario
# ==========================

if prompt := st.chat_input("Escribe tu consulta sobre Mercado Central 24h..."):

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    with st.chat_message("user"):
        st.write(prompt)

    if agente:

        with st.chat_message("assistant"):

            with st.spinner("Procesando..."):

                try:

                    respuesta = agente.invoke(
                        {
                            "pregunta": prompt
                        }
                    )

                    texto = respuesta.get(
                        "respuesta",
                        "No se obtuvo respuesta."
                    )

                    st.write(texto)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": texto
                        }
                    )

                except Exception as e:

                    st.error(f"Error: {e}")
