"""
Logica del Agente Inteligente - Mercado Central 24h
"""

# =====================================================================
# 1. IMPORTACION DE LIBRERIAS Y CONFIGURACION DE ENTORNO
# =====================================================================

import os
from pathlib import Path
from typing import Literal, List, Dict, TypedDict, Optional

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
try:
    from langchain.chains.combine_documents import create_stuff_documents_chain
except ImportError:
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from pydantic import BaseModel, Field
from langgraph.graph import START, END, StateGraph


# =====================================================================
# 2. PROCESAMIENTO DE DOCUMENTOS E INDEXACION VECTORIAL
# =====================================================================

def cargar_politicas_empresa(directorio_path: str = "documentos/"):
    """
    Carga y segmenta los documentos PDF oficiales de Mercado Central 24h
    (Reglamento, Proveedores, FAQ, Atencion al cliente) para crear nuestra base de conocimientos.
    """
    docs = []
    ruta = Path(directorio_path)
    
    # Recorremos la carpeta para encontrar todos los archivos de la empresa en formato PDF
    for pdf_path in ruta.glob("*.pdf"):
        try:
            loader = PyMuPDFLoader(str(pdf_path))
            documentos_cargados = loader.load()
            docs.extend(documentos_cargados)
            print(f"Archivo leido exitosamente: {pdf_path.name}")
        except Exception as e:
            print(f"Error al leer el archivo {pdf_path.name}: {e}")

    # Dividimos los textos en fragmentos mas grandes para no superar el limite de cuota API (429)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,   # Aumentado para generar <100 fragmentos en total
        chunk_overlap=150   # Superposicion adecuada entre partes
    )
    docs_splits = splitter.split_documents(docs)
    print(f"Total de fragmentos creados: {len(docs_splits)}")
    return docs_splits


def inicializar_almacen_vectorial(chunks_documentos, gemini_api_key: str):
    """
    Creamos los embeddings usando Google y los guardamos en una base de datos local FAISS.
    """
    modelo_embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=gemini_api_key
    )

    vectorstore = FAISS.from_documents(
        chunks_documentos,
        modelo_embeddings
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.3, "k": 4}
    )
    return retriever


# =====================================================================
# 3. COMPONENTES DE INTELIGENCIA ARTIFICIAL Y PROMPTS
# =====================================================================

def inicializar_llm(gemini_api_key: str):
    """
    Inicializa el modelo de Gemini con temperatura 0 para respuestas objetivas.
    """
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0, 
        google_api_key=gemini_api_key
    )


def crear_cadena_rag(llm):
    """
    Define el prompt del sistema para el asistente IA y crea la cadena RAG.
    """
    prompt_rag = ChatPromptTemplate.from_messages([
        (
            "system",
            """Eres un especialista de soporte, atención al cliente y operaciones de Mercado Central 24h.\n
            Responde siempre utilizando strictly los conocimientos de los documentos oficiales proporcionados.\n
            Si la información requerida no se encuentra en el contexto, responde únicamente 'No lo sé'."""
        ),
        (
            "human",
            "Contexto de la empresa: {context}\n\nPregunta del usuario: {input}"
        )
    ])

    return create_stuff_documents_chain(llm, prompt_rag)


def busqueda_de_respuestas_RAG(pregunta: str, retriever, document_chain):
    """
    Ejecuta la recuperación de documentos mediante FAISS y evalúa el resultado.
    """
    documentos_relacionados = retriever.invoke(pregunta)

    if not documentos_relacionados:
        return {
            "respuesta": "No lo sé",
            "citaciones": [],
            "documentos_encontrados": False
        }

    answer = document_chain.invoke({
        "input": pregunta,
        "context": documentos_relacionados
    })

    if answer.rstrip(".!?") == "No lo sé":
        return {
            "respuesta": "No lo sé",
            "citaciones": [],
            "documentos_encontrados": False
        }

    return {
        "respuesta": answer,
        "citaciones": documentos_relacionados,
        "documentos_encontrados": True
    }


# =====================================================================
# 4. DEFINICION DE NODOS Y ARISTAS DEL GRAFO (LANGGRAPH)
# =====================================================================

class TriajeOut(BaseModel):
    decision: Literal["AUTO_RESOLVER", "PEDIR_INFO", "ABRIR_TICKET"] = Field(
        description="Ruta de atencion segun la intencion del usuario."
    )
    urgencia: Literal["BAJA", "MEDIANA", "ALTA"] = Field(
        description="Nivel de prioridad o urgencia del caso."
    )
    campos_faltantes: List[str] = Field(
        default_factory=list,
        description="Lista de datos o parametros no proporcionados por el usuario."
    )


class AgentState(TypedDict, total=False):
    pregunta: str
    triaje: dict
    respuesta: Optional[str]
    citaciones: Optional[list]
    documentos_encontrados: Optional[bool]
    rag_exito: bool
    accion_final: str


def nodo_triage(state: AgentState, llm) -> dict:
    pregunta = state.get("pregunta", "")
    llm_con_estructura = llm.with_structured_output(TriajeOut)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """Eres el modulo de triaje operativo de Mercado Central 24h.
            Clasifica la consulta del usuario en una de estas 3 rutas:
            
            1. AUTO_RESOLVER: Consultas sobre politicas, horarios, devoluciones, proveedores,
               compradores de area, o dudas generales cubiertas en nuestros manuales.
            2. PEDIR_INFO: La pregunta es muy vaga o le faltan datos esenciales para poder responder.
            3. ABRIR_TICKET: Quejas graves, reportes de acoso, emergencias o solicitudes explicitas
               de hablar con un humano o soporte tecnico.
            """
        ),
        ("human", "{input}")
    ])

    cadena_triaje = prompt | llm_con_estructura
    resultado = cadena_triaje.invoke({"input": pregunta})

    return {
        "triaje": resultado.model_dump()
    }


def nodo_auto_resolver(state: AgentState, retriever, document_chain) -> dict:
    pregunta = state.get("pregunta", "")
    resultado_rag = busqueda_de_respuestas_RAG(pregunta, retriever, document_chain)

    return {
        "respuesta": resultado_rag["respuesta"],
        "citaciones": resultado_rag["citaciones"],
        "documentos_encontrados": resultado_rag["documentos_encontrados"],
        "rag_exito": resultado_rag["documentos_encontrados"],
        "accion_final": "AUTO_RESUELTO"
    }


def nodo_pedir_info(state: AgentState) -> dict:
    triaje_datos = state.get("triaje", {})
    faltantes = ", ".join(triaje_datos.get("campos_faltantes", ["mas detalles"]))

    respuesta = (
        f"Para poder ayudarte correctamente con tu consulta sobre Mercado Central 24h, "
        f"por favor proporcionanos la siguiente informacion adicional: {faltantes}."
    )

    return {
        "respuesta": respuesta,
        "rag_exito": False,
        "accion_final": "INFO_SOLICITADA"
    }


def nodo_abrir_ticket(state: AgentState) -> dict:
    triaje_datos = state.get("triaje", {})
    urgencia = triaje_datos.get("urgencia", "MEDIANA")

    respuesta = (
        f"Tu solicitud ha sido registrada con urgencia **{urgencia}** y derivada al equipo "
        f"de atencion especializada de Mercado Central 24h. Un ejecutivo te contactara a la brevedad."
    )

    return {
        "respuesta": respuesta,
        "rag_exito": False,
        "accion_final": "TICKET_CREADO"
    }


# =====================================================================
# 5. CONSTRUCCION Y COMPILACION DEL WORKFLOW
# =====================================================================

def ruta_decision(state: AgentState) -> str:
    triaje_datos = state.get("triaje", {})
    return triaje_datos.get("decision", "AUTO_RESOLVER")


def compilar_agente_operaciones(llm, retriever, document_chain):
    workflow = StateGraph(AgentState)

    workflow.add_node("nodo_triage", lambda state: nodo_triage(state, llm))
    workflow.add_node("nodo_auto_resolver", lambda state: nodo_auto_resolver(state, retriever, document_chain))
    workflow.add_node("nodo_pedir_info", nodo_pedir_info)
    workflow.add_node("nodo_abrir_ticket", nodo_abrir_ticket)

    workflow.add_edge(START, "nodo_triage")

    workflow.add_conditional_edges(
        "nodo_triage",
        ruta_decision,
        {
            "AUTO_RESOLVER": "nodo_auto_resolver",
            "PEDIR_INFO": "nodo_pedir_info",
            "ABRIR_TICKET": "nodo_abrir_ticket"
        }
    )

    workflow.add_edge("nodo_auto_resolver", END)
    workflow.add_edge("nodo_pedir_info", END)
    workflow.add_edge("nodo_abrir_ticket", END)

    app = workflow.compile()
    return app
