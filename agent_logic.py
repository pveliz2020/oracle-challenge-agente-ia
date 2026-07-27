"""
Logica del Agente Inteligente - Mercado Central 24h
"""

# =====================================================================
# 1. IMPORTACION DE LIBRERIAS Y CONFIGURACION DE ENTORNO
# =====================================================================

# Definicion de dependencias de LangChain, LangGraph y Google Generative AI
import os
from pathlib import Path
from typing import Literal, List, Dict, TypedDict, Optional

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
try: #Solución ante problemas con langchain
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
            loader = PyMuPDFLoader(str(pdf_path))  # Utilizando la herramienta PyMuPDFLoader vista en el curso
            documentos_cargados = loader.load()
            docs.extend(documentos_cargados)
            print(f"Archivo leido exitosamente: {pdf_path.name}")
        except Exception as e:
            print(f"Error al leer el archivo {pdf_path.name}: {e}")

    # Dividimos los textos en fragmentos mas pequeños para facilitar la busqueda
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,  # Tamano de fragmento aprendido en clases
        chunk_overlap=30 # Superposicion 10% para no perder el hilo entre partes
    )
    docs_splits = splitter.split_documents(docs)
    print(f"Total de fragmentos creados: {len(docs_splits)}")
    return docs_splits


def inicializar_almacen_vectorial(chunks_documentos, gemini_api_key: str):
    """
    Creamos los embeddings
    usando Google y los guardamos en una base de datos local FAISS para realizar busquedas rapidas.
    """
    # Modelo de embeddings oficial de Google
    modelo_embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=gemini_api_key
    )

    # Creacion de la base de datos vectorial local FAISS vista en cusos previos de alura
    vectorstore = FAISS.from_documents(
        chunks_documentos,
        modelo_embeddings
    )

    # Configuracion del buscador con filtro de similitud para evitar respuestas inventadas
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.3, "k": 4} # Umbral exacto utilizado en nuestras clases
    )
    return retriever


# =====================================================================
# 3. COMPONENTES DE INTELIGENCIA ARTIFICIAL Y PROMPTS
# =====================================================================
# Inicializacion del modelo gemini-2.5-flash y definicion de plantillas de prompts para el modulo de triaje y el motor RAG.

def inicializar_llm(gemini_api_key: str):
    """
    Inicializa el modelo oficial de Google visto en el curso de IA ,
    con temperatura 0 para asegurar respuestas objetivas y precisas.
    """
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0, 
        google_api_key=gemini_api_key
    )


def crear_cadena_rag(llm):
    """
    Define el prompt del sistema para el asistente IA
    y crea la cadena de documentos como vimos en clases.
    """
    # Prompt del sistema alineado con los manuales y políticas de la empresa
    prompt_rag = ChatPromptTemplate.from_messages([
        (
            "system",
            """Eres un especialista de soporte, atención al cliente y operaciones de Mercado Central 24h.\n
            Responde siempre utilizando estrictamente los conocimientos de los documentos oficiales proporcionados.\n
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
    Ejecuta la recuperación de documentos relevantes mediante FAISS y evalúa
    si el contenido es concluyente antes de entregar la respuesta final.
    """
    # Recuperación de documentos basados en el umbral de similitud configurado
    documentos_relacionados = retriever.invoke(pregunta) #Invoke

    if not documentos_relacionados:
        return {
            "respuesta": "No lo sé",
            "citaciones": [],
            "documentos_encontrados": False
        }

    # Invocación de la cadena con el contexto recuperado
    answer = document_chain.invoke({
        "input": pregunta,
        "context": documentos_relacionados
    })

    # Validación estricta por si el modelo responde por defecto que no sabe
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

# Esquema de salida estricta para el triaje mediante Pydantic (visto en clases)
#Funciona como "Formulario de Clasificación"

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


# Estado compartido del agente a lo largo de todo el flujo (Graph State)
class AgentState(TypedDict, total=False):
    pregunta: str
    triaje: dict
    respuesta: Optional[str]
    citaciones: Optional[list]
    documentos_encontrados: Optional[bool]
    rag_exito: bool
    accion_final: str


def nodo_triage(state: AgentState, llm) -> dict:
    """
    Analiza la consulta del usuario y decide si se puede resolver con RAG (AUTO_RESOLVER),
    si le falta informacion (PEDIR_INFO) o si requiere atencion humana (ABRIR_TICKET).
    """
    pregunta = state.get("pregunta", "")

    # Forzamos la salida estructurada con el esquema Pydantic
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
    """
    Ejecuta la busqueda RAG sobre los PDFs de Mercado Central 24h.
    """
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
    """
    Solicita mas detalles al usuario cuando el triaje detecto informacion incompleta.
    """
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
    """
    Simula la derivacion a soporte humano o la creacion de un ticket interno.
    """
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
    """
    Funcion auxiliar para la arista condicional.
    Lee la decision tomada por el nodo de triaje y retorna el nombre del siguiente nodo.
    """
    triaje_datos = state.get("triaje", {})
    return triaje_datos.get("decision", "AUTO_RESOLVER")


def compilar_agente_operaciones(llm, retriever, document_chain):
    """
    Construye el grafo de control en LangGraph, define las aristas condicionales
    y compila el workflow final del agente inteligente de Mercado Central 24h.
    """
    #  Inicializamos el grafo utilizando nuestro AgentState
    workflow = StateGraph(AgentState)

    #  Definimos los nodos del grafo pasando las dependencias requeridas (currificacion / cierres)
    workflow.add_node("nodo_triage", lambda state: nodo_triage(state, llm))
    workflow.add_node("nodo_auto_resolver", lambda state: nodo_auto_resolver(state, retriever, document_chain))
    workflow.add_node("nodo_pedir_info", nodo_pedir_info)
    workflow.add_node("nodo_abrir_ticket", nodo_abrir_ticket)

    #  Punto de inicio del grafo
    workflow.add_edge(START, "nodo_triage")

    #  Arista condicional desde el triaje hacia el nodo correspondiente
    workflow.add_conditional_edges(
        "nodo_triage",
        ruta_decision,
        {
            "AUTO_RESOLVER": "nodo_auto_resolver",
            "PEDIR_INFO": "nodo_pedir_info",
            "ABRIR_TICKET": "nodo_abrir_ticket"
        }
    )

    #  Todos los nodos de respuesta conducen al final del flujo
    workflow.add_edge("nodo_auto_resolver", END)
    workflow.add_edge("nodo_pedir_info", END)
    workflow.add_edge("nodo_abrir_ticket", END)

    #  Compilacion del flujo ejecutable
    app = workflow.compile()
    return app
