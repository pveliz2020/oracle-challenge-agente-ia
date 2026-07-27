import os
import sys
from agent_logic import (
    cargar_politicas_empresa,
    inicializar_almacen_vectorial,
    inicializar_llm,
    crear_cadena_rag,
    compilar_agente_operaciones
)

def ejecutar_prueba_terminal():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ Error: No se encontró la variable GEMINI_API_KEY en la terminal.")
        sys.exit(1)

    print("==================================================")
    print("🤖 INICIALIZANDO AGENTE MERCADO CENTRAL 24H")
    print("==================================================")

    print("\n[1/4] Leyendo documentos PDF...")
    chunks = cargar_politicas_empresa("documentos/")
    if not chunks:
        print("❌ No se encontraron PDFs en la carpeta 'documentos/'.")
        return

    print("\n[2/4] Generando base vectorial en FAISS...")
    retriever = inicializar_almacen_vectorial(chunks, api_key)

    print("\n[3/4] Conectando con Gemini 2.5 Flash...")
    llm = inicializar_llm(api_key)
    document_chain = crear_cadena_rag(llm)

    print("\n[4/4] Compilando flujo en LangGraph...")
    agente = compilar_agente_operaciones(llm, retriever, document_chain)

    print("\n==================================================")
    print("✅ AGENTE LISTO. ESCRIBE TU CONSULTA (o 'salir'):")
    print("==================================================\n")

    while True:
        try:
            pregunta = input("\n👤 Usuario > ")
            if pregunta.lower().strip() in ["salir", "exit", "quit"]:
                print("👋 ¡Hasta luego!")
                break
            
            if not pregunta.strip():
                continue

            resultado = agente.invoke({"pregunta": pregunta})
            accion = resultado.get("accion_final", "N/A")
            respuesta = resultado.get("respuesta", "Sin respuesta.")
            
            print(f"\n🧠 [Ruta LangGraph]: {accion}")
            print(f"🤖 Agente > {respuesta}")

        except KeyboardInterrupt:
            print("\n👋 Sesión finalizada.")
            break

if __name__ == "__main__":
    ejecutar_prueba_terminal()
