# Sistema de Soporte Inteligente - Mercado Central 24h

## Descripción General
Este proyecto consiste en el desarrollo de un Agente de Inteligencia Artificial enfocado en la automatización de operaciones y atención al cliente para Mercado Central 24h, una cadena de supermercados con operación continua y presencia en Latinoamérica. 

La aplicación implementa una arquitectura RAG para procesar y consultar de manera precisa la documentación oficial de la empresa, como el Manual de Proveedores, el Reglamento Interno y las Políticas de Devolución.

## Arquitectura de la Solución
El núcleo de la solución se basa en una estructura orientada a grafos de agentes que gestiona el flujo de las consultas mediante tres módulos principales:

1. Clasificación y Triaje: Evalúa la entrada del usuario para identificar la intención de la consulta y determinar el camino óptimo de resolución.
2. Recuperación de Información (RAG): Si la consulta está cubierta por las políticas vigentes, el sistema extrae los fragmentos de texto más relevantes desde un almacén vectorial para dar una respuesta exacta y contextualizada.
3. Gestión de Excepciones: Cuando se detectan solicitudes especiales que escapan a la automatización (como solicitudes de aprobación o excepciones de política), el agente desvía el flujo para simular la apertura de un ticket de soporte con prioridad dinámica.

## Tecnologías y Herramientas Utilizadas
* Lenguaje de programación: Python
* Frameworks de Inteligencia Artificial: LangChain y LangGraph
* Modelo de Lenguaje Principal (LLM): Google Gemini 2.5 Flash
* Modelos de Representación Vectorial: Google Generative AI Embeddings
* Base de Datos Vectorial: FAISS (almacenamiento local)
* Interfaz de Usuario: Streamlit
* Infraestructura en la Nube: Oracle Cloud Infrastructure (OCI)

## Instrucciones para Ejecutar el Proyecto
*(Las instrucciones detalladas para la instalación de dependencias, configuración de variables de entorno y ejecución local se añadirán próximamente).*

## Ejemplos de Preguntas y Respuestas del Agente
*(Se incluirán casos reales de interacción del usuario con el sistema, documentando el comportamiento del módulo RAG y la gestión automática de tickets).*

## Evidencia del Despliegue en OCI
*(En esta sección se adjuntará el enlace público de acceso a la plataforma junto con las capturas de pantalla de la infraestructura ejecutándose en Oracle Cloud).*
