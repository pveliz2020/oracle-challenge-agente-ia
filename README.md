# Sistema de Soporte Inteligente - Mercado Central 24h

## Descripción General
Este proyecto consiste en el desarrollo de un Agente de Inteligencia Artificial enfocado en la automatización de operaciones y atención al cliente para **Mercado Central 24h**, una cadena de supermercados con operación continua y presencia en Latinoamérica. 

La aplicación implementa una arquitectura basada en **RAG (Retrieval-Augmented Generation)** y flujos de agentes para procesar y consultar de manera precisa la documentación oficial de la empresa, como el Manual de Proveedores, el Reglamento Interno y las Políticas de Devolución.

---

## Arquitectura de la Solución
El núcleo de la solución se basa en una estructura orientada a grafos de agentes que gestiona el flujo de las consultas mediante tres módulos principales:

1. **Clasificación y Triaje:** Evalúa la entrada del usuario para identificar la intención de la consulta y determinar el camino óptimo de resolución.
2. **Recuperación de Información (RAG):** Si la consulta está cubierta por las políticas vigentes, el sistema extrae los fragmentos de texto más relevantes desde un almacén vectorial para dar una respuesta exacta y contextualizada.
3. **Gestión de Excepciones:** Cuando se detectan solicitudes especiales que escapan a la automatización (como solicitudes de aprobación o excepciones de política), el agente desvía el flujo para simular la apertura de un ticket de soporte con prioridad dinámica.

---

## Tecnologías y Herramientas Utilizadas
* **Lenguaje de programación:** Python
* **Frameworks de Inteligencia Artificial:** LangChain y LangGraph
* **Modelo de Lenguaje Principal (LLM):** Google Gemini 2.5 Flash
* **Modelos de Representación Vectorial:** Google Generative AI Embeddings
* **Base de Datos Vectorial:** FAISS (almacenamiento local)
* **Interfaz de Usuario:** Streamlit
* **Infraestructura en la Nube:** Oracle Cloud Infrastructure (OCI - Oracle Linux 9)

---

## Instrucciones para Ejecutar el Proyecto

### 1. Clonar el repositorio
```bash
git clone <TU_URL_DE_GITHUB>
cd <nombre-de-tu-carpeta>
```

### 2. Instalar las dependencias
Asegúrate de tener Python 3.9 o superior instalado y ejecuta:
```bash
pip install -r requirements.txt
```

### 3. Configurar las variables de entorno
Crea un archivo llamado `.env` en la raíz del proyecto y añade tu API Key de Google Gemini:
```env
GOOGLE_API_KEY=tu_clave_de_api_aqui
```

### 4. Ejecutar la aplicación localmente
```bash
streamlit run app.py
```

---

## Ejemplos de Preguntas que el Agente Puede Responder
* ¿Cuál es el plazo máximo que tienen los proveedores para la reposición de productos perecederos?
* ¿Cuáles son las políticas de devolución para artículos electrónicos defectuosos?
* ¿Cómo se gestionan las solicitudes de excepciones fuera de horario de oficina para proveedores críticos?

---

## Ejemplos de Respuestas Generadas por el Agente

> **Pregunta del usuario:** ¿Cuál es el plazo máximo para la reposición de productos perecederos según el manual?
> 
> **Respuesta del Agente:** 
> Según el **Manual de Proveedores de Mercado Central 24h** (Sección 4.2 - Logística de Perecederos), el plazo máximo estipulado para la reposición de productos perecederos es de **24 horas** a partir de la emisión del reporte de stock crítico. Las entregas deben realizarse en los centros de distribución autorizados dentro de la ventana horaria de 06:00 a 10:00 AM.

---

## Evidencia del Despliegue en OCI
* **Enlace público de acceso:** `http://136.248.244.134:8501`
* **Infraestructura:** Instancia de Cómputo en la Nube ejecutándose sobre Oracle Linux 9 en Oracle Cloud Infrastructure (OCI), gestionada mediante acceso remoto seguro vía SSH y ejecutada en segundo plano con persistencia mediante `nohup`.
