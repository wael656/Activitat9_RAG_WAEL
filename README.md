# Actividad 9: Remake con RAG

**Descripción:** Esta aplicación es un chatbot avanzado que utiliza RAG para responder preguntas sobre los temas de la asignatura de IA y datos del instituto.

**Pipeline RAG:**
1. **Carga:** Usamos `DirectoryLoader` para leer archivos `.txt`.
2. **Chunking:** Fragmentamos el texto con `CharacterTextSplitter`.
3. **Embeddings:** Convertimos el texto a vectores con el modelo `nomic-embed-text` de Ollama.
4. **Almacenamiento:** Guardamos los vectores en una base de datos `ChromaDB` persistente.
5. **Generación:** Usamos `Llama 3.2` para generar respuestas basadas en el contexto recuperado.

**Instrucciones de ejecución:**
1. Tener Ollama instalado y ejecutando.
2. Ejecutar: `ollama pull llama3.2` y `ollama pull nomic-embed-text`.
3. Instalar librerías: `pip install chromadb ollama gradio langchain-community`.
4. Ejecutar: `python main.py`.

<img width="1912" height="822" alt="Captura de pantalla 2026-04-09 130919" src="https://github.com/user-attachments/assets/6cc93420-2e7a-470a-8b42-4012eff4adf7" />
