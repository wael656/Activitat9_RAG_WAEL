import os
import chromadb
import ollama
import gradio as gr
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter

DB_PATH = "./chroma_db"
DATA_PATH = "./knowledge-base/"

client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection(name="mi_conocimiento")

def cargar_documentos():
    if collection.count() == 0:
        print("Indexando documentos en ChromaDB...")
        loader = DirectoryLoader(DATA_PATH, glob="**/*.txt", loader_cls=TextLoader)
        docs = loader.load()
        
        splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)
        
        for i, chunk in enumerate(chunks):
            res = ollama.embeddings(model="nomic-embed-text", prompt=chunk.page_content)
            collection.add(
                ids=[f"id_{i}"],
                embeddings=[res['embedding']],
                documents=[chunk.page_content]
            )
        print(f"{collection.count()} documentos guardados.")
    else:
        print(f" Base de datos lista con {collection.count()} documentos.")

cargar_documentos()

def funcion_chat_rag(mensaje, historial, n_resultados):
    query_vector = ollama.embeddings(model="nomic-embed-text", prompt=mensaje)['embedding']
    
    results = collection.query(query_embeddings=[query_vector], n_results=int(n_resultados))
    contexto = "\n\n".join(results['documents'][0])
    
    prompt_con_contexto = f"""
    Eres un asistente del ITICBCN. Responde la PREGUNTA usando SOLO el CONTEXTO proporcionado.
    Si la respuesta no está en el contexto, di que no lo sabes.
    
    CONTEXTO:
    {contexto}
    
    PREGUNTA: {mensaje}
    """
    response = ollama.generate(model='llama3.2', prompt=prompt_con_contexto)
    return response['response']

with gr.Blocks(title="Asistente RAG ITICBCN") as demo:
    gr.Markdown("# RAG Explorer - Unidad IA")
    gr.Markdown("Este sistema consulta documentos locales sobre IA y el Instituto.")
    
    with gr.Row():
        with gr.Column(scale=1):
            slider_n = gr.Slider(1, 5, value=3, step=1, label="Fragmentos a consultar (N)")
            gr.Markdown("**Instrucciones:** Ajusta el slider para dar más o menos contexto a la IA.")
            
        with gr.Column(scale=3):
            chat = gr.ChatInterface(
                fn=funcion_chat_rag,
                additional_inputs=[slider_n],
            )

if __name__ == "__main__":
    demo.launch(debug=True)