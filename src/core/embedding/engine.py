from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from typing import Dict


def select_potential_context(embedding_model: OpenAIEmbeddings, question: str, datas: Dict[str, str], top_k=10):
    original_data_context = [sample.context for sample in datas]
    original_context_embedding = embedding_model.embed_documents(original_data_context)
    original_pairs = list(zip(original_data_context, original_context_embedding))

    faiss = FAISS.from_embeddings(original_pairs, embedding_model)
    search_res = faiss.similarity_search(question, k=top_k)
    
    search_res = [res.page_content for res in search_res]
        
    filtered_datas = [element for element in datas if element.context in search_res]
    return filtered_datas