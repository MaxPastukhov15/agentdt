from langchain.tools import tool
from langchain_core.vectorstores import VectorStoreRetriever
from db.vectordb import VectorRepository
from typing import List
from state import AgentState

@tool
def search_chemistry_collection(query: str):

    retriever: VectorStoreRetriever = VectorRepository(location_to_save="./db/tmp/chemistry_collection", 
                                 collection_name="chemistry_collection").get_retriever()    
    results = retriever.invoke(query)

    context = "\n\n".join([
        f"{doc.metadata['title']}: {doc.page_content}" for doc in results]
        )
    
    return context


TOOLS = [search_chemistry_collection]