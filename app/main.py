#from core.agent import Agent
#from langchain_core.messages import HumanMessage
from db.vectordb import VectorRepository
import time

'''
graph = Agent().make()

while True:
    query = input()

    answer = graph.invoke({"messages" : [HumanMessage(query)], "citation_links" : None}, config={"configurable": {"thread_id": "haha12020"}})
'''

start = time.perf_counter()
with VectorRepository(location_to_save="./db/tmp/chemistry_collection", 
                                 collection_name="chemistry_collection") as repo:
    results = repo.get_retriever().invoke("Что такое атом?")
end = time.perf_counter()

print(f"{results[0]}\n execution time (sec): {(end - start):.4f}")

'''
start = time.perf_counter()
startdb = VectorRepository(location_to_save="./db/tmp/chemistry_collection", 
                                 collection_name="chemistry_collection").startup_db()
end = time.perf_counter()

print(f"\nexecution time (sec): {(end - start):.4f}")
'''