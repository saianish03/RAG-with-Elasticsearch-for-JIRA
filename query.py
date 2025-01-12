import warnings
warnings.filterwarnings('ignore')

from index import elastic_vector_store

from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import VectorStoreIndex, Response, Settings, QueryBundle

llm = Ollama(model="stablelm2")
Settings.embed_model = OllamaEmbedding("stablelm2")

index = VectorStoreIndex.from_vector_store(elastic_vector_store)
query_engine = index.as_query_engine(llm, similarity_top_k=10)

query="Give me information about Issue Key SRCTREEWIN-14221"

bundle = QueryBundle(query, embedding=Settings.embed_model.get_query_embedding(query))
result = query_engine.query(bundle)
print(result)