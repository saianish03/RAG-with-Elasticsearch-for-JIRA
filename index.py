import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import os
from llama_index.core import Document, Settings
from llama_index.core.ingestion import IngestionPipeline
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.elasticsearch import ElasticsearchStore
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv

def docs_from_file(fi):
    """
    Function: To read data from input file and convert it into
    data LlamaIndex Document objects

    Input:
    fi - file name

    Returns: Document Objects
    """

    jira_issues = pd.read_csv(fi)

    documents = [Document(text=row['Summary'], metadata={"Issue key": row['Issue key'], "Issue Type": row['Issue Type'], "Status": row['Status']}) for i, row in jira_issues.iterrows()]

    return documents

load_dotenv('.env')

# use Elasticsearchstore (a vectorstore) which uses the API and Cloud Key for automatic configuration of elastic search index and data management

elastic_vector_store = ElasticsearchStore(index_name='jira_issues',
                                          vector_field='summary_vector',
                                          text_field='summary',
                                          es_cloud_id=os.getenv("ELASTIC_CLOUD_ID"),
                                          es_api_key=os.getenv("ELASTIC_API_KEY"))

def main():
    """
    Function: To get document objects and use ollama embeddings
    from llama index for the summaries and create an ingestion pipeline
    in to perform transformations like chunking and then get embeddings
    and storing it in the ES vector store

    Input: None
    Returns: None
    """

    documents = docs_from_file("./GFG_FINAL.csv")
    embeddings = OllamaEmbedding('stablelm2')
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=100, chunk_overlap=10),
            embeddings
        ],
        vector_store=elastic_vector_store
    )

    print("Starting the pipeline....")
    pipeline.run(documents=documents)
    print("Done executing the pipeline!")


if __name__ == "__main__":
    main()