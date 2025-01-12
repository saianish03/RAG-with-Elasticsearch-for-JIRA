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

import pandas as pd

def convert_csv_with_key_and_description(input_csv, output_csv):
    """
    Converts a CSV file into a new CSV with two columns: 'Key' and 'Description'.
    - The 'Key' column contains the 'Key' column value from the original dataset.
    - The 'Description' column contains the rest of the row data in natural text format.

    Input:
    input_csv: Path to the input CSV file.
    output_csv: Path to the output CSV file.

    Returns: None
    """

    df = pd.read_csv(input_csv, sep=";")
    
    key_column = df['Key']
    prio_column = df['Priority']
    status_column = df['Status']
    
    description_column = df.apply(
        lambda row: " ".join(
            f"{col} is {val} and" if pd.notnull(val) and str(val).strip() and col not in ["Key", "Priority", "Status"]
            else "" for col, val in row.items()
        ).strip().rstrip("and"),
        axis=1
    )
    
    result_df = pd.DataFrame({
        "key": key_column,
        "priority": prio_column,
        "status": status_column,
        "description": description_column
    })

    result_df.to_csv(output_csv, index=False, encoding="utf-8")


def docs_from_file(fi):
    """
    Function: To read data from input file and convert it into
    data LlamaIndex Document objects

    Input:
    fi - file name

    Returns: Document Objects
    """

    jira_issues = pd.read_csv(fi)

    documents = [Document(text=row['description'], metadata={"Issue key": row['key'], "priority": row['priority'], "status": row['status']}) for i, row in jira_issues.iterrows()]

    return documents

load_dotenv('.env')

# use Elasticsearchstore (a vectorstore) which uses the API and Cloud Key for automatic configuration of elastic search index and data management

elastic_vector_store = ElasticsearchStore(index_name='buildr-bug-fixes',
                                          vector_field='description_vector',
                                          text_field='description',
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

    convert_csv_with_key_and_description("./1.0.1/snapshot/buildr-full-bug-fix-dataset-fixed.csv", "./transformed-buildr-full-bug-fix.csv")
    documents = docs_from_file("./transformed-buildr-full-bug-fix.csv")
    embeddings = OllamaEmbedding('stablelm2')
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=350, chunk_overlap=50),
            embeddings
        ],
        vector_store=elastic_vector_store
    )

    print("Starting the pipeline....")
    pipeline.run(documents=documents)
    print("Done executing the pipeline!")


if __name__ == "__main__":
    main()