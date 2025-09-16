from langchain_astradb import AstraDBVectorStore
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from flipkart.data_converter import DataConverter
from config.config import Config

class DataIngestion:
    def __init__(self):
        self.embeddings = HuggingFaceEndpointEmbeddings(model=Config.HF_EMBEDDING_MODEL)
        self.vector_store = AstraDBVectorStore(
            collection_name="flipkart_db",
            api_endpoint=Config.ASTRA_DB_API_ENDPOINT,
            token=Config.ASTRA_DB_APPLICATION_TOKEN,
            namespace=Config.ASTRA_DB_KEYSPACE,
            embedding=self.embeddings
        )

    def ingest_data(self, load_existing: bool = True):
        if load_existing:
            return self.vector_store
        
        converter = DataConverter("data/flipkart_product_review.csv")
        documents = converter.convert_to_documents()
        self.vector_store.add_documents(documents)

if __name__ == "__main__":
    ingestion = DataIngestion()
    ingestion.ingest_data(load_existing=False)  