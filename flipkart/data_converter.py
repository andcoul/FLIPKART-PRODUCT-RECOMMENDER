import pandas as pd
from langchain.schema import Document

class DataConverter:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def convert_to_documents(self) -> list[Document]:
        df = pd.read_csv(self.file_path)
        documents = []
        for _, row in df.iterrows():
            content = f"Product Name: {row['product_title']}\nDescription: {row['review']}"
            metadata = {
                "product_name": row['product_title']
            }
            documents.append(Document(page_content=content, metadata=metadata))
        return documents
    

if __name__ == "__main__":
    converter = DataConverter("data/flipkart_product_review.csv")
    docs = converter.convert_to_documents()
    for doc in docs[:10]:  # Print first 2 documents for verification
        print(doc.page_content)
        print(doc.metadata)
        print("-----")