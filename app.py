from flipkart.data_ingestion import DataIngestion
from flipkart.retrieval_chain import RetrievalChainBuilder
from flask import Flask, request, render_template, Response
from prometheus_client import Counter, generate_latest

from dotenv import load_dotenv
load_dotenv()

def create_app():

    app = Flask(__name__)

    response_counter = Counter('responses_total', 'Total number of responses served')   

    ingestion = DataIngestion()
    vector_store = ingestion.ingest_data(load_existing=True)
    chain_builder = RetrievalChainBuilder(vector_store)
    rag_chain = chain_builder.build_chain()

    @app.route('/')
    def init_index(): 
        return render_template('index.html')
    
    @app.route('/metrics')
    def metrics():
        return generate_latest(response_counter), 200, {'Content-Type': 'text/plain; version=0.0.4; charset=utf-8'}

    @app.route('/get', methods=['POST'])
    def get_response():
        input = request.form['msg']
        response = rag_chain.invoke({"input": input}, config={"configurable": {"session_id": "default_session"}})
        answer = response['answer']

        if answer:
            response_counter.inc()

        return answer

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5001, debug=True)