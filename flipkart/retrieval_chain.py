from langchain_groq import ChatGroq
from langchain.chains import create_history_aware_retriever,create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from config.config import Config

class RetrievalChainBuilder:
    def __init__(self,vector_store):
        self.vector_store = vector_store
        self.model = ChatGroq(model=Config.GROQ_MODEL,temperature=0.5)
        self.history_store = {}

    def _get_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.history_store:
            self.history_store[session_id] = ChatMessageHistory()
        return self.history_store[session_id]
    
    def build_chain(self) -> RunnableWithMessageHistory:
        retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k":3})

        context_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given the chat history and user question, rewrite it as a standalone question."),
            MessagesPlaceholder(variable_name="chat_history"), 
            ("human", "{input}")  
        ])

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", """You're an e-commerce bot answering product-related queries using reviews and titles.
                          Stick to context. Be concise and helpful.\n\nCONTEXT:\n{context}\n\nQUESTION: {input}"""),
            MessagesPlaceholder(variable_name="chat_history"), 
            ("human", "{input}")  
        ])

        history_aware_retriever = create_history_aware_retriever(
            retriever = retriever,
            llm = self.model,
            prompt = context_prompt
        )

        question_chain = create_stuff_documents_chain(
            llm = self.model,
            prompt = qa_prompt
        )

        retriever_chain = create_retrieval_chain(history_aware_retriever, question_chain)

        return RunnableWithMessageHistory(
            retriever_chain,
            get_session_history=self._get_history,
            input_messages_key="input",
            output_message_key="answer",
            history_messages_key="chat_history"            
        )
