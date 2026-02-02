from src.state.rag_state import RAGState


class RAGNodes:
    """ Contains node function for RAG workflow """ 


    def __init__(self , retriever, llm) :
        """ 
        Initialize RAG NODES        
        """

        self.retriever  = retriever
        self.llm  = llm

    def retrieve_docs(self , state : RAGState) -> RAGState :
        """ 
        Retriever relevant document node

        Args : 
            State : Current RAG state

        Returns :
            Updated RAG state with retrived documents
        """

        docs = self.retriever.invoke(state.question)
        
        return RAGState(
            question= state.question,
            retrieved_docs= docs 
        )



    def generate_answer(self, state : RAGState) -> RAGState :
        """ 
        Generate answer from retrieved documents node

        Args  :
            state : Current RAG state with retrieved documents

        Returns  :
            Updated RAG state with generated answer
        """
        context = "\n\n". join([doc.page_content for doc in state.retrieved_docs])
        prompt = """ 
        Answer the question based on the context.
        Context : {context}

        Question : {state.question}
        """

        # Generate Response :
        response = self.llm.invoke(prompt)

        return RAGState(
            question = state.question,
            retrieved_docs = state.retrieved_docs,
            answer = response.content
        )




