""" Vector store module for document embedding and retrieval """

from typing import List
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document



class VectorStore :
    """ Manages Vector store applications  """

    def __init__(self) :
        self.embedding = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
        self.vectorestore= None
        self.retriever = None


    def create_retriever(self, documents : List[Document]) :
        """ 
        Create vector store from document
        
        Args : 
            documents :   List of Document to embed
        """

        self.vectorestore = FAISS.from_documents(documents, self.embedding)
        self.retriever = self.vectorstore.as_retriever()


    def get_retriever(self) :
        """
        Get the retriever instance 

        Returns :
            Retriever instance
        """

        if self.retriever is None :
            raise ValueError("Vectore store not initialsed. Call create_retriever first.")

        return  self.retriever

    
