from langgraph.graph import StateGraph, END
from src.state.rag_state import RAGState
from src.nodes.reActnode import RAGNodes

class GraphBuilder() :
    """ Builds and manages the LangGraph Workflow """


    def __init(self, retriever, llm) :
        """ Initialize Graph Builder """
        self.nodes = None
        self.graph = None 


    def build(self) :
        """ 
        Build the RAG workflow graph
        
        Returns :
            Compiled Graph Instance
        """
        # Create  state graph
        builder = StateGraph(RAGState)

        # Add nodes
        builder.add_node("retriever", self.nodes.retriever_docs)
        builder.add_node("responder", self.nodes.generate_answer)


        # Entry Point
        builder.set_entry_point("retriever")

        # Add Edges
        builder.add_edge("retriever", "responder")
        builder.add_edge("responder", END)

        # compile Graph
        self.graph = builder.compile()
        return self.graph

