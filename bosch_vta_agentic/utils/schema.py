from typing import List, Dict
from pydantic import BaseModel, Field
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    ServiceContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core import Settings
from llama_index.llms.groq import Groq
from llama_index.embeddings.jinaai import JinaEmbedding
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.memory import ChatMemoryBuffer
import os
from dotenv import load_dotenv

load_dotenv()


class Document(BaseModel):
    content: str = Field(..., description="The content of the document")
    metadata: dict = Field(
        default_factory=dict, description="Metadata associated with the document"
    )


class QueryResult(BaseModel):
    answer: str = Field(..., description="The answer to the query")
    source_nodes: List[str] = Field(
        ..., description="The source nodes used to generate the answer"
    )


class AutoTechnicianRAG:
    def __init__(
        self,
        manuals_path: str,
        online_resources_path: str,
        index_path: str,
    ):
        self.manuals_path = manuals_path
        self.online_resources_path = online_resources_path
        self.index_path = index_path
        self.system_prompt = """
        You are an Expert Automobile Technician AI assistant designed to help novice technicians diagnose and solve vehicular problems efficiently. Your knowledge comes from two primary sources:

        1. Technical Manuals: Comprehensive guides and manuals from various automobile manufacturers.
        2. Online Resources: Up-to-date information from reputable automotive websites, forums, and databases.

        When assisting a novice technician:
        1. Always start by asking for the vehicle make, model, and year if not provided.
        2. Gather information about the specific problem or symptoms the vehicle is experiencing.
        3. Use your knowledge base to provide step-by-step diagnostic procedures.
        4. Explain technical terms in simple language that a novice can understand.
        5. Suggest potential causes of the problem, starting with the most common or likely issues.
        6. Provide detailed repair instructions when applicable, including necessary tools and safety precautions.
        7. If a problem seems too complex for a novice, advise seeking help from a more experienced technician.
        8. Always prioritize safety in your recommendations.

        Remember, your goal is to educate and guide the novice technician through the diagnostic and repair process, enhancing their skills and confidence over time.
        """

        llm, embed_model = self.get_service_context()
        Settings.llm = llm
        Settings.embed_model = embed_model
        self.service_context = ServiceContext.from_defaults(
            llm=llm, embed_model=embed_model
        )
        self.indexes: Dict[str, VectorStoreIndex] = {}
        self.agent = None

        # Load or create indexes
        self.load_or_create_indexes()

    def get_service_context(self):
        groq_llm = Groq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"))
        jina_embed_model = JinaEmbedding(
            api_key=os.getenv("JINA_API_KEY"),
            model="jina-embeddings-v2-base-en",
        )
        return groq_llm, jina_embed_model

    def load_or_create_indexes(self):
        if self.check_existing_indexes():
            self.load_indexes()
        else:
            self.create_indexes()
        self._create_agent()

    def check_existing_indexes(self) -> bool:
        return all(
            os.path.exists(os.path.join(self.index_path, index_name))
            for index_name in ["manuals", "online_resources"]
        )

    def load_indexes(self):
        print("Loading existing indexes...")
        for index_name in ["manuals", "online_resources"]:
            index_dir = os.path.join(self.index_path, index_name)
            storage_context = StorageContext.from_defaults(persist_dir=index_dir)
            self.indexes[index_name] = load_index_from_storage(storage_context)
        print("Indexes loaded successfully.")

    def create_indexes(self):
        print("Creating new indexes...")
        # Load and index technical manuals
        manuals_documents = SimpleDirectoryReader(
            self.manuals_path,
            recursive=True,
        ).load_data()
        if not manuals_documents:
            raise ValueError("No manual documents loaded. Check the manuals path.")
        self.indexes["manuals"] = VectorStoreIndex.from_documents(
            manuals_documents,
            service_context=self.service_context,
        )

        # Load and index online resources
        online_resources_documents = SimpleDirectoryReader(
            self.online_resources_path,
            recursive=True,
        ).load_data()
        if not online_resources_documents:
            raise ValueError(
                "No online resource documents loaded. Check the online resources path."
            )
        self.indexes["online_resources"] = VectorStoreIndex.from_documents(
            online_resources_documents,
            service_context=self.service_context,
        )

        # Save the newly created indexes
        self.save_indexes()
        print("New indexes created and saved successfully.")

    def _create_agent(self):
        tools = []
        for index_name, index in self.indexes.items():
            query_engine = index.as_query_engine(similarity_top_k=10)
            tool = QueryEngineTool(
                query_engine=query_engine,
                metadata=ToolMetadata(
                    name=f"{index_name}_search",
                    description=f"Search the {index_name} for detailed automotive technical information and repair procedures.",
                ),
            )
            tools.append(tool)

        self.agent = ReActAgent.from_tools(
            tools,
            llm=self.service_context.llm,
            verbose=True,
            system_prompt=self.system_prompt,
            memory=ChatMemoryBuffer.from_defaults(token_limit=4096),
        )

    def query(self, query: str) -> QueryResult:
        if not self.agent:
            raise ValueError(
                "Agent not created. There might be an issue with index loading or creation."
            )
        response = self.agent.chat(query)
        return QueryResult(
            answer=response.response,
            source_nodes=[],  # Note: ReActAgent doesn't provide source nodes directly
        )

    def save_indexes(self):
        for index_name, index in self.indexes.items():
            index_dir = os.path.join(self.index_path, index_name)
            os.makedirs(index_dir, exist_ok=True)
            index.storage_context.persist(persist_dir=index_dir)


# Example usage
# if __name__ == "__main__":
#     auto_tech_rag = AutoTechnicianRAG(
#         manuals_path="bosch_vta_agentic/data/technical_manuals",
#         online_resources_path="bosch_vta_agentic/data/online_resources",
#         index_path="indexes"
#     )

#     # The indexes will be loaded if they exist, or created if they don't

#     result = auto_tech_rag.query("I have a 2018 Toyota Camry that won't start. The engine cranks but doesn't turn over. What should I check first?")
#     print(result.answer)

#     # For subsequent runs, just initialize the pipeline again:
#     # It will automatically load the existing indexes instead of recreating them
#     auto_tech_rag = AutoTechnicianRAG(
#         manuals_path="bosch_vta_agentic/data/technical_manuals",
#         online_resources_path="bosch_vta_agentic/data/online_resources",
#         index_path="indexes"
#     )
#     result = auto_tech_rag.query("How do I check and replace the spark plugs on a 2015 Honda Civic?")
#     print(result.answer)
