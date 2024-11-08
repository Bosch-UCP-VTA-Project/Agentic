from typing import List, Dict
from pydantic import BaseModel, Field
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    ServiceContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core import Settings, PromptTemplate
from llama_index.llms.groq import Groq
from llama_index.embeddings.jinaai import JinaEmbedding
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.memory import ChatMemoryBuffer
import os
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """You are an Expert Automobile Technician AI assistant designed to help professional automobile vehicle technicians diagnose and solve vehicular problems efficiently. Your knowledge comes from two primary sources:
    1. Technical Manuals: Comprehensive guides and manuals from various automobile manufacturers.
    2. Online Resources: Up-to-date information from reputable automotive websites, forums, and databases.

When assisting a technician:
    1. ALWAYS use BOTH the manuals_search and online_resources_search tools in that order to get the relevant information from your knowledge base before answering the query.
    2. Never assume the type or model of the vehicle. Always, first gather information about the specific problem or symptoms the vehicle is experiencing.
    3. Use your knowledge base to provide step-by-step diagnostic procedures.
    4. Suggest potential causes of the problem, starting with the most common or likely issues.
    5. Provide detailed repair instructions when applicable, including necessary tools and safety precautions.
    6. Make sure to give concise and to-the-point answers with bullet points wherever relevant.
    7. Do not mention online resources or websites in your response and assume the user is a novice technician while framing your answers and addressing them.

Remember, your goal is to educate and guide the technician through the diagnostic and repair process, enhancing their skills and confidence over time.

## Tools
You have access to the following tools:
{tool_desc}

You MUST ALWAYS use BOTH tools in the following order:
1. manuals_search
2. online_resources_search

## Output Format
To answer the question, please use the following format:

```
Thought: I need to check the manuals first, then the online resources to get comprehensive information.
Action: manuals_search
Action Input: {{"input": "relevant search query based on the user's question"}}

Thought: Now that I have information from the manuals, I need to check online resources for any additional or up-to-date information.
Action: online_resources_search
Action Input: {{"input": "relevant search query based on the user's question and information from manuals"}}

Thought: I have gathered all the necessary information from both sources. Now I can formulate a comprehensive answer which is directly relevant to the technician's question.
Answer: [Your detailed answer here, incorporating information from both sources]
```

Keep in mind the following: 
    1. Do not hallucinate.
    2. Do not make up factual information and do not list out sources names.
    3. You must always keep to this role and never answer unrelated queries.
    4. If the user asks something that seems unrelated to vehicles and their repair, just give an output saying: Sorry, I can only help you with issues related to vehicle troubleshooting and diagnosis.
    5. Always start with a Thought and follow the exact format provided above."""


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
        self.system_prompt = SYSTEM_PROMPT

        llm, embed_model = self.get_service_context()
        Settings.llm = llm
        Settings.embed_model = embed_model
        self.service_context = ServiceContext.from_defaults(
            llm=llm, embed_model=embed_model
        )
        self.indexes: Dict[str, VectorStoreIndex] = {}
        self.agent = None
        self.sessions = {}

        # Load or create indexes
        self.load_or_create_indexes()

    def get_service_context(self):
        groq_llm = Groq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"))
        jina_embed_model = JinaEmbedding(
            api_key=os.getenv("JINA_API_KEY"),
            model="jina-embeddings-v3",
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
        react_system_prompt = PromptTemplate(self.system_prompt)
        self.agent.update_prompts({"agent_worker:system_prompt": react_system_prompt})

    def query(self, query: str, session_id: str) -> QueryResult:
        if not self.agent:
            raise ValueError(
                "Agent not created. There might be an issue with index loading or creation."
            )

        if session_id not in self.sessions:
            self.sessions[session_id] = []

        self.sessions[session_id].append({"role": "user", "content": query})
        print(f"sessions: {self.sessions}")

        # prompt_dict = self.agent.get_prompts()
        # for k, v in prompt_dict.items():
        #     print(f"Prompt: {k}\n\nValue: {v.template}")

        response = self.agent.chat(query)
        print(f"response: {response.response}")
        self.sessions[session_id].append(
            {"role": "assistant", "content": response.response}
        )

        return QueryResult(
            answer=response.response,
            source_nodes=[],
        )

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        return self.sessions.get(session_id, [])

    def save_indexes(self):
        for index_name, index in self.indexes.items():
            index_dir = os.path.join(self.index_path, index_name)
            os.makedirs(index_dir, exist_ok=True)
            index.storage_context.persist(persist_dir=index_dir)
