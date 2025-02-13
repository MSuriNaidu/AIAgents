import typer
from typing import Optional, List
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
from phi.vectordb.pgvector import PgVector2
from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.assistant import Assistant
from phi.storage.assistant.postgres import PgAssistantStorage
from sentence_transformers import SentenceTransformer  # ✅ Use local embeddings
from dotenv import load_dotenv
import os

# ✅ Load environment variables
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["PHI_MODEL_PROVIDER"] = "groq"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ✅ Define local database
db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"

# ✅ Load the local embedding model (No API calls)
local_embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ✅ Custom Embedder Class (Ensures Local Processing)
class CustomEmbedder:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.dimensions = self.model.get_sentence_embedding_dimension()  # ✅ Set dimensions

    def get_embedding(self, text: str):
        return self.model.encode(text).tolist()

    def get_embedding_and_usage(self, text: str):
        embedding = self.get_embedding(text)
        return embedding, {"tokens_used": len(text.split())}  # ✅ Simulating token usage data

embedder = CustomEmbedder()

# ✅ Ensure vector DB uses LOCAL embeddings
knowledge_base = PDFUrlKnowledgeBase(
    urls=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    model=Groq(id="llama-3.3-70b-versatile"),  # ✅ Uses Groq Llama 3 (FREE)
    vector_db=PgVector2(collection="recipes", db_url=db_url, embedder=embedder)  # ✅ Uses LOCAL embeddings
)

# ✅ Load knowledge base (No API Calls)
knowledge_base.load(recreate=True, upsert=True)

storage = PgAssistantStorage(
    table_name="pdf_assistant",
    #model=Groq(id="llama-3.3-70b-versatile"),
    db_url=db_url
)

groq_model = Groq(
    id="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

def pdf_assistant(new: bool=True, user:str = "user"):
    run_id: Optional[str] = None

    if not new:
        existing_run_ids: List[str] = storage.get_all_run_ids(user);
        if len(existing_run_ids) > 0:
            run_id = existing_run_ids[0]

    assistant = Assistant(
        run_id=run_id,
        user_id=user,
        knowledge_base=knowledge_base,
        model=groq_model,
        storage=storage,
        show_tool_calls=True,
        search_knowledge=True,
        read_chat_history=True
    )

    if run_id is None:
        run_id=assistant.run_id
        print(f"Started running: {run_id}\n");
    else:
        print(f"continuing run: {run_id}\n")
    assistant.cli_app(markdown=True)

if __name__=="__main__":
    typer.run(pdf_assistant)
