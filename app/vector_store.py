import chromadb
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import dotenv
from uuid import uuid4
from langchain_core.documents import Document


# Load environment variables from .env file
dotenv.load_dotenv()


#Setting up Google API
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db"
)

def add_documents_to_vector_store(documents):
    uuids = [str(uuid4()) for _ in range(len(documents))]
    vector_store.add_documents(documents=documents, ids=uuids)

document_11 = Document(
    page_content="Patient reports persistent cough for the past two weeks. Denies fever or chest pain. Recommended chest X-ray.",
    metadata={"source": "clinical_note"},
    id=11,
)

document_12 = Document(
    page_content="A recent study shows that daily exercise reduces the risk of heart disease by 35%.",
    metadata={"source": "research"},
    id=12,
)

document_13 = Document(
    page_content="Just got my flu shot today. Arm is sore but glad I’m protected for the season.",
    metadata={"source": "tweet"},
    id=13,
)

document_14 = Document(
    page_content="Breaking: The CDC reports a rise in COVID-19 cases this winter, urging people to get boosters.",
    metadata={"source": "news"},
    id=14,
)

document_15 = Document(
    page_content="MRI results show a small herniated disc at L4-L5. Physical therapy recommended before considering surgery.",
    metadata={"source": "clinical_note"},
    id=15,
)

document_16 = Document(
    page_content="Top 5 superfoods that can help boost your immune system naturally.",
    metadata={"source": "website"},
    id=16,
)

document_17 = Document(
    page_content="Clinical trial results: The new Alzheimer’s drug slowed cognitive decline by 20% in early-stage patients.",
    metadata={"source": "research"},
    id=17,
)

document_18 = Document(
    page_content="Why does my head hurt every morning? Here are 7 common causes of recurring headaches.",
    metadata={"source": "website"},
    id=18,
)

document_19 = Document(
    page_content="Hospital reports show a 15% increase in asthma-related ER visits during allergy season.",
    metadata={"source": "news"},
    id=19,
)

document_20 = Document(
    page_content="Feeling anxious before my surgery tomorrow. Hope everything goes smoothly 🙏",
    metadata={"source": "tweet"},
    id=20,
)



documents = [document_11, document_12, document_13, document_14, document_15, 
            document_16, document_17, document_18, document_19, document_20]

# Add documents to the vector store
add_documents_to_vector_store(documents)