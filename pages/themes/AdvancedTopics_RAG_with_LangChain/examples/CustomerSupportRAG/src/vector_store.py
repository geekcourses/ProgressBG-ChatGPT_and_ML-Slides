from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Load the documents from your folder
loader = DirectoryLoader('../support_docs', glob="./*.txt", loader_cls=TextLoader)
raw_documents = loader.load()

# Create a text splitter that splits on paragraphs
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,  # Maximum size of each chunk
    chunk_overlap=0,  # No overlap between chunks
    separators=["\n\n"],  # Split on double newlines (paragraphs) first
    length_function=len,
)

# Split the text
docs = text_splitter.split_documents(raw_documents)

# Initialize the Embedding Model (Local HuggingFace)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 'persist_directory saves the index so you don't have to re-index every time.
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print(f"Successfully indexed {len(docs)} chunks into ChromaDB.")