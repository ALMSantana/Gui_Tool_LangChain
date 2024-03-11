# informações de visto, saúde e segurança, restaurante, transporte, hospedagem, itinerário

#pip install langchain_text_splitters faiss-cpu
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.globals import set_debug
import os
from langchain.chains import RetrievalQA

load_dotenv()
MODEL_LLM= "gpt-4-0125-preview"

set_debug(True)

llm = ChatOpenAI(model=MODEL_LLM,temperature=0.5, api_key=os.getenv("OPENAI_API_KEY"))

# LER documento
loader = TextLoader("documentos/Passeios_Recife.txt")

documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(texts, embeddings)

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=db.as_retriever()
)

question = "Você pode indicar passeios para 3 dias de viagem com crianças?"
result = qa_chain(
    {"query": question}
    )

print(result["result"])