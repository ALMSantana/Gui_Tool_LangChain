# informações de visto, saúde e segurança, restaurante, transporte, hospedagem, itinerário
# Retrieval-Augmented Generation
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
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()
MODEL_LLM= "gpt-4-0125-preview"

set_debug(True)

llm = ChatOpenAI(model=MODEL_LLM,temperature=0.5, api_key=os.getenv("OPENAI_API_KEY"))

# LER documento
loader = PyPDFLoader("documentos/Porto de Galinhas.pdf")

documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(texts, embeddings)

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=db.as_retriever()
)

prompt_template_recife = """
    Priorize indicar passeios citados no: {context}
    
    Pergunta: {question}

    # FORMATO DE SAÍDA
    Itinenário para quantidade de dias informada

    ### Dia 1: nome do passeio
    - Manhã: descrição
    - Tarde: descrição
    """

qa_chain_recife = PromptTemplate.from_template(prompt_template_recife)

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=db.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": qa_chain_recife}
)

qtd_dias = input("Informe a quantidade de dias: ")
cidade = input("Informe a cidade desejada: ")
tipo_turismo = input("Informe o tipo de turismo desejado (natureza, compras...): ")


question = f"Você recomendar passeios para a de {cidade} para quem curte turismo {tipo_turismo} para {qtd_dias} dias?"
result = qa_chain({"query": question})
print(result["result"])
