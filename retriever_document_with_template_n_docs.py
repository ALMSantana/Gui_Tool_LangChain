# informações de visto, saúde e segurança, restaurante, transporte, hospedagem, itinerário
# Retrieval-Augmented Generation
#pip install -U langchain-community
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.globals import set_debug
import os
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()
MODEL_LLM= "gpt-4-0125-preview"

set_debug(True)

llm = ChatOpenAI(model=MODEL_LLM,temperature=0.5, api_key=os.getenv("OPENAI_API_KEY"))

loaders = [
    PyPDFLoader("documentos/Fernando de Noronha.pdf"),
    PyPDFLoader("documentos/Passeios_Recife_Guia.pdf"),
    PyPDFLoader("documentos/Porto de Galinhas.pdf")
]
docs = []
for loader in loaders:
    docs.extend(loader.load())


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 0
)

texts = text_splitter.split_documents(docs)

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(texts, embeddings)


prompt_template_pernambuco = """
    Priorize indicar passeios citados no: {context} e envolva pelo menos duas cidades diferentes.
    
    Pergunta: {question}

    # FORMATO DE SAÍDA
    Itinenário para quantidade de dias informada

    ### Dia 1: nome do passeio
    - Cidade: nome da cidade
    - Acesso: como chegar
    - Manhã: descrição
    - Tarde: descrição
    """

qa_chain_pernambuco = PromptTemplate.from_template(prompt_template_pernambuco)

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=db.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": qa_chain_pernambuco}
)

qtd_dias = input("Informe a quantidade de dias: ")
estado = input("Informe o estado desejado: ")
tipo_turismo = input("Informe o tipo de turismo desejado (natureza, compras...): ")


question = f"Você pode recomendar passeios para cidades próximas dentro do estado de {estado} para turismo {tipo_turismo} para {qtd_dias} dias?"
result = qa_chain({"query": question})
print(result["result"])

