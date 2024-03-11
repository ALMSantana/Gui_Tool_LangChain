from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from langchain.globals import set_debug
import os

set_debug(True)
load_dotenv()
MODEL_LLM= "gpt-4-0125-preview"

llm = ChatOpenAI(model=MODEL_LLM,temperature=0.5, api_key=os.getenv("OPENAI_API_KEY"))
template = """Você é um chatbot que ajuda pessoas com dicas de viagens.

Histórico da Conversa:
{chat_history}

Nova Pergunta: {pergunta}
Resposta:"""
prompt = PromptTemplate.from_template(template)

memory = ConversationBufferMemory(memory_key="chat_history")
conversation = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
    memory=memory
)

estado = input("Informe um estado para passeio: ")
tipo_turismo = input("Indique um tipo de turismo (ex.: natureza, compras, dança, etc): ")

response = conversation({"pergunta": f"Você pode recomendar uma cidade do estado de {estado} para atividades relacionadas a {tipo_turismo}?"})
print("\n-------------")
print(memory.load_memory_variables({}))
print("\n-------------")
print("\nResposta\n", response)
response = conversation({"pergunta": f"Agora você pode me indicar uma atração ou atividade para executar nessa cidade, relacionada a este tipo de turismo?"})["text"]
print(memory.load_memory_variables({})) 
print("\n-------------")
print("\nResposta\n", response)