from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
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

message_1 = f"Você pode recomendar uma cidade do estado de {estado} para atividades relacionadas a {tipo_turismo}?"
response_1 = conversation({"pergunta": message_1})["text"]

message_2 = f"Agora você pode me indicar uma atração ou atividade para executar nessa cidade, relacionada a este tipo de turismo?"
response_2 = conversation({"pergunta": message_2})["text"]

history = ChatMessageHistory()
history.add_user_message(message_1)
history.add_ai_message(response_1)

history.add_user_message(message_2)
history.add_ai_message(response_2)

print("\nHistórico")
print(history.messages)