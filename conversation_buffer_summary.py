from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.globals import set_debug
import os
from langchain.memory import ConversationSummaryMemory

set_debug(True)
load_dotenv()
MODEL_LLM= "gpt-4-0125-preview"

llm = ChatOpenAI(model=MODEL_LLM,temperature=0.5, api_key=os.getenv("OPENAI_API_KEY"))

memory = ConversationSummaryMemory(llm=llm)
memory.save_context({"input": "Olá!"}, {"output": "Tudo bem?"})
memory.save_context({"input": "Em português: Você pode me recomendar locais de passeio para 7 dias com, e sem crianças no Brasil?"}, {"output": "Para uma viagem de 7 dias com crianças no Brasil, sugiro Rio de Janeiro, Gramado e Canela, Foz do Iguaçu e Porto de Galinhas. Para uma viagem sem crianças, considere Bonito, Búzios, Chapada dos Veadeiros e Fernando de Noronha. Cada destino oferece experiências únicas e diversas, desde praias deslumbrantes até aventuras na natureza e atrações culturais. Certifique-se de verificar as restrições de viagem e medidas de segurança antes de planejar sua viagem."})

print(memory.load_memory_variables({}))

