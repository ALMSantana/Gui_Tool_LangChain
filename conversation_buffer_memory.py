from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from langchain.globals import set_debug
import os

set_debug(True)
load_dotenv()
MODEL_LLM= "gpt-4-0125-preview"

llm = ChatOpenAI(model=MODEL_LLM,temperature=0.5, api_key=os.getenv("OPENAI_API_KEY"))
memory = ConversationBufferMemory()
conversation = ConversationChain(llm=llm, memory=memory, verbose=False)

response_1 = conversation.predict(input="Você pode recomendar cidades ou locais para uma viagem no Brasil para 7 dias de viagem, com e sem crianças?")
print(response_1)

print(memory.load_memory_variables({})) #dados ficam salvos aqui