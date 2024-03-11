from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
from langchain.globals import set_debug
import os

load_dotenv()
MODEL_LLM= "gpt-4-0125-preview"

set_debug(True)

llm = ChatOpenAI(model=MODEL_LLM,temperature=0.5, api_key=os.getenv("OPENAI_API_KEY"))


template_turismo = PromptTemplate.from_template(
        """
        Human: Você pode recomendar uma cidade do estado de {estado} para turismo?
        AI: A cidade do {estado} indicada é

        # FORMATO DE SAÍDA
        Apresente o nome da cidade e a atividade que será realizada
        """
)

chain = (
    {"estado": RunnablePassthrough()} 
    | template_turismo
    | llm
    | StrOutputParser()
)

chain.invoke("Pernambuco")

