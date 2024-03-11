from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.globals import set_debug
from langchain.chains import LLMChain, SimpleSequentialChain
import os

set_debug(True)
load_dotenv()
MODEL_LLM= "gpt-4-0125-preview"

llm = ChatOpenAI(model=MODEL_LLM,temperature=0.5, api_key=os.getenv("OPENAI_API_KEY"))

prompt_1_cidade = PromptTemplate.from_template(
    """
    Qual cidade do estado de {estado} você recomenda para uma viagem de 7 dias?

    #Foramto de Saída
    Apenas o nome da Cidade Recomendada
    """


)
cadeia_1_cidade = LLMChain(llm=llm, prompt=prompt_1_cidade)

prompt_2_passeios = PromptTemplate.from_template(
"""
Quais as principais atrações turísticas que você recomenda para a cidade {cidade}?

#FORMATO DE SAÍDA 
{cidade}
Lista com 5 lugares para visitar
"""


)
cadeia_2_passeios = LLMChain(llm=llm, prompt=prompt_2_passeios)
cadeia_linear = SimpleSequentialChain(chains=[cadeia_1_cidade, cadeia_2_passeios],
                                            verbose=True)

print(cadeia_linear.invoke("São Paulo"))