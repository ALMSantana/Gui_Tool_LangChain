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


template_1_cidade = PromptTemplate.from_template(
        """
        Human: Você pode recomendar uma cidade do estado de {estado} para turismo do tipo {tipo_turismo}
        AI: A cidade do {estado} indicada para o tipo de turismo {tipo_turismo} é

        # FORMATO DE SAÍDA
        Nome Cidade: nome da cidae
        Quantidade de Dias recomendados: N dias
        Recomendado para turistas que gostam de: descrever perfil do turista
        """
)

template_2_passeios = PromptTemplate.from_template(
        """
        Quais as principais atrações turísticas que você recomenda para a cidade {cidade}?

        Lista com 3 lugares para visitar.

        #FORMATO DE SAÍDA 
        Nome do Lugar: (nome do lugar)
        Custo Estimado: (custo do lugar)
        Recomendado para Crianças: sim ou não e porque.
        """
)


chain_1_cidade = (
    template_1_cidade
    | llm
    | StrOutputParser()
)

chain_2_passeios = (
    {"cidade": chain_1_cidade}
    | template_2_passeios
    | llm
    | StrOutputParser()
)

chain_2_passeios.invoke({"estado": "São Paulo", "tipo_turismo": "Natureza"})

