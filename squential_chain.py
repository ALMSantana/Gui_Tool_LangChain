from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.globals import set_debug
from langchain.chains import LLMChain, SequentialChain
import os

set_debug(True)
load_dotenv()
MODEL_LLM= "gpt-4-0125-preview"

llm = ChatOpenAI(model=MODEL_LLM,temperature=0.5, api_key=os.getenv("OPENAI_API_KEY"))

template_1_destino = """
Liste um destino popular no estado de {nome_estado} para quem gosta de {hobby}.

# FORMATO DE SAÍDA
Apenas o nome do destino
"""
prompt_1_destino = PromptTemplate(input_variables=["nome_estado", "hobby"],
                                template=template_1_destino)
cadeia_1_destino = LLMChain(llm=llm,
                        prompt=prompt_1_destino,
                        output_key="destino",
    verbose=True)

template_2_passeio = """
Qual é o melhor lugar em {destino} para {hobby}?

# FORMATO DE SAÍDA
Sugestão de passeio e uma breve descrição do que fazer, incluindo preço
"""
prompt_2_passeio = PromptTemplate(input_variables=["destino", "hobby"],
                            template=template_2_passeio)
cadeia_2_passeio = LLMChain(llm=llm,
                        prompt=prompt_2_passeio,
                        output_key="sugestao_passeio",
    verbose=True)

cadeia_final = SequentialChain(
    chains=[cadeia_1_destino, cadeia_2_passeio],
    input_variables=["nome_estado", "hobby"],
    output_variables=["destino", "sugestao_passeio"],
    verbose=True
)
print(cadeia_final.invoke({"nome_estado": "São Paulo", "hobby": "trilha"}))