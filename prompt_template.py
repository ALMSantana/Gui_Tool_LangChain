from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.globals import set_debug
import os

set_debug(True)
load_dotenv()
MODEL_LLM= "gpt-4-0125-preview"

prompt_template = PromptTemplate.from_template(
        """
        Human: Você pode recomendar uma cidade do estado de {estado} para turismo {tipo_turismo}?
        AI: A cidade do {estado} para {tipo_turismo} é

        # FORMATO DE SAÍDA
        Apresente o nome da cidade e a atividade que será realizada
        """
    )

prompt = prompt_template.format(estado="São Paulo", tipo_turismo="Natureza")

llm = ChatOpenAI(model=MODEL_LLM,temperature=0.5, api_key=os.getenv("OPENAI_API_KEY"))
print(llm.invoke(prompt))