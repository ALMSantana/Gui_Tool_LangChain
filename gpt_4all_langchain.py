#pip install --upgrade --quiet  gpt4all > /dev/null
# https://gpt4all.io/index.html
# https://huggingface.co/Pi3141/alpaca-native-7B-ggml/tree/397e872bf4c83f4c642317a5bf65ce84a105786e

from langchain.prompts import PromptTemplate
from langchain_community.llms import GPT4All
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.globals import set_debug

set_debug(True)

# "orca-mini-3b-gguf2-q4_0.gguf" 
local_path = (
    "modelos/mistral-7b-instruct-v0.1.Q4_0.gguf" 
)

llm = GPT4All(model=local_path)


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