from langchain.llms import HuggingFaceHub
from langchain import PromptTemplate, LLMChain
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceHub(
    repo_id="huggingfaceh4/zephyr-7b-alpha", 
    model_kwargs={"temperature": 0.5, "max_length": 64,"max_new_tokens":512}
)

query = "Qual a cidade do Tocantins com maior população?"

prompt = f"""
 <|system|>
Você é um assistente que responde em português de forma bastante cordial.
</s>
 <|user|>
 {query}
 </s>
 <|assistant|>
"""

print(llm.invoke(prompt))








prompt_template = PromptTemplate.from_template(
    """
    Human: Qual a cidade do {estado} com maior {categoria}?
    AI: A cidade do estádo {estado} com maior {categoria} é
    """
)

prompt = prompt_template.format(estado="Tocantins", categoria="população")

llm = HuggingFaceHub(
    repo_id="huggingfaceh4/zephyr-7b-alpha", 
    model_kwargs={"temperature": 0.5, "max_length": 64,"max_new_tokens":512}
)
response = llm.invoke(prompt)

print(response)