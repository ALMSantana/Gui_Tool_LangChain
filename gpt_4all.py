#pip install --upgrade --quiet  gpt4all > /dev/null
# https://gpt4all.io/index.html
# https://huggingface.co/Pi3141/alpaca-native-7B-ggml/tree/397e872bf4c83f4c642317a5bf65ce84a105786e

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from gpt4all import GPT4All


model = GPT4All("mistral-7b-instruct-v0.1.Q4_0.gguf", "modelos")
output = model.generate("Qual a capital da frança? ")

print(output)