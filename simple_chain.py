from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model='gpt-5')

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="Write a detailed report on the following topic: {topic}",
    input_variables=["topic"],
)

prompt2 = PromptTemplate(
    template="Summarize the following report in a concise manner:\n\n{report}",
    input_variables=["report"],
)

chain = prompt1 | model | parser | prompt2 | model | parser

def get_input():
    topic = input("Enter a topic for the report: ")
    return {'topic': topic}


result = chain.invoke(get_input())

print("Final Summary:\n", result)