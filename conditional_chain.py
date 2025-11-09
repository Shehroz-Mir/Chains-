from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model='gpt-5', temperature=0.5)

model2 = ChatAnthropic(model='claude-sonnet-4-5')

parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(
        ..., description="The sentiment of the feedback"
    )

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template = "Classify the sentiment of the following feedback as positive, or negative:\n\n{feedback}\n {format_instructions}",
    input_variables = ["feedback"],
    partial_variables={'format_instructions': parser2.get_format_instructions()},
)

classifier_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(
    template='Write an appropriate response to the following postive feedback:\n\n{feedback}',
    input_variables=['feedback'],
)

prompt3 = PromptTemplate(
    template='You are a telecom customer support assistant.\n Write an appropriate response to the following negative feedback:\n\n{feedback}',
    input_variables=['feedback'],
)

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == "positive", prompt2 | model2 | parser),
    (lambda x: x.sentiment == "negative", prompt3 | model2 | parser),
    RunnableLambda(lambda x: "No matching branch found.")
)

chain = classifier_chain | branch_chain

result = chain.invoke({"feedback": "The product was disgusting and arrived late."})

print(result)