from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
import os
from dotenv import load_dotenv

load_dotenv()

model1 = ChatOpenAI(model='gpt-5')

model2 = ChatAnthropic(model='claude-sonnet-4-5')

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="Generate short and simple notes on the following text:\n {text}",
    input_variables=["text"],
)

prompt2 = PromptTemplate(
    template="Generate 5 short questions and answers based on the following text:\n {text}",
    input_variables=["text"],
)

prompt3 = PromptTemplate(
    template="Merge the provided notes and question answers into a single document:\n Notes: {notes}\n Q&A: {quiz}",
    input_variables=["notes", "quiz"],
)

parser = StrOutputParser()

parallel_chain = RunnableParallel(
    {'notes': prompt1 | model1 | parser, 
     'quiz': prompt2 | model2 | parser}
)

merged_chain = prompt3 | model1 | parser 

chain = parallel_chain | merged_chain

# with open('SLMs.txt', 'r', encoding="utf-8") as file:
#     text = file.read()

text = """"
Small language models (SLMs) are compact, efficient, and don’t need massive servers—unlike their large language models (LLMs) counterparts. They’re built for speed and real-time performance and can run on our smartphones, tablets, or smartwatches.

In this article, we’ll examine the top 15 SLMs of 2025 and explore their strengths, weaknesses, and what makes each model unique.

SLMs timeline

Source: Lu et al., 2024

I’ll jump straight to discussing the models, but if you need a primer on small language models, I wrote a separate article here: Small Language Models: A Guide With Examples.

1. Qwen2: 0.5B, 1B, and 7B
Qwen2 it’s a family of models, with sizes that go from 0.5 billion to 7 billion parameters. If you’re working on an app that needs a super lightweight model, the 0.5B version is perfect.

However, if you need something more robust for tasks like summarization or text generation, the 7B model is where you’ll get the most performance. It’s scalable and can be tailored to your specific needs. 

Qwen2 models may not match the broad abilities of huge AI models in complex thinking, but they're great for many practical uses where speed and efficiency matter most. They're particularly useful for apps requiring quick responses or limited resources.

Parameters: 0.5 billion, 1 billion, and 7 billion versions
Access: https://huggingface.co/Qwen
Open source: Yes, with an open-source license
2. Mistral Nemo 12B
With 12 billion parameters, Mistral Nemo 12B model is great for complex NLP tasks like language translation and real-time dialogue systems. It competes with models like Falcon 40B and Chinchilla 70B, but it can still run locally without a massive infrastructure setup. It’s one of those models that balances complexity with practicality.

Parameters: 12 billion
Access: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2
Open source: Yes, with an Apache 2.0 license
Develop AI Applications
Learn to build AI applications using the OpenAI API.
3. Llama 3.1 8B
Moving on to Llama 3.1 8B, this model has 8 billion parameters, and it provides an amazing balance between power and efficiency. It’s great for tasks like question answering and sentiment analysis.

Llama 3.1 8B provides reasonably good performance if you need fast results without huge computing power. It is perfect for those who want speed without sacrificing accuracy.

To get hands-on experience with this model, read this tutorial on RAG with Llama 3.1 8B, Ollama, and Langchain.

Parameters: 8 billion
Access: https://ollama.com/library/llama3
Open source: Yes, but with usage restrictions
4. Pythia
Let’s talk about the Pythia series, a set of models ranging from 160 million to 2.8 billion parameters, designed for reasoning and coding skills tasks. If you’re into software development, Pythia is great for handling structured, logic-based tasks where accuracy and logic are key. It’s perfect for coding environments where you need the model to think in a structured, logical way.

Now, compared to other models like GPT-Neo, Pythia performs better at tasks like coding and reasoning as it is built for these focused applications. However, when you throw it into more general language tasks, things can get a little shaky - Phi 3.5 and Llama 3.1 8B might perform more consistently in those broader areas. One thing to note is that Pythia public training transparency and customization options are quite impressive. You can tweak it to fit you specific needs, which makes it an incredibly flexible tool.

Parameters: 160M – 2.8B
Access: https://github.com/EleutherAI/pythia
Open Source: Yes
5. Cerebras-GPT
Cerebras-GPT is a model that’s efficient and quick. With parameters ranging from 111 million to 2.7 billion, it’s designed for environments where computational resources are limited, but you still need great performance. Cerebras-GPT brings great results without eating up all your resources.

Now, compared to larger models like GPT-3 or LLaMA 13B, Cerebras-GPT might not have the same extensive training, but it follows Chinchilla scaling laws, which means it’s incredibly compute-efficient. Models like GPT-J and GPT-NeoX might be bulkier, but Cerebras-GPT maximizes performance while keeping resource usage low. If you need scalability and efficiency, this model is optimized to give you the best of both worlds.

Parameters: 111M – 2.7B
Access: https://github.com/Cerebras
Open source: Yes
6. Phi-3.5
This model has 3.8 billion parameters, but here’s what makes it unique: 128K tokens of context length. What does that mean? It can handle long documents or tasks that involve multi-turn conversations without losing context. It’s multilingual too, which makes it a strong competitor against models like Llama 13B and GPT-3.5, but with much lower computational demands. This model is great for document summarization, multilingual tasks, and logical reasoning.

Parameters: 3.8 billion
Access: https://huggingface.co/microsoft/phi-2
Open source: Yes, for research purposes only.
7. StableLM-zephyr
StableLM-Zephyr is a small language model with 3 billion parameters that is great when you want accuracy and speed. This model provides a fast inference and performs incredibly well in environments where quick decision-making is key, like edge systems or low-resource devices. If you need something that’s sharp and fast, StableLM-Zephyr is a great option.

StableLM-Zephyr excels in tasks that involve reasoning and even role-playing. While it’s lighter and faster, it may not handle more complex tasks like writing or coding as well as the larger models, but for its size, it’s a great performer. If speed and efficiency are your priorities, StableLM-Zephyr is a solid choice.

Parameters: 3B
Access: https://github.com/StabilityAI/stablelm
Open source: Yes
8. TinyLlama
Let’s talk about TinyLlama, a compact model with 1.1 billion parameters that performs really well for its size. It’s designed for efficiency, and it is perfect for devices that can’t handle the heavy computational load of larger models.

For real-world tasks, TinyLlama actually does better than models like Pythia-1.4B, especially for commonsense reasoning. It doesn’t have the raw power of models like LLaMA 13B, but it has a great balance between performance and resource efficiency. That makes it ideal for scenarios where you need strong AI capabilities without overloading the system, especially on mobile and edge devices.

Parameters: 1.1B
Access: https://github.com/tinyLlama
Open source: Yes
9. MobileLLaMA
MobileLLaMA is a specialized version of LLaMA built to perform really well on mobile and low-power devices. With 1.4 billion parameters, it’s designed to give you a balance between performance and efficiency, especially on devices with limited resources. 

MobileLLaMA is optimized for speed and low-latency AI applications on the go. With versions like MobileLLaMA-1.4B and MobileLLaMA-2.7B, it easily outpaces smaller models like TinyLLaMA 1.1B and competes closely with OpenLLaMA 3B - all while being about 40% faster. If you need real-time AI right on your device, MobileLLaMA is perfect. This model is built to bring high-performance AI straight to your mobile or edge systems without the need for heavy infrastructure.

Parameters: 1.4B
Access: https://github.com/mobileLLaMA
Open source: Yes
10. LaMini-GPT
LaMini-GPT is a compact yet powerful model ranging from 774 million to 1.5 billion parameters that’s been specifically designed for multilingual tasks. It’s particularly strong in resource-constrained environments, meaning it can handle multiple languages without needing a lot of computational power, which is a great fit for devices or systems with limited resources.

Something interesting about LaMini-GPT is that it was developed through knowledge distillation from larger models from the GPT family, which allows it to perform really well in instruction-following tasks. With over 2.58 million instruction-response pairs in its dataset, it’s optimized for handling specific tasks and instructions more efficiently than larger models. However, while it’s incredibly efficient and lightweight, especially for focused tasks, it is not that great for broader applications that require deep contextual understanding or more general text generation. If you’re looking for something fast and efficient, particularly in multilingual scenarios, LaMini-GPT is a solid choice.

Parameters: 774M – 1.5B
Access: https://github.com/LaMiniGPT
Open source: Yes
"""

result  = chain.invoke({'text': text})

print(result)

with open("SLMs_output.txt", "w", encoding="utf-8") as f:
    f.write(result)

print(chain.get_graph().draw_ascii())