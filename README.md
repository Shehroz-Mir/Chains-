# LangChain Chain Patterns

A collection of practical examples demonstrating different LangChain chain patterns for building LLM-powered applications. This repository showcases three fundamental chain architectures: Sequential, Conditional, and Parallel chains.

## Overview

This project provides hands-on examples of how to implement various chain patterns using LangChain with both OpenAI and Anthropic models. Each example demonstrates a specific pattern that can be applied to real-world use cases.

## Chain Patterns

### 1. Sequential Chain (`simple_chain.py`)
Demonstrates a simple sequential chain where outputs from one step feed into the next.

**Use Case**: Report generation and summarization
- Takes a topic as input
- Generates a detailed report using GPT-5
- Summarizes the report into a concise format
- Displays the chain graph structure

**Flow**: `Topic Input → Generate Report → Summarize Report → Output`

### 2. Conditional Chain (`conditional_chain.py`)
Implements branching logic based on classification results using `RunnableBranch`.

**Use Case**: Customer feedback routing system
- Classifies feedback sentiment (positive/negative) using structured output with Pydantic
- Routes to different response templates based on sentiment
- Positive feedback → General acknowledgment
- Negative feedback → Telecom customer support response
- Uses both OpenAI (GPT-5) for classification and Claude (Sonnet 4-5) for response generation

**Flow**: `Feedback → Classify Sentiment → Branch (Positive/Negative) → Generate Response`

### 3. Parallel Chain (`parallel_chain.py`)
Executes multiple chains simultaneously using `RunnableParallel` for improved efficiency.

**Use Case**: Educational content generation
- Processes text input through two parallel chains:
  - **Notes Chain**: Generates concise notes using GPT-5
  - **Q&A Chain**: Creates questions and answers using Claude Sonnet 4-5
- Merges both outputs into a unified study document
- Demonstrates multi-model orchestration

**Flow**:
```
           ┌─→ Generate Notes (GPT-5) ─┐
Text Input ┤                            ├→ Merge → Final Document
           └─→ Generate Q&A (Claude) ───┘
```

## Features

- Multiple LLM providers (OpenAI GPT-5, Anthropic Claude Sonnet 4-5)
- Structured output parsing with Pydantic
- Chain visualization with ASCII graphs
- Environment variable management with dotenv
- File I/O operations for batch processing
- Real-world use case examples

## Prerequisites

- Python 3.8+
- OpenAI API key
- Anthropic API key

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Shehroz-Mir/Chains-.git
cd Chains
```

2. Install required dependencies:
```bash
pip install langchain-openai langchain-anthropic langchain-core python-dotenv pydantic
```

## Setup

1. Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

2. Ensure the `.env` file is listed in `.gitignore` to prevent exposing API keys.

## Usage

### Running Simple Chain
```bash
python simple_chain.py
```
- You'll be prompted to enter a topic
- The script will generate a detailed report and then summarize it
- Outputs the chain graph structure and final summary

### Running Conditional Chain
```bash
python conditional_chain.py
```
- Currently processes a hardcoded feedback example
- Modify line 52 to test with different feedback:
```python
result = chain.invoke({"feedback": "Your feedback text here"})
```

### Running Parallel Chain
```bash
python parallel_chain.py
```
- Processes text from `SLMs.txt` (or embedded text about Small Language Models)
- Generates notes and Q&A in parallel
- Saves the merged output to `SLMs_output.txt`
- Displays the chain graph structure

## Project Structure

```
Chains/
├── simple_chain.py          # Sequential chain example
├── conditional_chain.py     # Branching chain example
├── parallel_chain.py        # Parallel execution example
├── SLMs.txt                 # Sample input text about Small Language Models
├── SLMs_output.txt          # Generated output from parallel chain
├── .env                     # Environment variables (API keys)
├── .gitignore              # Git ignore file
└── README.md               # This file
```

## Key Concepts

### LangChain Components Used

1. **Models**:
   - `ChatOpenAI`: OpenAI's GPT models
   - `ChatAnthropic`: Anthropic's Claude models

2. **Prompts**:
   - `PromptTemplate`: Structured prompt templates with variables

3. **Parsers**:
   - `StrOutputParser`: Basic string output parsing
   - `PydanticOutputParser`: Structured output parsing with validation

4. **Runnables**:
   - `RunnableParallel`: Execute multiple chains simultaneously
   - `RunnableBranch`: Conditional routing based on input
   - `RunnableLambda`: Custom lambda functions in chains

5. **Chain Composition**:
   - Chains are composed using the pipe operator `|`
   - Example: `prompt | model | parser`

## Understanding Chain Graphs

Each script includes `chain.get_graph().draw_ascii()` which visualizes the chain structure. This helps understand the data flow and dependencies in your chain.

## Example Outputs

### Simple Chain
- Input: "Artificial Intelligence"
- Output: A comprehensive report on AI followed by a concise summary

### Conditional Chain
- Input: "The product was disgusting and arrived late."
- Output: A professional customer support response addressing negative feedback

### Parallel Chain
- Input: Article about Small Language Models
- Output: Combined document with notes and Q&A for study purposes

## Customization

You can easily modify these examples for your own use cases:

1. **Change Models**: Swap `gpt-5` with `gpt-4o` or other available models
2. **Adjust Prompts**: Modify templates in `PromptTemplate` to fit your needs
3. **Add More Branches**: Extend conditional chains with additional sentiment categories
4. **Parallel Chains**: Add more parallel operations for richer outputs
5. **Temperature Settings**: Adjust `temperature` parameter for creativity vs. consistency

## Contributing

Feel free to fork this repository and experiment with different chain patterns. Some ideas:
- Add error handling and retry logic
- Implement streaming responses
- Add more complex conditional logic
- Create hybrid chains combining multiple patterns
- Add unit tests for chain components

## License

This project is for educational purposes.

## Resources

- [LangChain Documentation](https://python.langchain.com/)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Anthropic API Documentation](https://docs.anthropic.com/)
