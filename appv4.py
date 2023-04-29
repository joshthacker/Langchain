from gpt_index import SimpleDirectoryReader, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import gradio 
from gradio import inputs, components
from pathlib import Path
import sys
import os

os.environ["OPENAI_API_KEY"] = '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'

# first initialize the large language model
llm = OpenAI(
    temperature=0,
    openai_api_key="OPENAI_API_KEY",
    model_name="text-davinci-003"
)

template = """You are a chatbot having a conversation with a human.

{chat_history}
Human: {input}
Chatbot:"""

prompt = PromptTemplate(
    input_variables=["chat_history", "input"], 
    template=template
)


memory = ConversationBufferMemory(memory_key="chat_history")


# now initialize the conversation chain
conversation = ConversationChain(llm=llm, memory=memory, prompt=prompt)

def construct_index(directory_path):
    max_input_size = 4096
    num_outputs = 512
    max_chunk_overlap = 20
    chunk_size_limit = 600

    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.7, model_name="gpt-3.5-turbo", max_tokens=num_outputs))

    documents = SimpleDirectoryReader(directory_path).load_data()

    index = GPTSimpleVectorIndex.from_documents(documents)

    index.save_to_disk('index.json')

    return index

def chatbot(input_text):
    index = GPTSimpleVectorIndex.load_from_disk('index.json')
    prompt_kwargs = {"chat_history": conversation.memory.memory, "human_input": input_text}
    response = conversation.llm.generate(prompt=prompt, **prompt_kwargs)
    conversation.memory.append_to_memory(response.prompt)
    return response.generated_text


iface = gradio.Interface(fn=chatbot,
                     inputs=inputs.Textbox(lines=7, label="Enter your text"),
                     outputs="text",
                     title="Custom-trained AI Chatbot")

docs_folder = os.path.join(Path.home(), "digital_childrens_book")
index = construct_index(docs_folder)

iface.launch(share=True)


