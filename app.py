from gpt_index import SimpleDirectoryReader, GPTListIndex, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain import OpenAI
import gradio 
from gradio import inputs, components
from pathlib import Path
import sys
import os
from langchain.chains import ConversationChain

os.environ["OPENAI_API_KEY"] = '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'

def construct_index(directory_path):
    max_input_size = 4096
    num_outputs = 512
    max_chunk_overlap = 20
    chunk_size_limit = 600

    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.7, model_name="gpt-4", max_tokens=num_outputs))

    documents = SimpleDirectoryReader(directory_path).load_data()

    index = GPTSimpleVectorIndex.from_documents(documents)

    index.save_to_disk('index.json')

    return index

# Initialize llm predictor
llm = OpenAI(temperature=0.7, model_name="gpt-4", max_tokens=512)

# Initialize conversation chain
conversation = ConversationChain(llm=llm)

def chatbot(input_text):
    index = GPTSimpleVectorIndex.load_from_disk('index.json')
    response = index.query(input_text, response_mode="compact")
    return response.response

iface = gradio.Interface(fn=chatbot,
                     inputs=inputs.Textbox(lines=7, label="Enter your text"),
                     outputs="text",
                     title="Custom-trained AI Chatbot")

docs_folder = os.path.join(Path.home(), "storybookai")
index = construct_index(docs_folder)

iface.launch(share=True)