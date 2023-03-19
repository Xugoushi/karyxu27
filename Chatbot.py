from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, VectorDBQA
from langchain.document_loaders import DirectoryLoader
import bs4
import os
import nltk
import gradio as gr

# 将指定目录下的所有txt文件加载到文件列表中
loader = DirectoryLoader('../data/PaulGrahamEssaySmall/', glob='**/*.txt')
documents = loader.load()

# 将每个文档划分成较小的文本块，有助于更好的处理和搜索
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# 使用OpenAI预训练模型生成文本嵌入，这里使用的是OpenAIEmbeddings
embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])

# 从上一部生成的文本嵌入中创建Chroma对象
docsearch = Chroma.from_documents(texts, embeddings)


# 使用 VectorDBQA.from_chain_type 创建一个问答对象，
# 其中使用了 OpenAI 的语言模型作为计算问题和答案的核心组件，
# 并使用上面创建的 Chroma 对象作为文本检索引擎。
qa = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=docsearch)



def answer_question(question):
    answer = qa.run(question)
    return answer

# 使用 gr.Interface 创建用户界面，使用户可以输入问题并查看答案。
iface = gr.Interface(fn=answer_question, 
                     inputs="text", 
                     outputs="text",
                     title="问答机器人",
                     description="输入您的问题，获取答案。",
                     server_name="localhost",
                     server_port=7860,
                     share=False)
                     
iface.launch()