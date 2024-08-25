
import os
os.environ['OPENAI_API_KEY'] = "sk-..."
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import TextSplitter
from langchain_community.vectorstores import FAISS

# Define a class inheriting from TextSplitter to split text by lines

class LineTextSplitter(TextSplitter):
    def split_text(self, text):
        return text.split('\n')

# Initialize LineTextSplitter object
text_splitter = LineTextSplitter()
# Read data from reviews.txt file

with open('data\reviews.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Split the text into segments by lines
documents = text_splitter.split_text(text)

db = FAISS.from_texts(documents, OpenAIEmbeddings(model="text-embedding-3-large"))
db.save_local("openai_index\reviews_index")