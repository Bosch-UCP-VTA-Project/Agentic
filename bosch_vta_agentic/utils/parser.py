import os
from llama_parse import LlamaParse

FILE_PATH = "Automobile Mechanical and Electrical Systems.pdf"  # Replace this with the file path you want to parse

parser = LlamaParse(api_key=os.getenv("LLAMA_CLOUD_API_KEY"), result_type="markdown")

documents = parser.load_data(FILE_PATH)
