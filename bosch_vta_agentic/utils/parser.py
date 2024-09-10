import os
from llama_parse import LlamaParse
from dotenv import load_dotenv

load_dotenv()

pdf_file_path = "../../Automobile Mechanical and Electrical Systems.pdf"
markdown_file_path = "AutomobileMechanicalandElectricalSystems.md"

# Initialize the LlamaParse object
parser = LlamaParse(api_key=os.getenv("LLAMA_CLOUD_API_KEY"), result_type="markdown")

documents = parser.load_data(pdf_file_path)

md_text = ""

for doc in documents:
    md_text += doc.text + "\n\n"

with open(markdown_file_path, "w", encoding="utf-8") as file:
    file.write(md_text)

print(f"Parsed content has been saved to {markdown_file_path}")
