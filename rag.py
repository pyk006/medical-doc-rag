import os
from data import load_rag_data

## Langchain imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from transformers import AutoModelForCausalLM, AutoTokenizer

def populate_vector_store():
    if os.path.exists("FAISS-index"):
        return
    dataset = load_rag_data()
    all_documents = []
    ## make all context strings into documents
    for item in dataset['train']:
        context_string = " ".join(item['context']['contexts'])
        all_documents.append(Document(context_string))
    
    ## pass into FAISS
    embeddings = HuggingFaceEmbeddings(model_name="pritamdeka/S-PubMedBert-MS-MARCO")
    vector_store = FAISS.from_documents(all_documents, embeddings)

    vector_store.save_local("FAISS-index")
    print("FAISS data saved successfully!")
def process_query(query):
    embeddings = HuggingFaceEmbeddings(model_name="pritamdeka/S-PubMedBert-MS-MARCO")
    vector_store = FAISS.load_local(folder_path="FAISS-index", embeddings=embeddings, allow_dangerous_deserialization=True)
    ## Vector similarity search for now (possibly will update to a hybrid search (e.g vector + keyword) if performance isn't great)
    documents = vector_store.similarity_search(query, k=3)
    
    ## load in best model from training and generate the response
    model = AutoModelForCausalLM.from_pretrained("./model", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained("./model", device_map="auto")

    context = " ".join(document.page_content for document in documents)
    prompt = f"Context: {context} Query: {query} Answer: "

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=200)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(answer)