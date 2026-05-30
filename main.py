from data import load_data, load_rag_data, load_model
from rag import populate_vector_store, process_query
def main():
    populate_vector_store()
    query = input("Ask your medical related question: ")
    process_query(query)

if __name__ == "__main__":
    main()