import streamlit as st
import os
from bosch_vta_agentic.utils.schema import RAGPipeline


def main():
    st.title("Bosch VTA Agentic RAG Chatbot")

    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None
        st.session_state.messages = []

    if st.session_state.pipeline is None:
        data_folder = "./data"
        index_folder = "./index"

        if os.path.exists(index_folder):
            st.session_state.pipeline = RAGPipeline.load_index(index_folder)
            st.success("Loaded existing index.")
        else:
            st.session_state.pipeline = RAGPipeline(data_folder)
            st.session_state.pipeline.load_and_index_documents(index_folder)
            st.session_state.pipeline.save_index(index_folder)
            st.success("Created and saved new index.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What would you like to know?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            result = st.session_state.pipeline.query(prompt)
            st.markdown(result.answer)
            with st.expander("Sources"):
                for source in result.source_nodes:
                    st.markdown(source)

        st.session_state.messages.append(
            {"role": "assistant", "content": result.answer}
        )


if __name__ == "__main__":
    main()
