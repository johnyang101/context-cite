import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from context_cite.cc_groq import GroqContextCiter
from pinecone import Pinecone
from openai import OpenAI
from groq import Groq
import cohere
import torch as ch
from typing import List
from difflib import get_close_matches
from nltk.tokenize import sent_tokenize
from st_click_detector import click_detector

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
COHERE_API_KEY = st.secrets["COHERE_API_KEY"]

openai_client =  OpenAI(
    api_key=OPENAI_API_KEY,
)

groq_client = Groq(api_key=GROQ_API_KEY)

cohere_client = cohere.Client(api_key=COHERE_API_KEY)
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
# Set the title of the Streamlit app
st.title("Hypophosphatasia Q&A with Citations")

# Set the subtitle of the Streamlit app
st.subheader("To cite AI-generated sentences, just click!")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def _get_embedding(text) -> List[float]:
    if isinstance(text, str):
        text = [text]
    embedding_response = openai_client.embeddings.create(input=text,
                                                         model="text-embedding-3-small",
                                                         dimensions=256) #hardcoded for our index
    embeddings = ch.stack([ch.tensor(item.embedding) for item in embedding_response.data])
    if embeddings.dim() == 1:
        return embeddings.unsqueeze(0)
    return embeddings.squeeze().tolist()

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
# Connect to the Pinecone index
index = pc.Index("stanfordrdhack") #index generated from other hackathon project for simplicity


def get_context(query) -> str:
    query_embedding = _get_embedding(query)
    # Perform a similarity search in the Pinecone index
    search_results = index.query(
        vector=query_embedding,
        namespace='hpp',
        include_values=True,
        include_metadata=True,
        top_k=5
    )

    # Extract the most relevant contexts
    relevant_contexts = [match['metadata']['text'] for match in search_results['matches']]

    # Combine the relevant contexts
    combined_context = " ".join(relevant_contexts)
    combined_context += "Hypophosphatasia is a rare, inherited metabolic disorder that affects the development of bones and teeth. It is caused by mutations in the ALPL gene, which encodes an enzyme called alkaline phosphatase. People with hypophosphatasia have low levels of alkaline phosphatase, which leads to abnormal mineralization of bones and teeth. The severity of the condition can vary widely, from mild forms that only affect the teeth to severe forms that can be life-threatening. Treatment for hypophosphatasia is focused on managing symptoms and preventing complications. This may include medications to increase alkaline phosphatase levels, physical therapy, and surgery to correct bone deformities."
    return combined_context

def run_rag(query: str) -> GroqContextCiter:
    # Define the context
    context = get_context(query)

    # Initialize the GroqContextCiter
    return GroqContextCiter(
        groq_model='llama3-70b-8192',
        context=context,
        query=query,
        groq_client=groq_client,
        openai_client=openai_client,
        cohere_client=cohere_client,
        num_ablations=8
    )


if "response_links" not in st.session_state:
    st.session_state.response_links = False

if "last_message" not in st.session_state:
    st.session_state.last_message = None

# Accept user input
if prompt := st.chat_input("Ask a question about hypophosphatasia:"):

    if st.session_state.last_message:
        with st.chat_message("assistant"):
            st.markdown(st.session_state.last_message)
        st.session_state.messages.append({"role": "assistant", "content": st.session_state.last_message})

    # Add user message to chat history
    user_query_message = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_query_message)
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    cc = run_rag(prompt)
    st.session_state.cc = cc
    response = cc.response
    
    def _html(response):
        sentences = sent_tokenize(response)
        content = ""
        for i, sentence in enumerate(sentences):
            content += f"<a href='#' id='{sentence}' style='color: black; text-decoration: none;' onmouseover='this.style.textDecoration=\"underline\"' onmouseout='this.style.textDecoration=\"none\"'>{sentence} </a>" #NOTE: the space at the end is intentional 
        return content

    response_links = _html(response)

    st.session_state.response_links = response_links
    #TODO: Need good state variable tracking whether it's assistant or not
    # st.session_state.sentences = sentences

    cc.messages.append(user_query_message)
    # Add assistant response to chat history
    assistant_message = {"role": "assistant", "content": response}
    # st.session_state.messages.append(assistant_message)
    cc.messages.append(assistant_message)

if st.session_state.response_links:
    with st.chat_message("assistant"):
        clicked = click_detector(st.session_state.response_links, key=st.session_state.messages[-1]["content"])
        mark_citation = f"**Citing** *{clicked}*" if clicked != "" else "**No citation generated**"
        st.write(mark_citation)
        last_message = mark_citation
    
    if clicked != "":
        # st.session_state.messages[-1]["content"] += f"\n\nCiting {clicked}"
        cc = st.session_state.cc
        clicked_sentence = clicked #.split(": ", 1)[1]
        attr_df = cc.get_rerank_df(clicked_sentence, top_k=5) #TODO: Use start & end idx in context_cite
        attr_df = attr_df[attr_df['Score'] != 0]
        with st.chat_message("assistant"):
            ranked_list_message = 'Here is a ranked list of relevant sentences with citations.'
            st.markdown(ranked_list_message)
            st.write(attr_df)
    else:
        last_message = f"**Cited** *{clicked_sentence}*"
        if "last_message" not in st.session_state:
            st.session_state["last_message"] = last_message
        else:
            if clicked != st.session_state["last_message"]:
                st.session_state["last_message"] = last_message
    
# Add this at the end of your file
st.markdown("""
**Disclaimer:** The information provided by this application is generated by AI and may not always be accurate. The citations provided are not necessarily verified, and we assume no liability for any incorrect or misleading information.
""")