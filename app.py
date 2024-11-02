import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader

#streamlit framework
st.set_page_config(page_title="Summarize text from YouTube or Website")
st.title("Summarize text from YouTube or Website")
st.subheader("Paste the URL")



#adding groq and url to summarize
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key",value="", type = "password")

llm = ChatGroq(model= "Gemma-7b-It", groq_api_key = groq_api_key)

prompt_template = """ 
Provide a summary of the following content in 300 words,
Content : {text}
"""

prompt = PromptTemplate(
    template= prompt_template,
    input_variables=["text"]
)

url = st.text_input("URL", label_visibility= "collapsed")

if st.button("Summarize the content"):
    #validate all the inputs
    if not groq_api_key.strip() or not url.strip():
        st.error("Please provide the info")
    elif not validators.url(url):
        st.error("Please enter valid url")
    else:
        try:
            with st.spinner("Waiting..."):
                #loading the website or YT data
                if "youtube.com" in url:
                    loader = YoutubeLoader.from_youtube_url(url, add_video_info=False)
                else:
                    loader = UnstructuredURLLoader(urls = [url], ssl_verify = False, 
                                                   headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"})
                docs = loader.load()

                #chain for summarization
                chain = load_summarize_chain(llm, chain_type="stuff", prompt = prompt)
                summary = chain.run(docs)

                st.success(summary)
        except Exception as e:
            st.exception(f"Exception:{e}")
