!pip install chromadb
!pip install langchain_community
!pip install langchain_groq
from langchain_groq import ChatGroq
llm = ChatGroq(
    temperature=0,
    groq_api_key='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',
    model_name="llama-3.1-8b-instant"
)
response = llm.invoke("The first person to land on moon was ...")
print(response.content)
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://jobs.nike.com/job/R-51642")
page_data = loader.load().pop().page_content
print(page_data)

from langchain_core.prompts import PromptTemplate

prompt_extract = PromptTemplate.from_template(
        """
        ### SCRAPED TEXT FROM WEBSITE:
        {page_data}
        ### INSTRUCTION:
        The scraped text is from the career's page of a website.
        Your job is to extract the job postings and return them in JSON format containing the
        following keys: role, experience, skills and description.
        Only return the valid JSON.
        ### VALID JSON (NO PREAMBLE):
        """
)

chain_extract = prompt_extract | llm
res = chain_extract.invoke(input={'page_data':page_data})
type(res.content)

from langchain_core.output_parsers import JsonOutputParser

json_parser = JsonOutputParser()
json_res = json_parser.parse(res.content)
json_res

job = json_res[0]
job['skills']

import pandas as pd

df = pd.read_csv("/content/my_portfolio.csv")
df

import uuid
import chromadb

client = chromadb.PersistentClient('vectorstore')
collection = client.get_or_create_collection(name="portfolio")

if not collection.count():
    for _, row in df.iterrows():
        collection.add(documents=row["Techstack"],
                       metadatas={"links": row["Links"]},
                       ids=[str(uuid.uuid4())])
links = collection.query(query_texts=job['skills'], n_results=2).get('metadatas', [])
links

prompt_email = PromptTemplate.from_template(
        """
        ### JOB DESCRIPTION:
        {job_description}

        ### INSTRUCTION:
        You are Job Scraping bot, a business development executive at Xemonic. Xemonic is an AI & Software Consulting company dedicated to facilitating
        the seamless integration of business processes through automated tools.
        Over our experience, we have empowered numerous enterprises with tailored solutions, fostering scalability,
        process optimization, cost reduction, and heightened overall efficiency.
        Your job is to write a cold email to the client regarding the job mentioned above describing the capability of Xemonic
        in fulfilling their needs.
        Also add the most relevant ones from the following links to showcase Xemonic portfolio: {link_list}
        Remember you are Job Scraping bot, BDE at Xemonic.
        Do not provide a preamble.
        ### EMAIL (NO PREAMBLE):

        """
        )

chain_email = prompt_email | llm
res = chain_email.invoke({"job_description": str(job), "link_list": links})
print(res.content)
