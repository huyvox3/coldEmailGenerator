from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

load_dotenv()

class Chain:
        def __init__(self):
            self.llm = ChatGroq(temperature=0, groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.1-70b-versatile")
        
        def extract_jobs(self, cleaned_text):
            prompt_extract = PromptTemplate.from_template(
                """
                ### SCRAPED TEXT FROM WEBSITE:
                {page_data}
                ### INSTRUCTION:
                The scraped text is from the career's page of a website.
                Your job is to extract the job postings and return them in JSON format
                  containing the following keys:
                  `role`, `experience`, `skills` and `description`.
                Only return the valid JSON.
                ### VALID JSON (NO PREAMBLE):
                """
            )
            chain_extract = prompt_extract | self.llm
            res = chain_extract.invoke(input={"page_data": cleaned_text})
            try:
                json_parser = JsonOutputParser()
                res = json_parser.parse(res.content)
            except OutputParserException:
                raise OutputParserException("Context too big. Unable to parse jobs.")
            return res if isinstance(res, list) else [res]

        def write_mail(self, job,link):
            prompt_email = PromptTemplate.from_template(
                    """
                    The subjetct for the email should be Application for the job
                    ### JOB DESCRIPTION:
                    {job_description}
                    ### INSTRUCTION:
                    You are Huy, a Software Engineer, currently studying at VKU( Vietname-Korea University). 
                    Over our experience, we have made numerous projects. Your job is to write a 
                    cold email to the client regarding the job mentioned above describing capability 
                    of VKU in making good students.
                    Also add the most relevant ones from the following links to showcase Huy's portfolio: {link_list}
                        Remember you are Huy, SE and currently studying at VKU. 
                        Do not provide a preamble.
                        ### EMAIL (NO PREAMBLE):
                        
                    """
                )

            chain_email = prompt_email |self.llm
            res = chain_email.invoke(input={"job_description": job, "link_list": link})
            return res.content

       

if __name__ == "__main__":
    print(os.getenv("GROQ_API_KEY"))