import os
import json
from langchain_openai import ChatOpenAI

TEMPLATES_ROOT_PATH = os.environ["TEMPLATES_ROOT_PATH"]

def keyword_extraction(question: str, model: str = "gpt-4o", temperature: float = 0.0):
    """
    Extracts keywords from the question.
    """
    template_name = "keyword_extraction"
    file_name = f"template_{template_name}.txt"
    template_path = os.path.join(TEMPLATES_ROOT_PATH, file_name)
    KEYWORD_EXTRACTION_PROMPT = ""
    with open(template_path, "r") as file:
        KEYWORD_EXTRACTION_PROMPT = file.read()

    task = (
        f"Question: {question}"
        # f"Hint: {hint}"
    )
    
    prompt = KEYWORD_EXTRACTION_PROMPT.format(TASK=task)
    llm = ChatOpenAI(model=model, temperature=temperature)
    response = llm.invoke(prompt)
    response = response.content
    
    try:
        keywords = eval(response)
    except:
        print(f"Failed to extract keywords: {response}")
        keywords = []
    return keywords

if __name__ == "__main__":

    example = {
        "question_id": 9,
        "db_id": "california_schools",
        "question": "Among the schools with the average score in Math over 560 in the SAT test, how many schools are directly charter-funded?",
        "evidence": "",
        "SQL": "SELECT COUNT(T2.`School Code`) FROM satscores AS T1 INNER JOIN frpm AS T2 ON T1.cds = T2.CDSCode WHERE T1.AvgScrMath > 560 AND T2.`Charter Funding Type` = 'Directly funded'",
        "difficulty": "simple"
    }
    question = example["question"]
    keywords = keyword_extraction(question=question)
    print(keywords)