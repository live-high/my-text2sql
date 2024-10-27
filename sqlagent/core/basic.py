from sqlagent.preprocessing.entity_retrieval import entity_retrieval
from sqlagent.preprocessing.keyword_extraction import keyword_extraction
from sqlagent.runner.database_manager import DatabaseManager
from sqlagent.database_utils.db_info import get_db_all_tables, get_table_all_columns, get_db_schema
from sqlagent.database_utils.execution import execute_sql
import os
from langchain_openai import ChatOpenAI
TEMPLATES_ROOT_PATH = os.environ["TEMPLATES_ROOT_PATH"]

def candidate_generation(question: str, schema_string: str, model:str="gpt-4o", temperature: float = 0.0):
    
    template_name = "candidate_generation"
    file_name = f"template_{template_name}.txt"
    template_path = os.path.join(TEMPLATES_ROOT_PATH, file_name)
    CANDIDATE_GENERATION_PROMPT = ""
    with open(template_path, "r") as file:
        CANDIDATE_GENERATION_PROMPT = file.read()

    task = (
        f"Question: {question}"
        # f"Hint:\n{hint}"
    )
    
    prompt = CANDIDATE_GENERATION_PROMPT.format(DATABASE_SCHEMA=schema_string, TASK=task)
    
    llm = ChatOpenAI(model=model, temperature=temperature)
    response = llm.invoke(prompt)
    response = response.content
    return response


def basic_pipeline(question: str, db_mode: str, db_id: str, model:str="gpt-4o", temperature: float = 0.0):
    keywords = keyword_extraction(question=question, model=model, temperature=temperature)
    retrieval_result = entity_retrieval(question=question, keywords=keywords, db_mode=db_mode, db_id=db_id)
    print(keywords, retrieval_result)
    
    schema_with_examples = retrieval_result["similar_values"]
    db_manager = DatabaseManager(db_mode=db_mode, db_id=db_id)
    schema_string = db_manager.get_database_schema_string(
        tentative_schema=get_db_schema(db_manager.db_path), 
        schema_with_examples=schema_with_examples, 
        schema_with_descriptions=None, 
        include_value_description=True
    )
    response = candidate_generation(question=question, schema_string=schema_string, model=model, temperature=temperature)
    result = {
        "response": response,
        "keywords": keywords,
        "retrieval_result": retrieval_result,
        "schema_string": schema_string,
    }
    return result



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
    db_id = example["db_id"]
    DB_MODE = "dev"
    DB_ID = db_id
    # keywords = ["Charter Funding Type", "FundingType", "directly charter-funded"]
    # keywords = ["Charter Funding Type"]
    # DB_ROOT_PATH = f"/Users/yanghuanye/clientstores/CHESS/data/"
    # retrieval_result = {'similar_columns': {'schools': ['FundingType']}, 'similar_values': {'frpm': {'Charter Funding Type': ['Directly funded']}, 'schools': {'FundingType': ['Directly funded']}}}
    
    keywords = ['schools', 'average score', 'Math', '560', 'SAT test', 'charter-funded', 'directly charter-funded'] 
    retrieval_result = {
        'similar_columns': {
            'schools': ['School']
        }, 
        'similar_values': {
            'schools': {'School': ['MethodSchools'], 'FundingType': ['Directly funded']}, 
            'frpm': {'School Name': ['MethodSchools'], 'Charter Funding Type': ['Directly funded']}
        }
    }
    
    # model = "claude-3-5-sonnet-20240620"
    model = "claude-3-5-sonnet-20241022"
    temperature = 1.0
    # basic_pipeline(question=question, db_mode=DB_MODE, db_id=DB_ID, model=model, temperature=temperature)
    
    sql = """SELECT COUNT(*) 
FROM satscores T1 
JOIN schools T2 ON T1.cds = T2.CDSCode 
WHERE T1.AvgScrMath > 560 
AND T1.AvgScrMath IS NOT NULL 
AND T2.FundingType = 'Directly funded'"""

    sql = example["SQL"]
    db_manager = DatabaseManager(db_mode=DB_MODE, db_id=DB_ID)
    result = execute_sql(db_manager.db_path, sql)


    print(sql, result)