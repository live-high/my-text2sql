import os
import regex as re
from itertools import combinations
from collections import defaultdict
from langchain_openai import ChatOpenAI
from sqlagent.database_utils.execution import execute_sql
from sqlagent.database_utils.db_info import get_db_schema
from sqlagent.runner.database_manager import DatabaseManager
from sqlagent.utils import is_valid_exec_result



TEMPLATES_ROOT_PATH = os.environ["TEMPLATES_ROOT_PATH"]
DB_ROOT_PATH = os.environ["DB_ROOT_PATH"]

def select_candidate(candidate_a, candidate_b, question: str, schema_string: str, model:str="gpt-4o", temperature: float = 0.0):
    
    template_name = "candidate_selection"
    file_name = f"template_{template_name}.txt"
    template_path = os.path.join(TEMPLATES_ROOT_PATH, file_name)
    SELECT_CANDIDATE_PROMPT = ""
    with open(template_path, "r") as file:
        SELECT_CANDIDATE_PROMPT = file.read()    
    prompt = SELECT_CANDIDATE_PROMPT.format(
        DATABASE_SCHEMA=schema_string, TASK=question,
        CANDIDATE_A_QUERY=candidate_a["query"],
        CANDIDATE_A_RESULT=candidate_a["result"],
        CANDIDATE_B_QUERY=candidate_b["query"],
        CANDIDATE_B_RESULT=candidate_b["result"]
    )
    llm = ChatOpenAI(model=model, temperature=temperature)
    response = llm.invoke(prompt)
    response = response.content
    if re.findall("A|a", response):
        return 0
    elif re.findall("B|b", response):
        return 1
    raise ValueError(f"Invalid response: {response}")


def selector(sqls, question, schema_string, db_mode, db_id, model="gpt-4o", temperature=0.0):
    candidates = []
    db_manager = DatabaseManager(db_mode=db_mode, db_id=db_id)
    candidates = []
    for sql in sqls:
        exec_result = execute_sql(db_manager.db_path, sql)
        if is_valid_exec_result(exec_result):
            candidates.append({
                "query": sql, 
                "result": exec_result
            })
    if len(candidates) < 2:
        return sqls[0]
    votes = defaultdict(int)
    candidate_ids = [i for i in range(len(candidates))]
    for candidate_a_id, candidate_b_id in combinations(candidate_ids, 2):
        try:
            choice = select_candidate(
                candidate_a=candidates[candidate_a_id], 
                candidate_b=candidates[candidate_b_id], 
                question=question, 
                schema_string=schema_string, 
                model=model, 
                temperature=temperature
            )
            if choice == 0:
                votes[candidate_a_id] += 1
            elif choice == 1:
                votes[candidate_b_id] += 1
        except Exception as e:
            print(f"Error in select_candidate: {e}")
            continue
        
    sorted_votes = sorted(votes.items(), key=lambda x: x[1], reverse=True)
    return sqls[sorted_votes[0][0]]


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
    sqls = [
        example["SQL"],
        "SELECT COUNT(*) AS directly_charter_funded_schools FROM satscores T1 JOIN schools T2 ON T1.cds = T2.CDSCode WHERE T1.AvgScrMath > 560 AND T2.FundingType = 'Directly funded'",
        "SELECT COUNT(*) AS num_schools FROM satscores T1 JOIN frpm T2 ON T1.cds = T2.CDSCode WHERE T1.AvgScrMath > 560 AND T2.`Charter Funding Type` = 'Direct'",
    ]
    db_manager = DatabaseManager(db_mode=DB_MODE, db_id=DB_ID)
    schema_string = get_db_schema(db_manager.db_path)
    ret = selector(sqls, question, schema_string, DB_MODE, DB_ID, model="gpt-4o", temperature=1.0)
    print(ret)
