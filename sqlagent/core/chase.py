import os
import json
import regex as re
from langchain_openai import ChatOpenAI
from sqlagent.runner.database_manager import DatabaseManager
from sqlagent.database_utils.db_info import get_db_schema
from sqlagent.database_utils.execution import execute_sql
from sqlagent.utils import parse_sql, parse_last_sql_code, is_valid_exec_result

from sqlagent.core.basic import preprocessing
# from sqlagent.preprocessing.entity_retrieval import entity_retrieval
# from sqlagent.preprocessing.keyword_extraction import keyword_extraction


TEMPLATES_ROOT_PATH = os.environ["TEMPLATES_ROOT_PATH"]


def divide_and_conquer(task: str, schema_string: str, model:str="gpt-4o", temperature: float = 0.0):
    template_name = "divide_and_conquer"
    file_name = f"template_{template_name}.txt"
    template_path = os.path.join(TEMPLATES_ROOT_PATH, file_name)
    DIVIDE_AND_CONQUER_PROMPT = ""
    with open(template_path, 'r') as file:
        DIVIDE_AND_CONQUER_PROMPT = file.read()

    prompt = DIVIDE_AND_CONQUER_PROMPT.format(DATABASE_SCHEMA=schema_string, TASK=task)
    llm = ChatOpenAI(model=model, temperature=temperature)
    response = llm.invoke(prompt)
    response = response.content
    return response


def divide(task: str, schema_string: str, model:str="gpt-4o", temperature: float = 0.0):
    template_name = "divide"
    file_name = f"template_{template_name}.txt"
    template_path = os.path.join(TEMPLATES_ROOT_PATH, file_name)
    DIVIDE_PROMPT = ""
    with open(template_path, 'r') as file:
        DIVIDE_PROMPT = file.read()

    prompt = DIVIDE_PROMPT.format(DATABASE_SCHEMA=schema_string, TASK=task)
    llm = ChatOpenAI(model=model, temperature=temperature)
    response = llm.invoke(prompt)
    response = response.content
    return response


def conquer(task: str, schema_string: str, divide_trajectories: str, model:str="gpt-4o", temperature: float = 0.0):
    template_name = "conquer"
    file_name = f"template_{template_name}.txt"
    template_path = os.path.join(TEMPLATES_ROOT_PATH, file_name)
    CONQUER_PROMPT = ""
    with open(template_path, 'r') as file:
        CONQUER_PROMPT = file.read()

    prompt = CONQUER_PROMPT.format(DATABASE_SCHEMA=schema_string, TASK=task, DIVIDE_TRAJECTORIES=divide_trajectories)
    llm = ChatOpenAI(model=model, temperature=temperature)
    response = llm.invoke(prompt)
    response = response.content
    return response


def assemble(task: str, schema_string: str, conquer_trajectories: str, model:str="gpt-4o", temperature: float = 0.0):
    template_name = "assemble"
    file_name = f"template_{template_name}.txt"
    template_path = os.path.join(TEMPLATES_ROOT_PATH, file_name)
    ASSEMBLE_PROMPT = ""
    with open(template_path, 'r') as file:
        ASSEMBLE_PROMPT = file.read()

    prompt = ASSEMBLE_PROMPT.format(DATABASE_SCHEMA=schema_string, TASK=task, CONQUER_TRAJECTORIES=conquer_trajectories)
    llm = ChatOpenAI(model=model, temperature=temperature)
    response = llm.invoke(prompt)
    response = response.content
    return response


def query_plan(task: str, schema_string: str, model:str="gpt-4o", temperature: float = 0.0):
    template_name = "query_plan"
    file_name = f"template_{template_name}.txt"
    template_path = os.path.join(TEMPLATES_ROOT_PATH, file_name)
    QUERY_PLAN_PROMPT = ""
    with open(template_path, 'r') as file:
        QUERY_PLAN_PROMPT = file.read()

    prompt = QUERY_PLAN_PROMPT.format(DATABASE_SCHEMA=schema_string, TASK=task)
    llm = ChatOpenAI(model=model, temperature=temperature)
    response = llm.invoke(prompt)
    response = response.content
    return response


def extract_examples(output_str: str):
    examples = re.findall(r'"input": ".*"\n"output": ".*"', output_str)
    return examples
    
    
def get_online_synthetic_examples():

    EXAMPLES = """ 
"input": "Among the countries in the group of Heavily Indebted Poor Countries, how many of them are under the lending category of the International Development Associations? (Hints: group of Heavily Indebted Poor Countries is OtherGroups = 'HIPC'; International Development Associations refers to lendingcategory = 'IDA')"
"output": "SELECT COUNT(CountryCode) FROM Country WHERE LendingCategory = 'IDA' AND OtherGroups = 'HIPC'"
 
"input": "What is the description of the footnote on the series code AG.LND.FRST.K2 in 1990 for Aruba? (Hints: Year = 1990; Aruba is the name of country where ShortName = 'Aruba')"
"output": "SELECT T2.Description FROM Country AS T1 INNER JOIN FootNotes AS T2 ON T1.CountryCode = T2.Countrycode WHERE T1.ShortName = 'Aruba' AND T2.Seriescode = 'AG.LND.FRST.K2' AND T2.Year = 'YR1990'"
"""
    DATABASE_NAME = "world_development_indicators"
    db_mode = "train"
    db_id = DATABASE_NAME

    similar_values = {
        "Country": {"LendingCategory": ["IDA"], "OtherGroups": ["HIPC"], "ShortName": ["Aruba"]},
        "FootNotes": {"Seriescode": ["AG.LND.FRST.K2"], "Year": ["YR1990"]},
    }
    schema_with_examples = similar_values
    db_manager = DatabaseManager(db_mode=db_mode, db_id=db_id)
    schema_string = db_manager.get_database_schema_string(
        tentative_schema=get_db_schema(db_manager.db_path), 
        schema_with_examples=schema_with_examples, 
        schema_with_descriptions=None, 
        include_value_description=True
    )
    DATABASE_SCHEMA = schema_string
    template = """The database ({DATABASE_NAME}) structure is defined by the following table schemas (comments after ’–’ provide additional column descriptions).
{DATABASE_SCHEMA}
**************************
The folloiwing are the examples generated for the above database schemas:
{EXAMPLES}
**************************"""

    prompt = template.format(DATABASE_NAME=DATABASE_NAME, DATABASE_SCHEMA=DATABASE_SCHEMA, EXAMPLES=EXAMPLES)

    return prompt


def online_synthetic_with_schema(task: str, schema_string: str, model:str="gpt-4o", temperature: float = 0.0, k: int = 3):
    template_name = "online_synthetic_with_schema"
    file_name = f"template_{template_name}.txt"
    template_path = os.path.join(TEMPLATES_ROOT_PATH, file_name)
    ONLINE_SYNTHETIC_PROMPT = ""
    with open(template_path, 'r') as file:
        ONLINE_SYNTHETIC_PROMPT = file.read()

    examples = [get_online_synthetic_examples()]
    examples = "\n\n".join(examples)
    prompt = ONLINE_SYNTHETIC_PROMPT.format(DATABASE_SCHEMA=schema_string, TRAIN_EXAMPLES=examples, k=k)
    llm = ChatOpenAI(model=model, temperature=temperature)
    response = llm.invoke(prompt)
    response = response.content
    return response


def online_synthetic_with_sql_feature(task: str, schema_string: str, model:str="gpt-4o", temperature: float = 0.0, k: int = 3,):
    template_name = "online_synthetic_with_sql_feature"
    file_name = f"template_{template_name}.txt"
    template_path = os.path.join(TEMPLATES_ROOT_PATH, file_name)
    ONLINE_SYNTHETIC_PROMPT = ""
    with open(template_path, 'r') as file:
        ONLINE_SYNTHETIC_PROMPT = file.read()

    prompt = ONLINE_SYNTHETIC_PROMPT.format(DATABASE_SCHEMA=schema_string, k=k)
    llm = ChatOpenAI(model=model, temperature=temperature)
    response = llm.invoke(prompt)
    response = response.content
    return response


def candidate_generation_with_examples(question: str, schema_string: str, examples: list, model:str="gpt-4o", temperature: float = 0.0):
    
    template_name = "candidate_generation_with_examples"
    file_name = f"template_{template_name}.txt"
    template_path = os.path.join(TEMPLATES_ROOT_PATH, file_name)
    CANDIDATE_GENERATION_PROMPT = ""
    with open(template_path, "r") as file:
        CANDIDATE_GENERATION_PROMPT = file.read()

    task = (
        f"\"input\": \"{question}\"\n"
        f"\"output\": "
        # f"Hint:\n{hint}"
    )
    examples = "\n\n".join(examples)
    prompt = CANDIDATE_GENERATION_PROMPT.format(DATABASE_SCHEMA=schema_string, TASK=task, EXAMPLES=examples)
    
    llm = ChatOpenAI(model=model, temperature=temperature)
    response = llm.invoke(prompt)
    response = response.content
    return response


def get_query_fix_example():
    query_fix_example_template = """
**************************
Table creation statements
{DATABASE_SCHEMA}
**************************
The original question is:
Question:
{TASK}
The SQL query executed was:
{QUERY}
The execution result:
{RESULT}
**************************
Correct the Query:
```sql
{FIXED_QUERY}
```
"""

    example = {
        "question_id": 107,
        "db_id": "financial",
        "question": "What is the gender of the youngest client who opened account in the lowest average salary branch?",
        "SQL": "SELECT 'T1'.'gender' FROM 'client' AS 'T1' INNER JOIN 'district' AS 'T2' ON 'T1'.'district_id' = 'T2'.'district_id' ORDER BY 'T2'.'A11' ASC, 'T1'.'birth_date' DESC NULLS LAST LIMIT 1",
        # "question": "What is the gender of the oldest client who opened his/her account in the highest average salary branch?",
        "evidence": "Earlier birthdate refers to older age; A11 refers to average salary",
        # "SQL": "SELECT T2.gender FROM district AS T1 INNER JOIN client AS T2 ON T1.district_id = T2.district_id ORDER BY T1.A11 DESC, T2.birth_date ASC LIMIT 1",
        "difficulty": "simple"
    }
    incorrect_sql = "SELECT T3.gender FROM district T1 JOIN client T2 ON T1.district_id = T2.district_id JOIN disp T4 ON T2.client_id = T4.client_id JOIN account T3 ON T4.account_id = T3.account_id WHERE T1.district_id = (SELECT district_id FROM district ORDER BY A11 DESC LIMIT 1) ORDER BY T2.birth_date ASC LIMIT 1;"
    fixed_sql = example["SQL"]
    db_mode = "dev"
    db_id = example["db_id"]
    hint = example["evidence"]

    db_manager = DatabaseManager(db_mode=db_mode, db_id=db_id)
    schema_string = db_manager.get_database_schema_string(
        tentative_schema=get_db_schema(db_manager.db_path), 
        schema_with_examples=schema_with_examples, 
        schema_with_descriptions=None, 
        include_value_description=True
    )
    task = (
        f"Question:\n{question}\n"
        f"Hint:\n{hint}\n"
    )
    exec_result = execute_sql(db_manager.db_path, incorrect_sql)
    
    prompt = query_fix_example_template.format(
        DATABASE_SCHEMA=schema_string,
        TASK=task,
        QUERY=incorrect_sql,
        RESULT=exec_result,
        FIXED_QUERY=fixed_sql,
    )
    return prompt


def query_fix(question: str, trajectory: dict, schema_string: str, model:str="gpt-4o", temperature: float = 0.0):
    
    template_name = "query_fix"
    file_name = f"template_{template_name}.txt"
    template_path = os.path.join(TEMPLATES_ROOT_PATH, file_name)
    QUERY_FIX_PROMPT = ""
    with open(template_path, "r") as file:
        QUERY_FIX_PROMPT = file.read()

    task = (
        f"Question:\n{question}\n"
        # f"Hint:\n{hint}"
    )
    examples = [get_query_fix_example()]
    prompt = QUERY_FIX_PROMPT.format(
        DATABASE_SCHEMA=schema_string,
        TASK=task, 
        QUERY=trajectory["SQL"],
        RESULT=trajectory["Execution Result"],
        EXAMPLES=examples,
        )
    
    llm = ChatOpenAI(model=model, temperature=temperature)
    response = llm.invoke(prompt)
    response = response.content
    return response


def DC_basic_pipeline(question, db_mode, db_id, model:str="gpt-4o", temperature: float = 0.0):

    retrieval_result_file = f"preprocessing_outputs/basic/{model}/retrieval_result.jsonl"
    retrieval_result = {}
    for line in open(retrieval_result_file):
        item = json.loads(line)
        if question.startswith(item["question"]):
            retrieval_result = item["retrieval_result"]
            break
    if not retrieval_result:
        keywords, retrieval_result, schema_string = preprocessing(question=question, db_mode=db_mode, db_id=db_id, model=model, temperature=temperature)
    else:
        schema_with_examples = retrieval_result["similar_values"]
        db_manager = DatabaseManager(db_mode=db_mode, db_id=db_id)
        schema_string = db_manager.get_database_schema_string(
            tentative_schema=get_db_schema(db_manager.db_path), 
            schema_with_examples=schema_with_examples, 
            schema_with_descriptions=None, 
            include_value_description=True
        )
    
    task = (
        f"Question: {question}"
        # f"Hint: {hint}"
    )
    response = divide_and_conquer(task=task, schema_string=schema_string, model=model, temperature=temperature)
    
    result = {
        "response": response,
        "retrieval_result": retrieval_result,
        "schema_string": schema_string,
    }
    return result


def DCA_basic_pipeline(question, db_mode, db_id, model:str="gpt-4o", temperature: float = 0.0):

    retrieval_result_file = f"preprocessing_outputs/basic/{model}/retrieval_result.jsonl"
    retrieval_result = {}
    for line in open(retrieval_result_file):
        item = json.loads(line)
        if question.startswith(item["question"]):
            retrieval_result = item["retrieval_result"]
            break
    if not retrieval_result:
        keywords, retrieval_result, schema_string = preprocessing(question=question, db_mode=db_mode, db_id=db_id, model=model, temperature=temperature)
    else:
        schema_with_examples = retrieval_result["similar_values"]
        db_manager = DatabaseManager(db_mode=db_mode, db_id=db_id)
        schema_string = db_manager.get_database_schema_string(
            tentative_schema=get_db_schema(db_manager.db_path), 
            schema_with_examples=schema_with_examples, 
            schema_with_descriptions=None, 
            include_value_description=True
        )
    
    task = (
        f"Question: {question}"
        # f"Hint: {hint}"
    )
    
    divide_response = divide(task=task, schema_string=schema_string, model=model, temperature=temperature)
    divide_trajectories = divide_response
    conquer_response = conquer(task=task, schema_string=schema_string, divide_trajectories=divide_trajectories, model=model, temperature=temperature)
    conquer_trajectories = divide_response
    assemble_response = assemble(task=task, schema_string=schema_string, conquer_trajectories=conquer_trajectories, model=model, temperature=temperature)
    
    result = {
        "response": assemble_response,
        "divide_response": divide_response,
        "conquer_response": conquer_response,
        "assemble_response": assemble_response,
        "retrieval_result": retrieval_result,
        "schema_string": schema_string,
    }
    return result



def DC_fix_pipeline(question, db_mode, db_id, model:str="gpt-4o", temperature: float = 0.0, num_fix: int = 3):

    retrieval_result_file = f"preprocessing_outputs/basic/{model}/retrieval_result.jsonl"
    retrieval_result = {}
    for line in open(retrieval_result_file):
        item = json.loads(line)
        if question.startswith(item["question"]):
            retrieval_result = item["retrieval_result"]
            break
    if not retrieval_result:
        keywords, retrieval_result, schema_string = preprocessing(question=question, db_mode=db_mode, db_id=db_id, model=model, temperature=temperature)
    else:
        schema_with_examples = retrieval_result["similar_values"]
        db_manager = DatabaseManager(db_mode=db_mode, db_id=db_id)
        schema_string = db_manager.get_database_schema_string(
            tentative_schema=get_db_schema(db_manager.db_path), 
            schema_with_examples=schema_with_examples, 
            schema_with_descriptions=None, 
            include_value_description=True
        )
    
    task = (
        f"Question: {question}"
        # f"Hint: {hint}"
    )

    trajectories = []
    steps = []
    try:
        response = divide_and_conquer(task=task, schema_string=schema_string, model=model, temperature=temperature)
        print(f"Candidate generation answer: {response}")

    except Exception as e:
        print(f"Error in divide_and_conquer: {e}")
        response = str(e)

    steps.append({
        "response": response,
    })

    try:
        sql = parse_last_sql_code(parse_sql(response))
        db_manager = DatabaseManager(db_mode=db_mode, db_id=db_id)
        exec_result = execute_sql(db_manager.db_path, sql)
    except Exception as e:
        exec_result = str(e)
        print(f"Error in execute_sql: {e}")
        
    for i in range(num_fix):
        print(f"Round {i}")
        
        if not is_valid_exec_result(exec_result):
            trajectory = {
                "SQL": sql,
                "Execution Result": exec_result,
            }
            try:
                fixed_sql = query_fix(
                    question=question, 
                    trajectory=trajectory, 
                    schema_string=schema_string, 
                    model=model, 
                    temperature=temperature)
                print(f"Query fix: {fixed_sql}")
            except Exception as e:
                fixed_sql = ""
                print(f"Error in query fix: {e}")
            trajectory["Query Fix"] = fixed_sql
            trajectories.append(trajectory)
        else:
            print("Answer is valid")
            break

    result = {
        "response": response,
        "retrieval_result": retrieval_result,
        "schema_string": schema_string,
        "steps": steps,
        "trajectories": trajectories,
    }
    return result


def QP_basic_pipeline(question, db_mode, db_id, model:str="gpt-4o", temperature: float = 0.0):

    retrieval_result_file = f"preprocessing_outputs/basic/{model}/retrieval_result.jsonl"
    retrieval_result = {}
    for line in open(retrieval_result_file):
        item = json.loads(line)
        if question.startswith(item["question"]):
            retrieval_result = item["retrieval_result"]
            break
    if not retrieval_result:
        keywords, retrieval_result, schema_string = preprocessing(question=question, db_mode=db_mode, db_id=db_id, model=model, temperature=temperature)
    else:
        schema_with_examples = retrieval_result["similar_values"]
        db_manager = DatabaseManager(db_mode=db_mode, db_id=db_id)
        schema_string = db_manager.get_database_schema_string(
            tentative_schema=get_db_schema(db_manager.db_path), 
            schema_with_examples=schema_with_examples, 
            schema_with_descriptions=None, 
            include_value_description=True
        )
    
    task = (
        f"Question: {question}"
        # f"Hint: {hint}"
    )
    response = query_plan(task=task, schema_string=schema_string, model=model, temperature=temperature)
    result = {
        "response": response,
        "retrieval_result": retrieval_result,
        "schema_string": schema_string,
    }
    return result


def OS_basic_pipeline(question, db_mode, db_id, model:str="gpt-4o", temperature: float = 0.0):

    retrieval_result_file = f"preprocessing_outputs/basic/{model}/retrieval_result.jsonl"
    retrieval_result = {}
    for line in open(retrieval_result_file):
        item = json.loads(line)
        if question.startswith(item["question"]):
            retrieval_result = item["retrieval_result"]
            break
    if not retrieval_result:
        keywords, retrieval_result, schema_string = preprocessing(question=question, db_mode=db_mode, db_id=db_id, model=model, temperature=temperature)
    else:
        schema_with_examples = retrieval_result["similar_values"]
        db_manager = DatabaseManager(db_mode=db_mode, db_id=db_id)
        schema_string = db_manager.get_database_schema_string(
            tentative_schema=get_db_schema(db_manager.db_path), 
            schema_with_examples=schema_with_examples, 
            schema_with_descriptions=None, 
            include_value_description=True
        )
    
    task = (
        f"Question: {question}"
        # f"Hint: {hint}"
    )
    
    examples_with_sql_feature_response = online_synthetic_with_sql_feature(task=task, schema_string=schema_string, model=model, temperature=temperature)
    examples_with_sql_feature = extract_examples(examples_with_sql_feature_response)
    examples_with_schema_response = online_synthetic_with_schema(task=task, schema_string=schema_string, model=model, temperature=temperature)
    examples_with_schema = extract_examples(examples_with_schema_response)
    examples = examples_with_sql_feature + examples_with_schema
    response = candidate_generation_with_examples(question=question, schema_string=schema_string, examples=examples, model=model, temperature=temperature)
    
    result = {
        "response": response,
        "examples_with_sql_feature_response": examples_with_sql_feature_response,
        "examples_with_schema_response": examples_with_schema_response,
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
    db_mode = "dev"
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
    schema_with_examples = retrieval_result["similar_values"]
    db_manager = DatabaseManager(db_mode=db_mode, db_id=db_id)
    schema_string = db_manager.get_database_schema_string(
        tentative_schema=get_db_schema(db_manager.db_path), 
        schema_with_examples=schema_with_examples, 
        schema_with_descriptions=None, 
        include_value_description=True
    )
    
    model = "claude-3-5-sonnet-20240620"
    model = "claude-3-5-sonnet-20241022"
    # model = "gpt-4o"
    # model = "gpt-4o-2024-08-06"
    temperature = 1.0

    task = (
        f"Question: {question}"
        # f"Hint: {hint}"
    )
    # response = divide_and_conquer(task=task, schema_string=schema_string, model=model, temperature=temperature)
    # response = query_plan(task=task, schema_string=schema_string, model=model, temperature=temperature)
    
    # response = divide(task=task, schema_string=schema_string, model=model, temperature=temperature)
    # divide_trajectories = response
    # divide_response = conquer(task=task, schema_string=schema_string, divide_trajectories=divide_trajectories, model=model, temperature=temperature)
    # conquer_trajectories = divide_response
    # response = assemble(task=task, schema_string=schema_string, conquer_trajectories=conquer_trajectories, model=model, temperature=temperature)
    
    examples_with_sql_feature = online_synthetic_with_sql_feature(task=task, schema_string=schema_string, model=model, temperature=temperature)
    examples_with_sql_feature = extract_examples(examples_with_sql_feature)
    examples_with_schema = online_synthetic_with_schema(task=task, schema_string=schema_string, model=model, temperature=temperature)
    examples_with_schema = extract_examples(examples_with_schema)
    examples = examples_with_sql_feature + examples_with_schema
    response = candidate_generation_with_examples(question=question, schema_string=schema_string, examples=examples, model=model, temperature=temperature)
    

    print(response)
    