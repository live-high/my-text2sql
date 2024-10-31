import regex as re
import os
from langchain_openai import ChatOpenAI
from sqlagent.preprocessing.entity_retrieval import entity_retrieval
from sqlagent.database_utils.execution import execute_sql
from sqlagent.runner.database_manager import DatabaseManager
from sqlagent.database_utils.db_info import get_db_schema

TEMPLATES_ROOT_PATH = os.environ["TEMPLATES_ROOT_PATH"]

def parse_sql(output_str):
    sql = re.findall(r"\"SQL\": \"(.*?)\"", output_str, re.DOTALL)
    if sql:
        sql = sql[0]
        if '\n' in sql:
            sql = re.sub(r"\n", ' ', sql)
        elif '\\n' in sql:
            sql = re.sub(r"\\n", ' ', sql)
        elif '\\\n' in sql:
            sql = re.sub(r"\\\n", ' ', sql)
        sql = re.sub(r"\s+", ' ', sql)
    else:
        sql = output_str
    return sql


def format_trajectories(trajectories: list):
    if not trajectories:
        return ""
    formatted_trajectories = "\nPrevious trials:\n"
    for trajectory in trajectories:
        for key, value in trajectory.items():
            formatted_trajectories += f"{key}: {value}\n"
        formatted_trajectories += "\n"
    formatted_trajectories += "\n(END PREVIOUS TRIALS)\n\n"
    return formatted_trajectories


def keyword_extraction_with_trajectories(question: str, trajectories: list, model: str = "gpt-4o", temperature: float = 0.0):
    """
    Extracts keywords from the question.
    """
    template_name = "keyword_extraction_with_trajectories"
    file_name = f"template_{template_name}.txt"
    template_path = os.path.join(TEMPLATES_ROOT_PATH, file_name)
    KEYWORD_EXTRACTION_PROMPT = ""
    with open(template_path, "r") as file:
        KEYWORD_EXTRACTION_PROMPT = file.read()

    task = (
        f"Question: {question}"
        # f"Hint: {hint}"
    )
    
    prompt = KEYWORD_EXTRACTION_PROMPT.format(TASK=task, TRAJECTORIES=format_trajectories(trajectories))
    llm = ChatOpenAI(model=model, temperature=temperature)
    response = llm.invoke(prompt)
    response = response.content
    
    try:
        keywords = eval(response)
    except:
        print(f"Failed to extract keywords: {response}")
        keywords = []
    return keywords


def candidate_generation_with_trajectories(question: str, schema_string: str, trajectories: list, model:str="gpt-4o", temperature: float = 0.0):
    
    template_name = "candidate_generation_with_trajectories"
    file_name = f"template_{template_name}.txt"
    template_path = os.path.join(TEMPLATES_ROOT_PATH, file_name)
    CANDIDATE_GENERATION_PROMPT = ""
    with open(template_path, "r") as file:
        CANDIDATE_GENERATION_PROMPT = file.read()

    task = (
        f"Question: {question}"
    )
    
    prompt = CANDIDATE_GENERATION_PROMPT.format(DATABASE_SCHEMA=schema_string, TASK=task, TRAJECTORIES=format_trajectories(trajectories))
    
    llm = ChatOpenAI(model=model, temperature=temperature)
    response = llm.invoke(prompt)
    response = response.content
    return response


def self_reflect(question: str, trajectories: list, schema_string: str, model:str="gpt-4o", temperature: float = 0.0):
    
    template_name = "reflection"
    # template_name = "reflection_with_exec"
    file_name = f"template_{template_name}.txt"
    template_path = os.path.join(TEMPLATES_ROOT_PATH, file_name)
    REFLECTION_PROMPT = ""
    with open(template_path, "r") as file:
        REFLECTION_PROMPT = file.read()

    task = (
        f"Question: {question}"
        # f"Hint:\n{hint}"
    )
    
    prompt = REFLECTION_PROMPT.format(DATABASE_SCHEMA=schema_string, TASK=task, TRAJECTORIES=format_trajectories(trajectories))
    
    llm = ChatOpenAI(model=model, temperature=temperature)
    response = llm.invoke(prompt)
    response = response.content
    return response


def is_valid(exec_result):
    if isinstance(exec_result, Exception):
        return False
    if isinstance(exec_result, str):
        return False
    if not exec_result:
        return False
    for item in exec_result:
        if item is None:
            return False
        if not item:
            return False
        if isinstance(item, list):
            for sub_item in item:
                if sub_item is None:
                    return False
    return True

def reflexion_pipeline(
        question: str,
        db_mode: str, db_id: str, 
        model: str, temperature: float,
        num_reflection: int = 3,
    ):
    keywords = []
    retrieval_result = {"similar_values": {}, "similar_columns": {}}
    schema_string = ""
    response = ""
    trajectories = []
    steps = []
    for i in range(num_reflection):
        print(f"Round {i}")
        try:
            keywords += keyword_extraction_with_trajectories(question=question, trajectories=trajectories, model=model, temperature=temperature)
            keywords = sorted(set(keywords))
            print(f"Keywords: {keywords}")
        except Exception as e:
            print(f"Error in keyword_extraction_with_trajectories: {e}")
            
        retrieval_result_i = entity_retrieval(question=question, keywords=keywords, db_mode=db_mode, db_id=db_id)
        for table, column_values in retrieval_result_i['similar_values'].items():
            for column, values in column_values.items():
                similar_values = retrieval_result['similar_values'].get(table, {}).get(column, [])
                similar_values += values
                similar_values = sorted(set(similar_values))
                if table not in retrieval_result['similar_values']:
                    retrieval_result['similar_values'][table] = {}
                retrieval_result['similar_values'][table][column] = similar_values
        for table, columns in retrieval_result_i['similar_columns'].items():
            similar_columns = retrieval_result['similar_columns'].get(table, [])
            similar_columns += columns
            similar_columns = sorted(set(similar_columns))
            retrieval_result['similar_columns'][table] = similar_columns

        print(f"Retrieval result: {retrieval_result}")
        
        schema_with_examples = retrieval_result["similar_values"]
        db_manager = DatabaseManager(db_mode=db_mode, db_id=db_id)
        schema_string = db_manager.get_database_schema_string(
            tentative_schema=get_db_schema(db_manager.db_path), 
            schema_with_examples=schema_with_examples, 
            schema_with_descriptions=None, 
            include_value_description=True
        )

        try:
            response = candidate_generation_with_trajectories(
                question=question, schema_string=schema_string, 
                trajectories=trajectories, 
                model=model, temperature=temperature)
            print(f"Candidate generation answer: {response}")

        except Exception as e:
            print(f"Error in candidate_generation_with_trajectories: {e}")
            continue

        steps.append({
            "keywords": keywords,
            "retrieval_result": retrieval_result,
            "schema_string": schema_string,
            "response": response,
        })
        
        try:
            sql = parse_sql(response)
            exec_result = execute_sql(db_manager.db_path, sql)
        except Exception as e:
            exec_result = str(e)
            print(f"Error in execute_sql: {e}")

        if not is_valid(exec_result):
            trajectories.append({
                "SQL": sql,
                "Execution Result": exec_result,
            })
            try:
                reflection = self_reflect(
                    question=question, 
                    trajectories=trajectories, 
                    schema_string=schema_string, 
                    model=model, 
                    temperature=temperature)
                print(f"Reflection: {reflection}")
            except Exception as e:
                reflection = ""
                print(f"Error in reflection: {e}")
            trajectories[-1]["Reflection"] = reflection
        else:
            print("Answer is valid")
            break
            # if re.findall(r"\"is_correct\": \"[Tt][Rr][Uu][Ee]\"|\"is_correct\": [Tt][Rr][Uu][Ee]", reflection):
            #     print("Answer is correct")
            #     break

        # ground_truth_result = execute_sql(db_manager.db_path, ground_truth)
        # print(f"Ground truth: {ground_truth}")
        # if set(exec_result) != set(ground_truth_result):
        #     trajectories.append({
        #         # "SQL": sql,
        #         "Answer": response,
        #     })
        #     try:
        #         reflection = self_reflect(question=question, trajectories=trajectories, schema_string=schema_string, model=model, temperature=temperature)
        #         print(f"Reflection: {reflection}")
        #     except Exception as e:
        #         reflection = ""
        #         print(f"Error in reflection: {e}")
        #     trajectories[-1]["Reflection"] = reflection
        # else:
        #     print("Answer is correct")
        #     break


    result = {
        "response": response,
        "steps": steps,
        "trajectories": trajectories,
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

    # model = "claude-3-5-sonnet-20240620"
    model = "claude-3-5-sonnet-20241022"
    temperature = 1.0
    reflexion_pipeline(question=question, db_mode=DB_MODE, db_id=DB_ID, model=model, temperature=temperature)
