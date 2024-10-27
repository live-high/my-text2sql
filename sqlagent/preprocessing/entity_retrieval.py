
import numpy as np
from typing import Any, Dict, List, Tuple
from langchain_openai import OpenAIEmbeddings
import logging
import concurrent.futures
import difflib
import os
import dotenv
from sqlagent.runner.database_manager import DatabaseManager


EMBEDDING_FUNCTION = OpenAIEmbeddings(model="text-embedding-3-small")


def entity_retrieval(question: str, keywords: List[str], db_mode: str, db_id: str) -> Dict[str, Any]:
    """
    Retrieves entities and columns similar to given question from the database.

    Args:
        question (str): The question string.
        keywords (Dict[str]): The keywords to retrieve entities for.
    Returns:
        Dict[str, Any]: A dictionary containing similar columns and values.
    """
    logging.info("Starting entity retrieval")

    similar_columns = get_similar_columns(keywords=keywords, question=question, db_mode=db_mode, db_id=db_id)
    result = {"similar_columns": similar_columns}
    
    similar_values = get_similar_entities(keywords=keywords, db_mode=db_mode, db_id=db_id)
    result["similar_values"] = similar_values

    logging.info("Entity retrieval completed successfully")
    return result


def get_similar_columns(keywords: List[str], question: str, db_mode: str, db_id: str) -> Dict[str, List[str]]:
    """
    Finds columns similar to given keywords based on question and hint.

    Args:
        keywords (List[str]): The list of keywords.
        question (str): The question string.
        hint (str): The hint string.

    Returns:
        Dict[str, List[str]]: A dictionary mapping table names to lists of similar column names.
    """
    logging.info("Retrieving similar columns")
    selected_columns = {}
    similar_columns = _get_similar_column_names(potential_column_names=keywords, question=question, db_mode=db_mode, db_id=db_id)
    for table_name, column_name in similar_columns:
        selected_columns.setdefault(table_name, []).append(column_name)
    return selected_columns


def _get_similar_column_names(potential_column_names: List[str], question: str, db_mode: str, db_id: str, threshold: float = 0.8) -> List[Tuple[str, str]]:
    """
    Finds column names similar to a keyword.

    Args:
        potential_column_names (List[str]): The list of potential column names.
        question (str): The question string.

    Returns:
        List[Tuple[str, str]]: A list of tuples containing table and column names.
    """
    schema = DatabaseManager(db_mode=db_mode, db_id=db_id).get_db_schema()

    similar_column_names = []
    for table, columns in schema.items():
        for column in columns:
            for potential_column_name in potential_column_names:
                if _does_keyword_match_column(potential_column_name, column, threshold=threshold):
                    similarity_score = _get_semantic_similarity_with_openai(f"`{table}`.`{column}`", [question])[0]
                    similar_column_names.append((table, column, similarity_score))

    similar_column_names.sort(key=lambda x: x[2], reverse=True)
    return [(table, column) for table, column, _ in similar_column_names[:1]]


def get_similar_entities(keywords: List[str], db_mode: str, db_id: str) -> Dict[str, Dict[str, List[str]]]:
    """
    Retrieves similar entities from the database based on keywords.

    Args:
        keywords (List[str]): The list of keywords.

    Returns:
        Dict[str, Dict[str, List[str]]]: A dictionary mapping table and column names to similar entities.
    """
    logging.info("Retrieving similar entities")
    selected_values = {}

    def get_similar_values_target_string(target_string: str):
        unique_similar_values = DatabaseManager(db_mode=db_mode, db_id=db_id).query_lsh(keyword=target_string, signature_size=100, top_n=10)
        return target_string, _get_similar_entities_to_keyword(target_string, unique_similar_values)

        
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(get_similar_values_target_string, ts): ts for ts in keywords}
        for future in concurrent.futures.as_completed(futures):
            target_string, similar_values = future.result()
            for table_name, column_values in similar_values.items():
                for column_name, entities in column_values.items():
                    if entities:
                        selected_values.setdefault(table_name, {}).setdefault(column_name, []).extend(
                            [(ts, value, edit_distance, embedding) for ts, value, edit_distance, embedding in entities]
                        )

    for table_name, column_values in selected_values.items():
        for column_name, values in column_values.items():
            max_edit_distance = max(values, key=lambda x: x[2])[2]
            selected_values[table_name][column_name] = list(set(
                value for _, value, edit_distance, _ in values if edit_distance == max_edit_distance
            ))
    return selected_values


def _get_similar_entities_to_keyword(keyword: str, unique_values: Dict[str, Dict[str, List[str]]]) -> Dict[str, Dict[str, List[Tuple[str, str, float, float]]]]:
    """
    Finds entities similar to a keyword in the database.

    Args:
        keyword (str): The keyword to find similar entities for.
        unique_values (Dict[str, Dict[str, List[str]]]): The dictionary of unique values from the database.

    Returns:
        Dict[str, Dict[str, List[Tuple[str, str, float, float]]]]: A dictionary mapping table and column names to similar entities.
    """
    return {
        table_name: {
            column_name: _get_similar_values(keyword, values)
            for column_name, values in column_values.items()
        }
        for table_name, column_values in unique_values.items()
    }


def _get_similar_values(target_string: str, values: List[str]) -> List[Tuple[str, str, float, float]]:
    """
    Finds values similar to the target string based on edit distance and embedding similarity.

    Args:
        target_string (str): The target string to compare against.
        values (List[str]): The list of values to compare.

    Returns:
        List[Tuple[str, str, float, float]]: A list of tuples containing the target string, value, edit distance, and embedding similarity.
    """
    edit_distance_threshold = 0.3
    top_k_edit_distance = 5
    embedding_similarity_threshold = 0.6
    top_k_embedding = 1

    edit_distance_similar_values = [
        (value, difflib.SequenceMatcher(None, value.lower(), target_string.lower()).ratio())
        for value in values
        if difflib.SequenceMatcher(None, value.lower(), target_string.lower()).ratio() >= edit_distance_threshold
    ]
    edit_distance_similar_values.sort(key=lambda x: x[1], reverse=True)
    edit_distance_similar_values = edit_distance_similar_values[:top_k_edit_distance]
    similarities = _get_semantic_similarity_with_openai(target_string, [value for value, _ in edit_distance_similar_values])
    embedding_similar_values = [
        (target_string, edit_distance_similar_values[i][0], edit_distance_similar_values[i][1], similarities[i])
        for i in range(len(edit_distance_similar_values))
        if similarities[i] >= embedding_similarity_threshold
    ]

    embedding_similar_values.sort(key=lambda x: x[2], reverse=True)
    return embedding_similar_values[:top_k_embedding]


def _does_keyword_match_column(keyword: str, column_name: str, threshold: float = 0.9) -> bool:
    """
    Checks if a keyword matches a column name based on similarity.

    Args:
        keyword (str): The keyword to match.
        column_name (str): The column name to match against.
        threshold (float, optional): The similarity threshold. Defaults to 0.9.

    Returns:
        bool: True if the keyword matches the column name, False otherwise.
    """
    keyword = keyword.lower().replace(" ", "").replace("_", "").rstrip("s")
    column_name = column_name.lower().replace(" ", "").replace("_", "").rstrip("s")
    similarity = difflib.SequenceMatcher(None, column_name, keyword).ratio()
    return similarity >= threshold


def _get_semantic_similarity_with_openai(target_string: str, list_of_similar_words: List[str]) -> List[float]:
    """
    Computes semantic similarity between a target string and a list of similar words using OpenAI embeddings.

    Args:
        target_string (str): The target string to compare.
        list_of_similar_words (List[str]): The list of similar words to compare against.

    Returns:
        List[float]: A list of similarity scores.
    """
    target_string_embedding = EMBEDDING_FUNCTION.embed_query(target_string)
    all_embeddings = EMBEDDING_FUNCTION.embed_documents(list_of_similar_words)
    similarities = [np.dot(target_string_embedding, embedding) for embedding in all_embeddings]
    return similarities


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
    keywords = ["Charter Funding Type", "FundingType", "directly charter-funded"]
    keywords = ["Charter Funding Type"]
    # DB_ROOT_PATH = f"/Users/yanghuanye/clientstores/CHESS/data/"
    retrieval_result = entity_retrieval(question=question, keywords=keywords, db_mode=DB_MODE, db_id=DB_ID)
    print(retrieval_result)