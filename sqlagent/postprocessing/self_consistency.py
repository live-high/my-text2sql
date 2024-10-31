
from sqlagent.runner.database_manager import DatabaseManager


def aggregate_sqls(sqls, db_mode, db_id):
    db_manager = DatabaseManager(db_mode=db_mode, db_id=db_id)
    results = []
    for sql in sqls:
        try:
            result = db_manager.execute_sql(sql)
            res = {"SQL": sql, "RESULT": result, "STATUS": "OK"}
        except Exception as e:
            print(f"Error in validate_sql_query: {e}")
            res = {"SQL": sql, "RESULT": str(e), "STATUS": "ERROR"}
        results.append(res)
    
    clusters = {}
    # Group queries by unique result sets
    for result in results:
        if result['STATUS'] == 'OK':
            # Using a frozenset as the key to handle unhashable types like lists
            key = frozenset(tuple(row) for row in result['RESULT'])
            if key in clusters:
                clusters[key].append(result['SQL'])
            else:
                clusters[key] = [result['SQL']]
                
    if clusters:
        # Find the largest cluster
        largest_cluster = max(clusters.values(), key=len, default=[])
        # Select the shortest SQL query from the largest cluster
        if largest_cluster:
            return min(largest_cluster, key=len)
    
    raise Exception("No valid SQL clusters found. Returning the first SQL query.")
    


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
    ret = aggregate_sqls(sqls, db_mode=DB_MODE, db_id=DB_ID)
    print(ret)