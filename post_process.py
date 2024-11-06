import json
import regex as re
import argparse
from sqlagent.utils import parse_sql, parse_last_sql_code


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--result_file', type=str)
    args_parser.add_argument('--output_file', type=str)
    args_parser.add_argument('--mode', type=str, default="sql", choices=["sql", "retrieval_result"])
    

    args = args_parser.parse_args()
    result_file = args.result_file
    output_file = args.output_file
    mode = args.mode

    if mode == "sql":

        results = {}
        with open(args.result_file, 'r') as f:
            for line in f:
                if line == '\n':
                    continue
                example = json.loads(line)
                prediction = example["pred"]["response"]
                db_id = example["db_id"]
                question_id = example["question_id"]
                if question_id not in results:
                    results[question_id] = (prediction, db_id)
                    
        for question_id, (prediction, db_id) in results.items():
            sql = prediction
            sql = parse_sql(sql)
            sql = parse_last_sql_code(sql)
            results[question_id] = sql + '\t----- bird -----\t' + db_id

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
    
    
    elif mode == "retrieval_result":

        results = {}
        with open(args.result_file, 'r') as f:
            for line in f:
                if line == '\n':
                    continue
                example = json.loads(line)
                question_id = example["question_id"]
                question = example["question"]
                gt_sql = example["SQL"]
                if "pred" in example:
                    pred = example["pred"]
                    if "retrieval_result" not in pred and "steps" in pred:
                        retrieval_result = pred["steps"][-1]["retrieval_result"]
                    elif "retrieval_result" in pred:
                        retrieval_result = pred["retrieval_result"]
                    else:
                        raise ValueError("No retrieval result found")
                else:
                    raise ValueError("No prediction found")
                if question_id not in results:
                    results[question_id] = (retrieval_result, question, gt_sql)
                    
        with open(output_file, 'w') as f:
            f.write("")
        
        results = sorted(results.items(), key=lambda x: int(x[0]))

        with open(output_file, 'a') as f:
            for question_id, (retrieval_result, question, gt_sql) in results:
                item = {
                    "question_id": question_id,
                    "question": question,
                    "SQL": gt_sql,
                    "retrieval_result": retrieval_result,
                    }
                f.write(json.dumps(item) + '\n')

