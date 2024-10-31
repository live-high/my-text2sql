import json
import regex as re
import argparse
from sqlagent.utils import parse_sql


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--result_file', type=str)
    args_parser.add_argument('--output_file', type=str)

    args = args_parser.parse_args()
    result_file = args.result_file
    output_file = args.output_file

    results = {}
    with open(args.result_file, 'r') as f:
        for line in f:
            example = json.loads(line)
            prediction = example["pred"]["response"]
            db_id = example["db_id"]
            question_id = example["question_id"]
            if question_id not in results:
                results[question_id] = (prediction, db_id)
                
    for question_id, (prediction, db_id) in results.items():
        sql = parse_sql(prediction)
        results[question_id] = sql + '\t----- bird -----\t' + db_id

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
