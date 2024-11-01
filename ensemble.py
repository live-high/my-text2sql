from collections import defaultdict
import argparse
import json
from func_timeout import func_timeout, FunctionTimedOut
from sqlagent.utils import parse_sql
from sqlagent.postprocessing.self_consistency import aggregate_sqls
from sqlagent.postprocessing.selector import selector
from sqlagent.runner.database_manager import DatabaseManager
from sqlagent.database_utils.db_info import get_db_schema



if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--eval_path', type=str, default='')
    args_parser.add_argument('--ensemble_model', type=str, default='')
    args_parser.add_argument('--model_pred_path', type=str, default='.')
    args_parser.add_argument('--emsemble_output_path', type=str, default='.')
    args_parser.add_argument('--ensemble_method', type=str, default='agg')
    args = args_parser.parse_args()

    eval_path = args.eval_path
    ensemble_model = args.ensemble_model
    model_pred_path = args.model_pred_path
    ensemble_method = args.ensemble_method
    emsemble_output_path = args.emsemble_output_path
    ensemble_model = ensemble_model.split(",")

    eval_data = json.load(open(eval_path, 'r'))

    ensemble_model_id = "+".join([i.replace("/", "_") for i in ensemble_model])
    output_file = f"{emsemble_output_path}/{ensemble_method}_{ensemble_model_id}_pred_sql.json"
    with open(output_file, 'w') as f:
        f.write("")
    
    model_preds = defaultdict(dict)
    for model in ensemble_model:
        pred_sql_path = f"{model_pred_path}/{model}_pred_sql.json"
        with open(pred_sql_path, 'r') as f:
            for line in f:
                example = json.loads(line)
                prediction = example["pred"]
                db_id = example["db_id"]
                question_id = example["question_id"]
                model_preds[question_id][model] = prediction

    if ensemble_method == "agg":
        meta_time_out = 30

        for question_id, example in enumerate(eval_data):
            db_id = example["db_id"]
            db_mode = "dev"
            predictions = model_preds.get(question_id, {})
            sqls = [parse_sql(v["response"]) for v in predictions.values()]
            
            error = ""
            try:
                final_sql = func_timeout(
                    meta_time_out, aggregate_sqls, args=(sqls, db_mode, db_id))
            except FunctionTimedOut:
                error = "Timeout"
                print(f"Timeout in {question_id}")
                final_sql = sqls[0]
            except Exception as e:
                error = str(e)
                print(f"Error in {question_id}")
                final_sql = sqls[0]

            example["question_id"] = question_id
            example["pred"] = {
                "response": final_sql,
                "ensemble_predictions": predictions,
                "error": error
            }
            
            with open(output_file, 'a') as f:
                f.write(json.dumps(example) + "\n")
        print(f"Ensemble results have been saved to {output_file}")
            
    elif ensemble_method == "selector":
        meta_time_out = 30
        selector_model = "claude-3-5-sonnet-20241022"
        selector_temperature = 1.0
        for question_id, example in enumerate(eval_data):
            db_id = example["db_id"]
            question = example["question"]
            db_mode = "dev"
            predictions = model_preds.get(question_id, {})
            sqls = [parse_sql(v["response"]) for v in predictions.values()]

            retrieval_result = {"similar_values": {}, "similar_columns": {}}
            for pred in predictions.values():
                if "steps" in pred:
                    retrieval_result_i = pred["steps"][-1]["retrieval_result"]
                else:
                    retrieval_result_i = pred["retrieval_result"]
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
            
            schema_with_examples = retrieval_result["similar_values"]
            db_manager = DatabaseManager(db_mode=db_mode, db_id=db_id)
            schema_string = db_manager.get_database_schema_string(
                tentative_schema=get_db_schema(db_manager.db_path), 
                schema_with_examples=schema_with_examples, 
                schema_with_descriptions=None, 
                include_value_description=True
            )
            error = ""
            try:
                final_sql = func_timeout(
                    meta_time_out, 
                    selector, 
                    args=(sqls, question, schema_string, db_mode, db_id, selector_model, selector_temperature)
                )
            except FunctionTimedOut:
                error = "Timeout"
                print(f"Timeout in {question_id}")
                final_sql = sqls[0]
            except Exception as e:
                error = str(e)
                print(f"Error in {question_id}")
                final_sql = sqls[0]


            example["question_id"] = question_id
            example["pred"] = {
                "response": final_sql,
                "ensemble_predictions": predictions,
                "error": error
            }
            
            with open(output_file, 'a') as f:
                f.write(json.dumps(example) + "\n")
        print(f"Ensemble results have been saved to {output_file}")