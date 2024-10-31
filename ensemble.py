from collections import defaultdict
import argparse
import json
from sqlagent.utils import parse_sql
from sqlagent.postprocessing.self_consistency import aggregate_sqls


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

    ensemble_model_id = "+".join([i.split('/')[-1] for i in ensemble_model])
    output_file = f"{emsemble_output_path}/{ensemble_method}_{ensemble_model_id}_pred_sql.json"
    # with open(output_file, 'w') as f:
    #     f.write("")

    if ensemble_method == "agg":
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

        for question_id, example in enumerate(eval_data):
            db_id = example["db_id"]
            DB_MODE = "dev"
            DB_ID = db_id
            predictions = model_preds.get(question_id, {})
            sqls = [parse_sql(v["response"]) for v in predictions.values()]
            final_sql = aggregate_sqls(sqls, db_mode=DB_MODE, db_id=DB_ID)
            res = {}
            example["question_id"] = question_id
            example["pred"] = {
                "response": final_sql,
                "ensemble_predictions": predictions,
            }
            
            with open(output_file, 'a') as f:
                f.write(json.dumps(example) + "\n")
        print(f"Ensemble results have been saved to {output_file}")
            
