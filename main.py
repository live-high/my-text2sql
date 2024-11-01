import os
import json
import argparse
from sqlagent.core.basic import basic_pipeline
from sqlagent.core.reflexion import reflexion_pipeline


def main(eval_data, pass_ids, model, temperature, pred_sql_path, pipeline_type):
    success_ids = []
    pred_sqls = {}
    for question_id, example in enumerate(eval_data):
        if question_id in pass_ids:
            success_ids.append(question_id)
            continue
        example["question_id"] = question_id
        question = example["question"]
        db_id = example["db_id"]
        db_mode = "dev"
        hint = example["evidence"]
        question = f"{question}\nHint: {hint}\n"
        try:
            if pipeline_type == "basic":
                result = basic_pipeline(question=question, db_mode=db_mode, db_id=db_id, model=model, temperature=temperature)
            elif pipeline_type == "reflexion":
                reflection_num = 3
                result = reflexion_pipeline(
                    question=question, db_mode=db_mode, db_id=db_id,
                    model=model, temperature=temperature, num_reflection=reflection_num,
                )
            success_ids.append(question_id)
        except Exception as e:
            print(f"Error in {pipeline_type} pipeline: {e}")
            continue
        example["pred"] = result

        pred_sqls[question_id] = example

        with open(pred_sql_path, 'a') as f:
            f.write(json.dumps(example) + "\n")
    return success_ids



if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--eval_path', type=str, default='')
    args_parser.add_argument('--pipeline_type', type=str, default='basic', choices=['basic', 'reflexion'])
    args_parser.add_argument('--model', type=str, default='claude-3-5-sonnet-20241022')
    args_parser.add_argument('--temperature', type=float, default=1.0)
    args_parser.add_argument('--output_path', type=str, default='.')
    args = args_parser.parse_args()

    eval_path = args.eval_path
    model = args.model
    temperature = args.temperature
    output_path = args.output_path
    pipeline_type = args.pipeline_type

    eval_data = json.load(open(eval_path, 'r'))

    
    pred_sql_path = f"{output_path}/{model}_pred_sql.json"
    pass_ids = []
    if os.path.exists(pred_sql_path):
        with open(pred_sql_path, 'r') as f:
            for line in f:
                if line == '\n':
                    continue
                example = json.loads(line)
                pass_ids.append(example["question_id"])
    failed_ids = [i for i in range(len(eval_data)) if i not in pass_ids]
    print(f"Failed ids: {failed_ids}")
    while failed_ids:
        pass_ids = main(eval_data, pass_ids, model, temperature, pred_sql_path, pipeline_type)
        print(f"Successfully processed {len(pass_ids)}/{len(eval_data)} questions")
        failed_ids = [i for i in range(len(eval_data)) if i not in pass_ids]
        print(f"Failed to process {len(failed_ids)} questions: {failed_ids}")