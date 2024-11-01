from collections import defaultdict
import sys
import json
import argparse
import sqlite3
import multiprocessing as mp
from func_timeout import func_timeout, FunctionTimedOut
from sqlagent.utils import is_valid_exec_result
empty_count = 0


def load_json(dir):
    with open(dir, 'r') as j:
        contents = json.loads(j.read())
    return contents


def result_callback(result):
    # exec_result.append(result[0])
    exec_result.append(result)


def check_item(item_pred, item_gt):
    ordered_pred = item_pred
    ordered_gt = item_gt
    if ordered_pred == ordered_gt:
        return 3
    elif all(x in ordered_pred for x in ordered_gt):
        return 2
    elif set(ordered_pred) & set(ordered_gt):
        return 1
    return 0


def check_results(res_pred, res_gt, order_by=False):
    if len(res_pred) == len(res_gt):
        scores = [check_item(p, g) for p, g in zip(res_pred, res_gt)]
        if all(i > 1 for i in scores):
            return 1
    return 0


def execute_sql(predicted_sql, ground_truth, db_path):
    conn = sqlite3.connect(db_path)
    # Connect to the database
    cursor = conn.cursor()
    cursor.execute(predicted_sql)
    predicted_res = cursor.fetchall()
    cursor.execute(ground_truth)
    ground_truth_res = cursor.fetchall()
    res = 0
    if set(predicted_res) == set(ground_truth_res):
        res = 1
    res_extra = check_results(predicted_res, ground_truth_res)

    return res, (res_extra, len(predicted_res), is_valid_exec_result(predicted_res))


def execute_model_ori(predicted_sql, ground_truth, db_place, idx, meta_time_out):
    try:
        res = func_timeout(meta_time_out, execute_sql,
                           args=(predicted_sql, ground_truth, db_place))
        res = res[0]
    except KeyboardInterrupt:
        sys.exit(0)
    except FunctionTimedOut:
        result = [(f'timeout',)]
        res = 0
    except Exception as e:
        result = [(f'error',)]  # possibly len(query) > 512 or not executable
        res = 0
    result = {
        'sql_idx': idx,
        'res': res,
    }
    return result


def execute_model(predicted_sql, ground_truth, db_place, idx, meta_time_out):
    try:
        res, exec_msg = func_timeout(meta_time_out, execute_sql,
                                     args=(predicted_sql, ground_truth, db_place))
    except KeyboardInterrupt:
        sys.exit(0)
    except FunctionTimedOut:
        exec_msg = "timeout"
        res = 0
    except Exception as e:
        res = 0
        exec_msg = str(e)
    result = {
        'sql_idx': idx,
        'res': res,
        'error': exec_msg,
        'pred': predicted_sql,
        'gold': ground_truth,
        'db': db_place,
    }
    return result


def package_sqls(sql_path, db_root_path, mode='gpt', data_mode='dev', total_num=None):

    db_path_list = []
    if mode == 'gpt':
        sql_data = json.load(open(sql_path + 'predict_' + data_mode + '.json', 'r'))
        if total_num is not None:
            clean_sqls = [""] * total_num
            db_path_list = [""] * total_num
        else:
            clean_sqls = [""] * len(sql_data)
            db_path_list = [""] * len(sql_data)
        for idx, sql_str in sql_data.items():
            if type(sql_str) == str:
                sql, db_name = sql_str.split('\t----- bird -----\t')
            else:
                sql, db_name = " ", "financial"
                
            clean_sqls[int(idx)] = sql
            db_path_list[int(idx)] = db_root_path + db_name + '/' + db_name + '.sqlite'

    elif mode == 'gt':
        sqls = open(sql_path + data_mode + '_gold.sql')
        sql_txt = sqls.readlines()
        if total_num is not None:
            clean_sqls = [""] * total_num
        else:
            clean_sqls = [""] * len(sql_txt)
        # sql_txt = [sql.split('\t')[0] for sql in sql_txt]
        for idx, sql_str in enumerate(sql_txt):
            sql, db_name = sql_str.strip().split('\t')
            clean_sqls[int(idx)] = sql
            db_path_list.append(db_root_path + db_name + '/' + db_name + '.sqlite')

    return clean_sqls, db_path_list


def run_sqls_parallel(sqls, db_places, num_cpus=1, meta_time_out=30.0):
    pool = mp.Pool(processes=num_cpus)
    for i, sql_pair in enumerate(sqls):

        predicted_sql, ground_truth = sql_pair
        pool.apply_async(
            execute_model,
            args=(predicted_sql, ground_truth, db_places[i], i, meta_time_out),
            callback=result_callback)
    pool.close()
    pool.join()


def sort_results(list_of_dicts):
    return sorted(list_of_dicts, key=lambda x: x['sql_idx'])


def compute_acc_by_diff(exec_results, diff_json_path):
    num_queries = len(exec_results)
    results = {res['sql_idx']: res['res'] for res in exec_results}
    contents = load_json(diff_json_path)
    simple_results, moderate_results, challenging_results = [], [], []

    for i, content in enumerate(contents):
        if i not in results:
            continue
        if content['difficulty'] == 'simple':
            simple_results.append(results[i])

        if content['difficulty'] == 'moderate':
            moderate_results.append(results[i])

        if content['difficulty'] == 'challenging':
            challenging_results.append(results[i])

    simple_acc = sum([res for res in simple_results])/len(simple_results)
    moderate_acc = sum([res for res in moderate_results])/len(moderate_results)
    challenging_acc = sum([res for res in challenging_results])/len(challenging_results)
    all_acc = sum(results.values())/num_queries
    count_lists = [len(simple_results), len(moderate_results), len(challenging_results), num_queries]
    return simple_acc * 100, moderate_acc * 100, challenging_acc * 100, all_acc * 100, count_lists


def print_data(score_lists, count_lists):
    levels = ['simple', 'moderate', 'challenging', 'total']
    print("{:20} {:20} {:20} {:20} {:20}".format("", *levels))
    print("{:20} {:<20} {:<20} {:<20} {:<20}".format('count', *count_lists))

    print('======================================    ACCURACY    =====================================')
    print("{:20} {:<20.2f} {:<20.2f} {:<20.2f} {:<20.2f}".format('accuracy', *score_lists))


def custom_json_dumps(obj, indent=4, depth=0):
    if isinstance(obj, dict):
        opening_brace = "{\n" if depth > 0 else "{"
        closing_brace = "\n}" if depth > 0 else "}"
        items = [
            f'"{key}": {custom_json_dumps(value, indent, depth + 1)}'
            if isinstance(key, str) else f'{key}: {custom_json_dumps(value, indent, depth + 1)}' for key,
            value in obj.items()]
        return opening_brace + ",\n".join([(" " * indent * (depth + 1)) + item for item in items]) + closing_brace
    elif isinstance(obj, list):
        opening_bracket = "[\n" if depth > 0 else "["
        closing_bracket = "\n]" if depth > 0 else "]"
        items = [custom_json_dumps(item, indent, depth + 1) for item in obj]
        return opening_bracket + ",\n".join([(" " * indent * (depth + 1)) + item for item in items]) + closing_bracket
    elif isinstance(obj, str):
        return f'"{obj}"'
    else:
        return json.dumps(obj)


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--predicted_sql_path', type=str, required=True, default='')
    args_parser.add_argument('--ground_truth_path', type=str, required=True, default='')
    args_parser.add_argument('--data_mode', type=str, required=True, default='dev')
    args_parser.add_argument('--db_root_path', type=str, required=True, default='')
    args_parser.add_argument('--num_cpus', type=int, default=1)
    args_parser.add_argument('--meta_time_out', type=float, default=30.0)
    args_parser.add_argument('--mode_gt', type=str, default='gt')
    args_parser.add_argument('--mode_predict', type=str, default='gpt')
    args_parser.add_argument('--difficulty', type=str, default='simple')
    args_parser.add_argument('--diff_json_path', type=str, default='')
    args_parser.add_argument('--exec_result_file', type=str, default='')
    args = args_parser.parse_args()

    exec_result = []

    # generate gt sqls:
    gt_queries, db_paths_gt = package_sqls(args.ground_truth_path, args.db_root_path, mode='gt',
                                           data_mode=args.data_mode)

    pred_queries, db_paths = package_sqls(args.predicted_sql_path, args.db_root_path, mode=args.mode_predict,
                                          data_mode=args.data_mode, total_num=len(gt_queries))
    
    query_pairs = list(zip(pred_queries, gt_queries))
    run_sqls_parallel(query_pairs, db_places=db_paths, num_cpus=args.num_cpus, meta_time_out=args.meta_time_out)
    exec_result = sort_results(exec_result)

    print(f"save exec result to {args.exec_result_file}")
    if args.exec_result_file:
        json.dump(exec_result, open(args.exec_result_file, 'w'), indent=4)
        # with open(args.exec_result_file, 'w') as f:
        #     f.write(custom_json_dumps(exec_result, indent=4, depth=1))

    print('start calculate')
    simple_acc, moderate_acc, challenging_acc, acc, count_lists = \
        compute_acc_by_diff(exec_result, args.diff_json_path)
    score_lists = [simple_acc, moderate_acc, challenging_acc, acc]
    print_data(score_lists, count_lists)
    print('===========================================================================================')
    print("Finished evaluation")
