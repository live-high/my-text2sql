import regex as re

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

def is_valid_exec_result(exec_result):
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