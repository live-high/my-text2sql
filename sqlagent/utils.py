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