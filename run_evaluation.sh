db_root_path='./data/dev_databases/'
data_mode='dev'
diff_json_path='./data/dev/dev.json'
predicted_sql_path='./exp_results/'
ground_truth_path='./data/dev/'
num_cpus=8
meta_time_out=30.0
mode_gt='gt'
mode_predict='gpt'


while getopts ":p:m:" opt; do
  case $opt in
    p) pipeline_type="$OPTARG"
    ;;
    m) model="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

predicted_sql_path="${predicted_sql_path}/${pipeline_type}/${model}/"
exec_result_file="${predicted_sql_path}/exec_result.json"

if [[ ! -d $predicted_sql_path ]]; then
    mkdir -p $predicted_sql_path
fi

python post_process.py \
--result_file pred_outputs/${pipeline_type}/${model}_pred_sql.json \
--output_file $predicted_sql_path/predict_dev.json


echo '''starting to compare with knowledge for ex'''
python3 -u ./evaluation.py --db_root_path ${db_root_path} --predicted_sql_path ${predicted_sql_path} --data_mode ${data_mode} \
--ground_truth_path ${ground_truth_path} --num_cpus ${num_cpus} --mode_gt ${mode_gt} --mode_predict ${mode_predict} \
--diff_json_path ${diff_json_path} --meta_time_out ${meta_time_out} \
--exec_result_file ${exec_result_file}
