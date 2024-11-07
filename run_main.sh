
## run pipeline

export OPENAI_API_KEY=""

export OPENAI_API_BASE="http://v2.open.venus.oa.com/llmproxy/"
export DB_ROOT_PATH="./data"

eval_path="./data/dev/dev.json"

while getopts ":p:m:t:" opt; do
  case $opt in
    p) pipeline_type="$OPTARG"
    ;;
    m) model="$OPTARG"
    ;;
    t) temperature="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

output_path="./pred_outputs/${pipeline_type}"
template_root_path="./sqlagent/templates"

if [[ temperature -eq 0 ]]; then
    temperature=1.0
fi

if [[ $pipeline_type != "basic" && $pipeline_type != "reflexion" ]]; then
   echo "pipeline_type is not basic or reflexion: $pipeline_type"
    if [[ $pipeline_type == "dca" || $pipeline_type == "qp" || $pipeline_type == "os" || $pipeline_type == "dc_fix" ]]; then
        if [[ ! -e "${template_root_path}/${pipeline_type}" ]]; then
            ln -s "chase" "${template_root_path}/${pipeline_type}"
        fi
    fi
    template_root_path="${template_root_path}/${pipeline_type}"
fi

if [ ! -d $output_path ]; then
    mkdir $output_path
fi


export TEMPLATES_ROOT_PATH=$template_root_path

python main.py \
--pipeline_type $pipeline_type \
--eval_path $eval_path \
--output_path $output_path \
--model $model \
--temperature $temperature
