### Install
```bash
pip install .
```

### Setup database preprocessing based on CHESS
1. Run [CHESS](https://github.com/ShayanTalaei/CHESS?tab=readme-ov-file#preprocessing) to preprocess the databases.
2. Copy the preprocessed databases to `./data/dev_databases`


### Setup Bird eval data
1. Copy the eval data to `./data/dev/dev.json`
2. Copy the gold sql to `./data/dev/dev_gold.sql`


### Run pipeline
```bash
pipeline_type=basic
model=gpt-4o
temperature=1.0
bash run_main.sh  -p $pipeline_type -m $model [-t temperature]
```

### Evaluation
```bash
pipeline_type=basic
model=gpt-4o
bash run_evaluation.sh  -p $pipeline_type -m $model
```
