# OAG

**Under Construction... The cleaned up code and the dataset will be released before August.**

## Requirements
- Linux
- python 3
- install requirements via ```
pip install -r requirements.txt``` 

## How to run
```bash
cd $project_path
export PYTHONPATH="$project_path:$PYTHONPATH"

# venue linking
python3 venue/run_all.py

# paper linking
### LSH method
python3 paper/Title2Vec.py  # train doc2vec model
python3 paper/Hash.py
### CNN method
python3 paper/CNN.py 

# author linking
python3 author/train.py
