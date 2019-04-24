## How to run
```bash
cd $project_path
export PYTHONPATH="$project_path:$PYTHONPATH"
### LSH method
python3 paper/Title2Vec.py  # train doc2vec model
python3 paper/Hash.py
### CNN method
python3 paper/CNN.py  # In CNN.py, mcnn.train(0): training CNN model; mcnn.evaluate(0): evaluate CNN model.
