Test assignment for interactivestandard

Installation: 
~~~
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision numpy Pillow transformers tqdm pathlib annoy scipy click marshmallow-dataclass pandas scikit-learn
~~~
Usage:
~~~
!!! Before usage, check the configs, especially device

python3 src/download_dataset.py configs/downloading_params.yaml          //download dataset, another option is to put unzipped test-task to data/raw/ (in case you don't have wget in your os)
python3 src/clean_dataset.py configs/cleaning_params.yaml                //dataset cleansing
python3 src/predict_pipeline.py configs/inference_params.yaml            //inference by itself 

~~~
Results:
~~~
{"homogenity_score": 0.9671600270041537, "completeness_score": 0.9214733343408211, "v_measure_score": 0.9437640922428762}
~~~
Scheme:
~~~

├── configs            		<- .yaml files for configuration
├── README.md          		<- The top-level README for developers using this project.
├── data
│   ├── processed      		<- The final, canonical data sets for modeling.
│   └── raw            		<- The original, immutable data dump.
│
├── logs                        <- Logs
│
├── models             		<- Trained and serialized models, model predictions, or model summaries
│
├── notebooks                   <- Jupyter notebook to get the idea
│
├── results                     <- Clusterized images
│
├── requirements.txt   		<- The requirements file for reproducing the analysis environment, e.g. generated with `pip freeze > requirements.txt`
│                         		
├── src                		<- Source code for use in this project.
│   ├── __init__.py    		<- Makes src a Python module
│   │
│   ├── entities       		<- dataclasses for configs
│   │
│   ├── features       		<- code to turn raw data into features for modeling
│   │
│   ├── utils          		<- utils for saving and measuring of quality
│   │
│   ├── download_dataset.py     <- python script for downloading from yandex cloud
│   │
│   ├── clean_dataset.py       	<- python script for cleaning data
│   │
│   ├── predict_pipeline.py     <- main pipeline for prediction


