# rclc_2019_baseline
Code for RCLC baseline experiment. This repo contains the code to download dataset and publication information from open access uri. Baseline experiment notebook can be seen at `src/rclc_2019_entity_indicative_naive_bayes_baseline.ipynb`.


## Quick Start

1. Get `corpus.jsonld` from [rclc repo](https://github.com/Coleridge-Initiative/rclc). Put it under `data/`.
2. We are using conda environment. Before we start, create conda environment from `environment.yml`
```
conda env create -f environment.yml
conda activate rclc_2019_baseline
```
3. Go to `src/`
4. Run `python download_corpus_resources.py` to download all resources, i.e. pdf files for publications, html files for datasets. Ensure that all pdf files are downloaded successfully.
```
ls ../data/resource/pubs/pdf/ | wc -l
```
5. Extract pdf meta data
```
mkdir ../data/resource/pubs/json/
java -Xmx6g -jar ../tools/science-parse-cli-assembly-2.0.2-SNAPSHOT.jar -o ../data/resource/pubs/json/ ../data/resource/pubs/pdf/
```
6. Convert pdf to text files
```
mkdir ../data/resource/pubs/text
python convert_pdf2text.py
```
7. Run baseline experiment notebook `rclc_2019_entity_indicative_naive_bayes_baseline.ipynb`
