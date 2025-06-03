# DA-TCMPO
## Create your conda environment
You can create a conda environment by referencing the datcmpo.yml file. You also need to download the pre-trained model m3e-base (https://huggingface.co/moka-ai/m3e-base), which is used to generate embeddings for Chinese text.
## Create your datasets
To split ch.csv into training, validation, and test sets, please run the following code. Upon completion, the files train.pt, val.pt, and test.pt will be generated.
```
python create_data.py
```
## Train the baseline model
To train, validate, and test the baseline model (AE-TCMPO), please run the following code. Upon completion, the files base.model and baseline.txt will be generated. The baseline.txt file contains the performance metrics of base.model on the test set.
```
python baseline.py
```
## Train the DA-TCMPO
To train, validate, and test our model (DA-TCMPO), please run the following code. Upon completion, the files mo.model and full.txt will be generated. The full.txt file contains the performance metrics of mo.model on the test set.
```
python train.py
```
