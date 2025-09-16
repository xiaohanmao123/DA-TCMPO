# DA-TCMPO
## Create your conda environment
You can create a conda environment by referencing the datcmpo.yml file. You also need to download the pre-trained model m3e-base (https://huggingface.co/moka-ai/m3e-base), which is used to generate embeddings for Chinese text.
## Create your datasets
To split ch.csv into training, validation, and test sets, please run the following code. Upon completion, the files train.pt, val.pt, and test.pt will be generated.The provided ch.csv is a simplified version of the CH dataset, intended solely for reproducing the core functionality of our model. For access to the complete dataset used in our study, please contact the corresponding author. We are happy to share it for academic and non-commercial research purposes upon reasonable request.
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
## Predict
Before running prediction, please make sure you already have a trained model file (mo.model). Without it, prediction cannot proceed.
You also need to prepare a CSV file that contains the prescriptions to be optimized and their corresponding indications.
We provide an example file predict.csv with the following format:(1)components: Herbs in the prescription that will be kept unchanged;(2)label: The herb that needs to be replaced;(3)function_description: Indications of the prescription.
You can generate the dataset predict.pt from predict.csv by running:
```
python create_predict.py
```
This will preprocess the input data and save it into predict.pt, which will then be used for prediction.<br>
After obtaining predict.pt, you can generate results by running:
```bash
python predict.py
```
The predicted outputs will be saved in predictions2.csv, which contains the following key columns:(1)PredNames: The herb recommended by the model;(2)LabelNames: The herb in the original prescription that is being replaced
