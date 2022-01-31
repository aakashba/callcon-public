# Guide for replication of results for paper "Function Call Graph Context Encoding for Neural Source Code Summarization" submitted for peer review currently
## Step 0 - Dataset building

We provide the compiled dataset as well as the scripts used to compile .The scripts to build call graph are in builders/ folder. The complete data and scripts can also be found at:

https://docs.google.com/uc?export=download&id=1PZovFibsZIca7Jc-8S1mIp6r08fOAxX7

## Step 1 - Training
To ensure no recursive errors or edits, create directories callcon/data and clone this git repository and  place the javastmt folders after decompresing the data

Create directory outdir, with 4 subdirectories  **outdir/{models, histories, viz, predictions}**
**Use Requirements.txt to get your python 3.x virtual environment in sync with our setup.** Venv is preferred. Common issues that might arise from updating an existing venv and solutions :
- GPU not recognized: checking the compatibility of your gpu cudnn/cuda or other drivers with the keras and tf versions fixes this.
- Tf unable to allocate tensor: upgrade to tf 2.4

To train the callcon-gru use the following command :
```
time python3 train.py --model-type=callcon-gru --batch-size=50 --epochs=10 --with-calls --gpu=0
```
Note: use --hops to change graph layer hops for models with graph layer. --with-graph is only used for the codegnngru model with an ast graph.

## Step 2 - Predictions
Training print screen will display the epoch at which the model converges, that is when the validation accuracy is not increase much or just before it starts to decrease and validation loss goes up. Once epoch is identified run the following script and replace file in this example with the trained model epoch and timestamp.

```
python3 predict.py path_to_model_epoch --with-calls --gpu=0
```
predicted comments for all models are provided in the predictions folder.

## Step 4 - Metrics
Bleu and Rouge scores as well a comparison script to insolate maximum improvement have been provided by the name of bleu.py, rougemetric.py and bleucompare.py all of them can be run with the similar commands
```
 python3 rougemetric.py path_to_predict_file
```
```
 python3 bleu.py path_to_predict_file
```
We have provided all the predicted comment files in predictions and ensemblepredictions directories.

