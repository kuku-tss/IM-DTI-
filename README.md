1.The key datasets utilized in this project include:

Davis BindingDB
BioSnap
BioSnap Protein
BioSnap Molecule
Dude
The value of model_architecture in the default_config.yaml for these datasets is SimpleCoembeddingNoSigmoid; the value for the TDC dataset is SimpleCoembeddingResnet.
Models
2.We employ the following advanced models for training:

ProBERT
ESM-1b
ProtT5-XL UniRef50
Training
3.To initiate the training process, you can utilize the train.py script. Configuration parameters can be customized through config.yaml. The specific training epochs for each model are as follows:

ProBERT: 50 epochs
ESM-1b: 10 epochs
ProtT5-XL UniRef50: 30 epochs
ProBERT: 50 epochs ESM-1b: 10 epochs ProtT5-XL UniRef50: 30 epochs 
4. Command to train the model Run the following command to train the model, which will give you the trained pt model file. Of course, before running, you need to set the epoch and ensure that the target_featurizer parameter you pass is the large model you need, accordingly replace the path of your train.py file: python train.py --run-id TestRun --config /path/to/your/train.py/config/default_config.yaml 
5. Extracting prediction output After successful training, you need to use data_train.py and data_test.py to extract the predictions of the model. These two files use the trained pt file to predict and obtain its output for both the training set and the test set. All three large models trained as extractors need to go through these two files. The output can be saved as a .txt file for further analysis. 
6. Command to extract predictions Execute the following command, replacing the path of your train.py file as needed: python data_train.py --run-id TestRun --config /path/to/your/train.py/config/default_config.yaml 
7. Integration of outputs Once you have the prediction outputs, use to_data.py to integrate the training and output .txt files. 
8. Random forest training After extracting the predictions, you can start training a random forest model using beiyes.py.
9. After training the random forest, you can use the pridict_pred.py file to predict the final output. Welcome to use it, though the process may be a bit cumbersome, we will gradually improve it.
There are some paths in the code that need to be changed to your own.
In the data loading section, we also shuffled the test set, which is not necessary; you can remove it.
