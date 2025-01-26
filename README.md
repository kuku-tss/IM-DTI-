1.The key datasets utilized in this project include:

Davis BindingDB
BioSnap
BioSnap Protein
BioSnap Molecule
Dude
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
4.Command to Train the Model
Run the following command to train the model, ensuring to replace the path to your train.py file accordingly:

python train.py --run-id TestRun --config /path/to/your/train.py/config/default_config.yaml
5.Prediction Output Extraction
Upon successful training, you will need to extract the model's predictions using data_train.py and data_test.py. The outputs can be saved in .txt files for further analysis.

6.Command to Extract Predictions
Execute the following command, replacing the path to your train.py file as necessary:

python data_train.py --run-id TestRun --config /path/to/your/train.py/config/default_config.yaml
7.Integration of Outputs
Once you have the prediction outputs, utilize to_data.py to consolidate the training and output .txt files.

8.Random Forest Training
Following the extraction of predictions, you can proceed to train a Random Forest model using beiyes.py. For generating the final predictions, you may use either b_IM.py or predict_pred.py, where you can adjust the evaluation metrics as per your requirements.
