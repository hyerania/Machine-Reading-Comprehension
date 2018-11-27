# Machine Reading Comprehension
Machine Reading Comprehension exploration on the SQuAD dataset in order to analyze the performance of different features and selecting the best model based on F1 and EM scores. The network consisted of an embedding layer, encoding layer, attention layer, and output layer. Further details on these layers can be found on our report

### Code Structure
* ```main.py```: Provides a number of configuration parameters in order to train different MRC models
* ```mrcModel.py```: Network setup with all the layers, the training loop, and the evaluation metrics
* ```layers.py```: Definitions of each of the layers
	* Contains LSTM and GRU settings, which are manually changed depending on what you are testing
* ```preprocessing.py```: Separating the dataset into context, questions, answers, and answer spans in their respective files
* ```offical_evaluation.py```: Evaluation script directly provided from the SQuAD dataset
* ```batcher.py```: Creates a batch object
* ```embedding.py```: Processes the GLoVe vector file and provides the embedding matrix
### Running the Code
* Install the required packages in a virtual or conda environment:
	* Python 3.6
	* Tensorflow 1.12
	* TQDM
	* NLTK
* Create a ```train/``` directory in order to save the log file with the results from the network
* To run the network, there are a number of parameters that can be used:
	* mode: Train in order to train the model with analysis on test set, default only initializes model
	* spanMode: True for SmartSpan, False otherwise
	* Highway: True for addition of highway layers, False otherwise
	* Bidaf: True for the use of BiDAF attention, False for dot product attention
	* CharCNN: True for the use character embedding, False for GLoVe vectors
* Based on our best model, the ```layers.py``` file is configured for LSTM and the following command can be run:
```
python main.py --mode=train --spanMode=True --Highway=False --Bidaf=True --CharCNN=False
```
* Once the model runs, your ```dataset/``` directory will contain the files that were preprocessed from the SQuAD dataset and ```train/``` directory will contain all the results from the network.
