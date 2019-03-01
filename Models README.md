
## Model Training

The models were trained using CNTK library and borrowing a leaf from the training provided by CNTK.ai well acknowledged in the report. However, the training provided did not support the epochs modification and the level of prediction accuracy did not in my program go above 22% and in the training above 48% though the epochs they supported were relative low - maximum 10. 

### Training a Model Using CIFAR-10 Dataset

The 60000 images were sourced from the CIFAR-10 data repository in Canada ferom their url from University of Toronto. The details is presented in the deep.py application which was used to have the environment including data repository setup prior to having the training started. Subsequently, the CNN module is my library, building on the training provided by CNTK.ai, from where most of the resources used in the training of each model was sourced. 

#### Training of Models in Phases

The first phase had just the script with all parameters hardcoded. The user could not decide for instance the number of epochs to run in each of these instances. The second phase allowed the user to decide the number of epochs to run and also the number of threads to deploy in each epoch's iteration. For instance if the user gave 250, this is not the number of threds actually but a divisor of the total number of size of the epoch, in this case 50000 for trainign and 10000 for testing. Therefore, with a parameter of 250, the number of threads to deploy would be 200 with 50000 as the size f the dataset or training examples we are using.

##### Phase 1


```python
python model_training_script.py #on linux or windows terminal OR
```


```python
%run model_training_script.py #on QtConsole terminal - an Anaconda platform for writing and or running Python Scripts
```

##### Phase 2


```python
python model_training_script.py --epochs num_integer --iter_size_div num_divides_epoch_size_to_whole_num #on linux or windows terminal OR
```


```python
%run model_training_script.py --iter_size_div num_divides_epoch_size_to_whole_num #on QtConsole terminal - an Anaconda platform for writing and or running Python Scripts
```

in the modle training phase one, the model names were also hard coded unlike in the variadic training phase where the name is determined at run time e.g. 50 epochs and iteration divisor of 250 for a vgg model would likely give a name containing vgg_50_250 and both phases ave to the cwd containg the training program. 

### Training Result Display

The result from the training as observed in the project and recorded in the report are in the training_display directory of both repositories - model_training and variadic_model_training
