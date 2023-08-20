#Co6mA

## Dependency
 
-python3

-tensorflow 1.15.0

-keras 2.3.1

-numpy 1.19.5

## Content

- dataset: data of Arabidopsis thaliana and Drosophila melanogaster for model training

- model: trained models

- data: the data sets collected from other methods

- code: the code of Co6mA and LA6mA

- result: the result of the combination of CBi6mA and LA6mA on the sub-Valid sheet of data sets

## Usage

### 1. train model

in code folder:

The script main_train.py in Co6mA folder used to train model CBi6mA. 

`file_name: the path of dataset`
`savepath: the save path of model`

The script main_train.py in AL_LA6mA folder used to train model LA6mA. 

`filename: the path of dataset`
`bestpath:  the save path of model`

This script ouput the trained model in the model folder. 

### 2. combine the trained models of CBi6mA and LA6mA

The script test.py in Co6mA folder used to combine the trained models of CBi6mA and LA6mA on the sub-Valid sheet of data sets. 

`file_name:  the path of dataset`
`filepath1: the trained model path of CBi6mA`
`filepath2: the trained model path of LA6mA`
`rst_file: the save path of result`

This script ouput the results in the result folder (obtain the best weight). 

### 3. predict 6mA-containing sequences

The script main_test.py in Co6mA folder is used to predict if a given sequence contain 6mA sites.

`file_name:  the path of dataset`
`filepath1: the trained model path of CBi6mA`
`filepath2: the trained model path of LA6mA`
`Change the weight parameter in line 133 to the weight value in result`

## Contact
If you are interested in our work or have any suggestions and questions about our research work, please feel free to contact us. E-mail: 
2112103059@zjut.edu.cn.
