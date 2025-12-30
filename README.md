# Python-Neural-Network
### Created by: Tristan Hertzog and Jaxon Powell

Demo video: 

A Deep Neural Network created with no imports or libraries with the exception of NumPy which was exclusively use for math logic. The intent behind this project is to prove and explore the inner workings of neural networks that implement regression and classification as well as minibatch

### Features:

* __Regression__: The basic linear or multivariable linear regression implementing forward and backward propagation and L2 loss for performance calculations
* __Classification__: Univariable, multivariable, and multivariate logistic regression similar implementation to regression implementing softmax and displaying the percentage of correctly identified classes as a performance metric
* __Gradient Descent__: Used in both regression and classification with a modifiable learning rate hyperparameter
* __MiniBatch__: Implementation of minibatch allowing for users to modify the batch size and report frequency
* __Arguments File__: A file allowing users to easily access and modify arguments including initalization range, activation functions, total updates, and more.
* __Make Data File__: A file for users who wish to create new data and test the NN on it. Has modifiable parameters to easily add wanted data.

### How to run:

**1.** Clone this repository\
**2.** Modify args.txt train/dev feature or targets to the correct wanted file name.\
**3.** In args.txt modify "Problem Mode" to match the directory in which you are using data from.\
**4.** Update "Output dimension" in args.txt to match the number of classes in classification or the output vector size. Both should be integers and info for dataset is given in data_info.txt under each data directory.\
**5.** Run in cloned directory:
    <pre>python main.py (Get-Content args.txt | Where-Object { $_ -notmatch "^//" -and $_ -ne "" })</pre>\
**6.** Tune for hyperparameters. Depending on data you may get overflow errors, these are not bugs or issues and NN will work properly depending on initialization matrix.\

### Creating new data:

**1.** After cloning the repository, locate make_data.py.\
**2.** Run in cloned directory:
    <pre>python make_data.py A1 A2 A3 A4 A5</pre>
    where the A1, ..., A5 are arguments that need to be modified. The description of which can be found at the top of make_data.py\
**3.** Locate newly found data in data file either under classification or reression.\
**4.** Locate newly created data information for args.txt in data_info.txt\
