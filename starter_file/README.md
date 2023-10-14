# Predicting annual income of neighborhoods of Madrid
This project aims to demonstrate the kwnoledge acquired throughout Udacity's AzureML Engineer Nanodegree.


## Project Set Up and Installation
All relevant files are found on the `starter_file` folder.

## Dataset

### Overview
For this project, we have selected an open dataset fetched from https://datos.madrid.es/portal/site/egob/menuitem.c05c1f754a33a9fbe4b2e4b284f1a5a0/?vgnextoid=71359583a773a510VgnVCM2000001f4a900aRCRD&vgnextchannel=374512b9ace9f310VgnVCM100000171f5a0aRCRD&vgnextfmt=default

It compiles socioeconomic information for all neighborhoods in Madrid. Information about families living in the neighborhood, their family members, their studies, the age of the buildings, number of schools, number of libraries, number of sport facilities, average age, number of kids, etcetera.

The target variable is the average annual income of a family given the characteristics of the neighborhood.

### Task
The aim of the project is to develop a machine learning model that will predict the annual income of the average home in the neighborhood. It is a regression task. 

### Access
The dataset was uploaded to the workspace, and consumed both in the notebook (for display purposes) as well as in the train.py script. In order to access the dataset, we need to give context to the python enviroment of which workspace to use.

## Automated ML
The AutoML configuration is standard, trying to use as much nodes as available in the compute cluster, and limiting the time available for the experiment. The rest of the configuration is specific for the regression task, indicating that the metric should be normalized RMSE, and using the compute cluster available for my analytics unit.

The AutoML runs correctly and we can check its progress thanks to the show_output=True parameter when submiting the job.
<imagen>

### Results
AutoML generated 29 pipelines, where we can check that the Stacking Ensemble is the best perfoming one.
It combines the performance of several weak learners to create a meta-model that is an ElasticNet trained on the outputs of the weak learners.
<imagen>

## Hyperparameter Tuning
I used a Decision Tree Regressor for predicting a continuous target variable. The reason for choosing a decision tree rather than other algorithms is explainability, and better performance with high cardinality datasets, such as the one we are using.

The training script is `train.py` which loads the dataset from the workspace and then cleans it (I already cleansed it before uploading, so the clean_data function just separates the target variable from the features.
It then splits the dataset into a train set and a test set, and fits a decision tree regressor using the hyperparameters provided in the arguments of the script. Then the script will finally log the R2 score.

<imagen>

We are letting hyperdrive sweep between combinations of three different hyperparameters of the decision tree:
* `max_depth`: This hyperparameter controls the overall complexity of the decision tree. It allows to get a trade-off between an underfitted and overfitted decision tree. We allowed for number between 5 and 25.
* `min_samples_leaf`: this hyperparameter allows to have leaves with a minimum number of samples and no further splits will be searched otherwise. We opted for 0.1% to 50% of the number of features.
* `min_samples_split`: similar to the previous hyperparameter, but the minimum number of samples is specified in each split instead of leaf. We opted for 0.1% to 50% of the number of features.

<imagen>

### Results
Unfortunately, Hyperdrive did not correctly run on either my company's AzureML suscription (due to some problem with docker containers), nor did on the Cloud lab provided by Udacity, where there seems to be an authentication error that seems to have happened to to expiry of a code.

I am thus unable to submit the best hyperparameter configuration from the hyperdrive run, nor I can register the best performing model.

<imagen>
<imagen>
<imagen>

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.
I deployed the model generated with AutoML on my company's AzureML suscription using the GUI. The endpoint is an HTTP REST API that is deployed using an Azure Container Instance (ACI).

In order to query the endpoint, we will need to perform a POST request to the endpoint URI using the `inference.py` script, that contains an example request.

## Screen Recording
[https://youtu.be/c8bgZIeZ8GM](https://youtu.be/c8bgZIeZ8GM)
