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

```
Current status: FeaturesGeneration. Generating features for the dataset.
Current status: DatasetCrossValidationSplit. Generating individually featurized CV splits.
Current status: ModelSelection. Beginning model selection.

********************************************************************************************
DATA GUARDRAILS: 

TYPE:         Missing feature values imputation
STATUS:       PASSED
DESCRIPTION:  No feature missing values were detected in the training data.
              Learn more about missing value imputation: https://aka.ms/AutomatedMLFeaturization

********************************************************************************************

TYPE:         High cardinality feature detection
STATUS:       PASSED
DESCRIPTION:  Your inputs were analyzed, and no high cardinality features were detected.
              Learn more about high cardinality feature handling: https://aka.ms/AutomatedMLFeaturization

********************************************************************************************

********************************************************************************************
ITER: The iteration being evaluated.
PIPELINE: A summary description of the pipeline being evaluated.
DURATION: Time taken for the current iteration.
METRIC: The result of computing score on the fitted pipeline.
BEST: The best observed score thus far.
********************************************************************************************

 ITER   PIPELINE                                       DURATION            METRIC      BEST
    0   MaxAbsScaler LightGBM                          0:00:23             0.0778    0.0778
    3   StandardScalerWrapper ElasticNet               0:00:22             0.2518    0.0778
    6   MaxAbsScaler ExtremeRandomTrees                0:00:21             0.0677    0.0677
    7   StandardScalerWrapper ElasticNet               0:00:20             0.1316    0.0677
    9   StandardScalerWrapper ElasticNet               0:00:24             0.1207    0.0677
    1   MaxAbsScaler XGBoostRegressor                  0:00:23             0.0724    0.0677
   10   MaxAbsScaler RandomForest                      0:00:22             0.0753    0.0677
   12   StandardScalerWrapper ElasticNet               0:00:24             0.1482    0.0677
    4   MaxAbsScaler ElasticNet                        0:00:23             0.1705    0.0677
   11   StandardScalerWrapper ElasticNet               0:00:23             0.1316    0.0677
   13   StandardScalerWrapper ElasticNet               0:00:24             0.1316    0.0677
   14   StandardScalerWrapper ElasticNet               0:00:23             0.0876    0.0677
   15   StandardScalerWrapper ElasticNet               0:00:23             0.1438    0.0677
   16   StandardScalerWrapper DecisionTree             0:00:19             0.1115    0.0677
    2   MaxAbsScaler ElasticNet                        0:00:22             0.1547    0.0677
    5   MaxAbsScaler ElasticNet                        0:00:23             0.2596    0.0677
    8   StandardScalerWrapper ElasticNet               0:00:21             0.2304    0.0677
   17   StandardScalerWrapper ElasticNet               0:00:18             0.1294    0.0677
   18   MaxAbsScaler DecisionTree                      0:00:22             0.1082    0.0677
   19   StandardScalerWrapper ExtremeRandomTrees       0:00:24             0.0714    0.0677
   23   StandardScalerWrapper RandomForest             0:02:04             0.0820    0.0677
   26   MaxAbsScaler GradientBoosting                  0:02:04             0.0630    0.0630
   27   MaxAbsScaler RandomForest                      0:02:04             0.1032    0.0630
   20   StandardScalerWrapper ExtremeRandomTrees       0:02:04             0.0683    0.0630
   21   StandardScalerWrapper ExtremeRandomTrees       0:02:04             0.0683    0.0630
   25   MaxAbsScaler ExtremeRandomTrees                0:02:05             0.0662    0.0630
   22   MaxAbsScaler ExtremeRandomTrees                0:02:05             0.0723    0.0630
   24   MaxAbsScaler ExtremeRandomTrees                0:02:04             0.0714    0.0630
   28    VotingEnsemble                                0:02:05             0.0584    0.0584
   29    StackEnsemble                                 0:02:04             0.0581    0.0581
```

### Results
AutoML generated 29 pipelines, where we can check that the Stacking Ensemble is the best perfoming one.
It combines the performance of several weak learners to create a meta-model that is an ElasticNet trained on the outputs of the weak learners.
![Alt text](https://github.com/dleston/nd00333-capstone/blob/master/starter_file/screenshots/best_automl_model.png?raw=true)

## Hyperparameter Tuning
I used a Decision Tree Regressor for predicting a continuous target variable. The reason for choosing a decision tree rather than other algorithms is explainability, and better performance with high cardinality datasets, such as the one we are using.

The training script is `train.py` which loads the dataset from the workspace and then cleans it (I already cleansed it before uploading, so the clean_data function just separates the target variable from the features.
It then splits the dataset into a train set and a test set, and fits a decision tree regressor using the hyperparameters provided in the arguments of the script. Then the script will finally log the R2 score.

```
def clean_data(data):
    # all data is numeric so there is no need to one-hot-encode any variable.
    x_df = data.to_pandas_dataframe()
    y_df = x_df.pop("Renta neta media anual de los hogares (Urban Audit)")
    return x_df, y_df


def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--max_depth', type=int, default=20,
                        help="The maximum depth of the tree.")
    parser.add_argument('--min_samples_split', type=float, default=0.1,
                        help="The minimum number of samples required to split an internal node (fraction of total samples)")
    parser.add_argument('--min_samples_leaf', type=float, default=0.1,
                        help="The minimum number of samples required to be at a leaf node (fraction of total samples)")

    args = parser.parse_args()

    run = Run.get_context()

    run.log("Maximum Depth:", np.int(args.max_depth))
    run.log("Minimum number of samples per split (% of samples):", np.round(args.min_samples_split,2))
    run.log("Minimum number of samples per leaf node (% of samples):", np.round(args.min_samples_leaf,2))


    subscription_id = '9a8ef160-b36c-4d1c-95d2-b381d53baaa3'
    resource_group = 'rg-bigdatanetworks-uad-pro'
    workspace_name = 'aml-BigDataNetworksuad-pro'

    ws = Workspace(subscription_id, resource_group, workspace_name)
    dataset = Dataset.get_by_name(ws, name='panel_indicadores_distritos_barrios_2022')    
    x, y = clean_data(dataset)

    # TODO: Split data into train and test sets.

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=33)

    model = DecisionTreeRegressor(max_depth=args.max_depth, min_samples_split=args.min_samples_split, min_samples_leaf=args.min_samples_leaf).fit(x_train, y_train)

    r2 = model.score(x_test, y_test)
    run.log("r2", np.float(r2))

    joblib.dump(model, 'outputs/model.pkl')

if __name__ == '__main__':
    main()
```
We are letting hyperdrive sweep between combinations of three different hyperparameters of the decision tree:
* `max_depth`: This hyperparameter controls the overall complexity of the decision tree. It allows to get a trade-off between an underfitted and overfitted decision tree. We allowed for number between 5 and 25.
* `min_samples_leaf`: this hyperparameter allows to have leaves with a minimum number of samples and no further splits will be searched otherwise. We opted for 0.1% to 50% of the number of features.
* `min_samples_split`: similar to the previous hyperparameter, but the minimum number of samples is specified in each split instead of leaf. We opted for 0.1% to 50% of the number of features.

### Results
Unfortunately, Hyperdrive did not correctly run on either my company's AzureML suscription (due to some problem with docker containers), nor did on the Cloud lab provided by Udacity, where there seems to be an authentication error that seems to have happened to to expiry of a code.

I am thus unable to submit the best hyperparameter configuration from the hyperdrive run, nor I can register the best performing model.

On my company's AzureML suscription:
![Alt text](https://github.com/dleston/nd00333-capstone/blob/master/starter_file/screenshots/hyperdrive_failed_company_aml.png?raw=true)
And on Udacity's Cloudlab:
![Alt text](https://github.com/dleston/nd00333-capstone/blob/master/starter_file/screenshots/hyperdrive_failed_udacity_cloudlab_1.png?raw=true)
![Alt text](https://github.com/dleston/nd00333-capstone/blob/master/starter_file/screenshots/hyperdrive_failed_udacity_cloudlab_3.png?raw=true)

## Model Deployment
I deployed the model generated with AutoML on my company's AzureML suscription using the GUI. The endpoint is an HTTP REST API that is deployed using an Azure Container Instance (ACI).
![Alt text](https://github.com/dleston/nd00333-capstone/blob/master/starter_file/screenshots/endpoint_active.png?raw=true)
In order to query the endpoint, we will need to perform a POST request to the endpoint URI using the `inference.py` script, that contains an example request.

```
import urllib.request
import json
import os
import ssl

def allowSelfSignedHttps(allowed):
    # bypass the server certificate verification on client side
    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context

allowSelfSignedHttps(True) # this line is needed if you use self-signed certificate in your scoring service.

# Request data goes here
# The example below assumes JSON formatting which may be updated
# depending on the format your endpoint expects.
# More information can be found here:
# https://docs.microsoft.com/azure/machine-learning/how-to-deploy-advanced-entry-script
data =  {
  "Inputs": {
    "data": [
      {
        "Barrio": 5,
        "Año medio de contrucción de inmuebles de uso residencial": 1999,
        "Apartamentos Municipales para Mayores": 1,
        "Asociaciones (Sección 1ª)": 2,
        "Asociaciones culturales y casas regionales": 0,
        "Asociaciones de mujeres": 0,
        "Asociaciones vecinales": 1,
        "Bibliotecas Comunidad Madrid": 1,
        "Bibliotecas Municipales": 1,
        "Campos de fútbol 11": 1,
        "Centro de Día de Atención a Niños y Niñas (de 3 a 12 años)": 0,
        "Centros de Adolescentes y Jóvenes (ASPA)": 0,
        "Centros de Apoyo a las Familias (CAF)": 0,
        "Centros de Atención a la Infancia (CAI)": 0,
        "Centros de Atención a las Adicciones (CAD y CCAD)": 0,
        "Centros de Día de Alzheimer y Físicos": 0,
        "Centros de Servicios Sociales": 0,
        "Centros deportivos Municipales": 2,
        "Centros Municipales de Mayores": 0,
        "Centros Municipales de Salud Comunitaria (CMSC)": 0,
        "Centros para personas sin hogar": 0,
        "Centros y Espacios Culturales": 0,
        "Colegios Públicos Infantil y Primaria": 5,
        "Duración media del crédito (meses) en transacción de vivienda": 200,
        "Edad media de la población": 48,
        "Escuelas Infantiles Municipales": 5,
        "Espacios de Igualdad": 0,
        "Espacios de Ocio para Adolescentes (El Enredadero)": 0,
        "Etapas educativas. Total niñas": 400,
        "Etapas educativas. Total niños": 400,
        "Fundaciones (Sección 2ª)": 2,
        "Hogares con un hombre solo mayor de 65 años": 200,
        "Hogares con una mujer sola mayor de 65 años": 200,
        "Hogares monoparentales: un hombre adulto con uno o más menores": 0,
        "Hogares monoparentales: una mujer adulta con uno o más menores": 0,
        "Índice de dependencia (Población de 0-15 + población 65 años y más / Pob. 16-64)": 40,
        "Instalaciones deportivas básicas": 1,
        "Mercados Municipales": 1,
        "Número de inmuebles de uso residencial": 1500,
        "Número Habitantes": 3000,
        "Paro registrado (número de personas registradas en SEPE en Febrero 2022)": 5,
        "Paro registrado (número de personas registradas en SEPE en Febrero 2022) Hombres": 5,
        "Paro registrado (número de personas registradas en SEPE en Febrero 2022) Mujeres": 5,
        "Pensión media mensual  Mujeres": 1800,
        "Pensión media mensual Hombres": 2100,
        "Personas con nacionalidad española": 2800,
        "Personas con nacionalidad española Hombres": 1400,
        "Personas con nacionalidad española Mujeres": 1400,
        "Personas con nacionalidad extranjera": 200,
        "Personas con nacionalidad extranjera Hombres": 100,
        "Personas con nacionalidad extranjera Mujeres": 100,
        "Piscinas cubiertas": 1,
        "Piscinas de verano": 1,
        "Pista de atletismo": 0,
        "Población de 0 a 14 años": 800,
        "Población de 15 a 29 años": 700,
        "Población de 30 a 44  años": 500,
        "Población de 45 a 64 años": 200,
        "Población de 65 a 79 años":200,
        "Población de 65 años y más": 200,
        "Población de 80 años y más": 400,
        "Población densidad (hab./Ha.)": 300,
        "Población en etapa educativa (Población de 3 a 16 años -16 no incluidos)": 0,
        "Población en etapa educativa de 0 a 2 años": 200,
        "Población en etapa educativa de 12 a 15 años": 200,
        "Población en etapa educativa de 3 a 5 años": 200,
        "Población en etapa educativa de 6 a 11 años": 200,
        "Población en etapas educativas": 800,
        "Población Hombres": 1500,
        "Población infantil femenina en etapa educativa de 0 a 2 años": 100,
        "Población infantil femenina en etapa educativa de 12 a 15 años": 100,
        "Población infantil femenina en etapa educativa de 3 a 5 años": 100,
        "Población infantil femenina en etapa educativa de 6 a 11 años": 100,
        "Población infantil masculina en etapa educativa de 0 a 2 años": 100,
        "Población infantil masculina en etapa educativa de 12 a 15 años": 100,
        "Población infantil masculina en etapa educativa de 3 a 5 años": 100,
        "Población infantil masculina en etapa educativa de 6 a 11 años": 100,
        "Población mayor/igual  de 25 años  con estudios superiores, licenciatura, arquitectura, ingeniería sup., estudios sup. no universitarios, doctorado,  postgraduado": 80,
        "Población mayor/igual  de 25 años  que no sabe leer ni escribir o sin estudios": 0,
        "Población mayor/igual  de 25 años con Bachiller Elemental, Graduado Escolar, ESO, Formación profesional 1º grado": 5,
        "Población mayor/igual  de 25 años con enseñanza primaria incompleta": 0,
        "Población mayor/igual  de 25 años con Formación profesional 2º grado, Bachiller Superior o BUP": 5,
        "Población mayor/igual  de 25 años con Nivel de estudios desconocido y/o no consta": 0,
        "Población mayor/igual  de 25 años con titulación media, diplomatura, arquitectura o ingeniería técnica": 10,
        "Población Mujeres": 1500,
        "Proporción de envejecimiento (Población mayor de 65 años/Población total)": 20,
        "Proporción de juventud (Población de 0-15 años/Población total) porcentaje": 20,
        "Proporción de personas migrantes (Población extranjera menos UE y resto países de OCDE / Población total)": 5,
        "Proporción de sobre-envejecimiento (Población mayor de 80 años/ Población mayor de 65 años)": 15,
        "Residencias para personas Mayores": 2,
        "Superficie (Ha.)": 150,
        "Superficie media de la vivienda (m2) en transacción": 120,
        "Tamaño medio del hogar": 3.22,
        "Tasa absoluta de paro registrado (Febrero  2022)": 3,
        "Tasa absoluta de paro registrado Hombres": 3,
        "Tasa absoluta de paro registrado Mujeres": 4,
        "Tasa bruta de natalidad (‰)": 12.0,
        "Tasa de crecimiento demográfico (porcentaje)": 10,
        "Tasa de desempleo en hombres de 16 a 24 años": 9.4,
        "Tasa de desempleo en hombres de 25 a 44 años": 4.4,
        "Tasa de desempleo en hombres de 45 a 64 años": 7.1,
        "Tasa de desempleo en mujeres de 16 a 24 años": 9.5,
        "Tasa de desempleo en mujeres de 25 a 44 años": 5.2,
        "Tasa de desempleo en mujeres de 45 a 64 años": 7.3,
        "Total hogares": 3500
      }
    ]
  },
  "GlobalParameters": 0.0
}

body = str.encode(json.dumps(data))

url = 'http://77c2d7fe-72d3-44d7-8be8-2b9ad0486b75.westeurope.azurecontainer.io/score'
# Replace this with the primary/secondary key or AMLToken for the endpoint
api_key = 'uEG6CQYUeQic60N2XMRCXQ4jgcSxLNpV'
if not api_key:
    raise Exception("A key should be provided to invoke the endpoint")


headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}

req = urllib.request.Request(url, body, headers)

try:
    response = urllib.request.urlopen(req)

    result = response.read()
    print(result)
except urllib.error.HTTPError as error:
    print("The request failed with status code: " + str(error.code))

    # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
    print(error.info())
    print(error.read().decode("utf8", 'ignore'))
```


## Screen Recording
[https://youtu.be/c8bgZIeZ8GM](https://youtu.be/c8bgZIeZ8GM)
