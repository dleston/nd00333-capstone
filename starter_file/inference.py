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