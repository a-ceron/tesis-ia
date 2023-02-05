# Metadatos para la generación del dataset

Un metadato es un archivo en formato **csv** que estará almacenado en la ruta

`./metadatos`

y deberá llevar el nombre del conjunto de datos, por ejemplo *test, validation, train*

`./metadatos/train.csv`

El contenido del archivo será de la siguiente forma
```YAML
    name:
        Nombre de la imagen
    path:
        Ruta completa de la imagen
    others:
        Usted puede agregar tantos valores como desee
```
Se puede crear de forma automática editando el archvio

`./app/models/tools.py`

