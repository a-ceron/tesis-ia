"""
###########################################
Este archivo contiene una implementación del 
dataset para poder utilizar las imagenes que 
se obtienen por una solicitud a la API de 
los imágenes astronómicas.

autor: Ariel Cerón González
fecha: Febrero 4, 2023

Bajo la tutoria del Dr. Gibran Fuentes
IIMAS, UNAM
###########################################
"""
from models.tools import myTools

from torch.utils.data import Dataset
from PIL import Image

class Dataset_Lens(Dataset):
    """
        Obtiene un dataset de imágenes dada una ruta

        Se espera que el conjunto de datos este contenida
        en un directorio cuya estructura sea 
        root:
            directorio:
                metadatos:
                    - archivos de identificación csv
                imagenes:
                    entrenamiento:
                        - imagenes para el entrenamiento
                    validación:
                        - imágenes para la validación
                    prueba:
                        - imagénes para la prueba
        la función buscará en el directorio las imagenes
        y los métadatos para la etiqueta dada

        In:
            root: str
                Nos indiaca la ruta inicial donde se encuentran los datos
            label: str
                Nos indica el tipo de archivos [entrenamiento, validación, prueba]
            transform: 
                Objeto de transformacioes para los datos
        Out:
            dataset: Dataset
                Regresa un objeto dataset

    """
    def __init__(self, root:str, label:str, transform=None) -> None:
        super(Dataset_Lens, self).__init__()
        
        self.metadatos = myTools.pd.read_csv(root + f'/metadatos/{label}.csv') 
        self.length = self.metadatos.shape[0]
        self.transform = transform

    def __getitem__(self, idx) -> tuple:
        meta_img = self.metadatos.iloc[idx]['path']
        name_img = self.metadatos.iloc[idx]['name']
        img = Image.open(meta_img).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return (img, name_img)

    def __len__(self) -> int:
        return self.length
