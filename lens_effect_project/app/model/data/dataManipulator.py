"""
###########################################
Esta es una implementación de un DataSet

autor: Ariel Cerón González
fecha: Febrero 4, 2023

Bajo la tutoria del Dr. Gibran Fuentes
IIMAS, UNAM
###########################################
"""
from torch.utils.data import Dataset, random_split
from pandas import read_csv
from PIL import Image
from os import listdir

class Lens(Dataset):
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
        super(Lens, self).__init__()
        
        self.metadatos = read_csv(root + f'/metadatos/{label}.csv') 
        self.length = self.metadatos.shape[0]
        self.transform = transform

    def __getitem__(self, idx) -> tuple:
        meta_img = self.metadatos.iloc[idx]['path']
        name_img = self.metadatos.iloc[idx]['name']
        img = Image.open(meta_img).convert('RGB')
        if img.verify():
            if self.transform:
                img = self.transform(img)
            return (img, name_img)
        return self.__getitem(idx + 1)

    def __len__(self) -> int:
        return self.length

class Lens2(Dataset):
    def __init__(self, path:str, transform=None) -> None:
        super(Lens2, self).__init__()
        self.path = path + '/'
        self.path_elements = listdir(path)
        self.length = len(self.path_elements)
        self.transform = transform

    def __getitem__(self, index) -> tuple:
        img = self.path_elements[index]
        path = self.path + img
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

    def __len__(self) -> int:
        return self.length
