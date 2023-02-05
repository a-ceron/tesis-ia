## --------------------------------------------
#
#	Tools for get galaxy images
#
#	aceron
#	IIMAS
#
## ---------------------------------------------
import requests
import pandas as pd
from os import listdir
# from sklearn.utils import Brunch
import random
import csv

def get_figure(url):
    return requests.get(url).content
	
def get_figures(save_path:str, n:int=100, save:bool=True, train=0, source=None):
    df = pd.read_csv(source, sep='\t', header=0)
    n_max = df.shape[0]
    if n > n_max:
       raise ValueError("{n} is grater than {n_max}")

    url = lambda a,b: f'https://www.legacysurvey.org/viewer/jpeg-cutout?ra={a}&dec={b}&size=128&layer=ls-dr9&pixscale=0.262&bands=grz'
    save_path = [save_path+'/train', save_path+'/test']

    if save:
        print("Dowloading...")
        if train > 0.:
            idx = list(df.index)
            idx_size = len(idx) - round(len(idx) * train)

            test_idx = set(random.choices(idx, k=idx_size))
            train_idx = list(set(idx).difference(test_idx))
            for i in test_idx:
                with open(f"{save_path[0]}/space_{i}.png", 'wb') as image:
                    image.write(get_figure(url(df['ra'][i],df['dec'][i])))
        for j in train_idx:
            with open(f"{save_path[1]}/space_{j}.png", 'wb') as image:
                image.write(get_figure(url(df['ra'][j],df['dec'][j])))

        print(f"Save figures at {save_path}")

    return df

def genera_metadatos(path:str, label:str):
    header = ['name', 'path'] # edit here pls just add
    with open(path+f'/metadatos/{label}.csv', mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        for image in listdir(path + f'/{label}'):
            row = [f'{image}', f'{path}/{label}/{image}'] # Edit here pls just add
            writer.writerow(row)
    

