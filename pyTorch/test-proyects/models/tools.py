## --------------------------------------------
#
#	Tools for get galaxy images
#
#	aceron
#	IIMAS
#
## ---------------------------------------------
import requests
from torchvision.datasets import ImageFolder

from pandas import read_csv
from os import listdir
# from sklearn.utils import Brunch
from PIL import Image
from models import const, tools
import random

def get_figure(url):
    return requests.get(url).content
	
def get_figures(save_path:str, n:int=100, save:bool=True, train=0, source=None):
    df = read_csv(source, sep='\t', header=0)
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

