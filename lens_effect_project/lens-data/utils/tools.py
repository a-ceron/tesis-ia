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


DATA_PATH = '../'
DATA_DIR = 'figures'
SOURCE = 'training_lens.tsv'

def get_figure(url):
	return requests.get(url).content
	
def get_figures(n:int=100, save:bool=True):
	"""Generate a data directory and download n figures"""
	url = lambda a,b: f'https://www.legacysurvey.org/viewer/jpeg-cutout?ra={a}&dec={b}&size=128&layer=ls-dr9&pixscale=0.262&bands=grz'
	df = read_csv(SOURCE, sep='\t', header=0)

	if save:
		print("Dowloading...")
		for i in range(n):
			with open(f"{DATA_PATH+DATA_DIR}/space_{i}.png", 'wb') as image:
				image.write(get_figure(url(df['ra'][i],df['dec'][i])))
		print(f"Figures save at: {DATA_PATH}")

	return df



class GalaxyLens:
	"""Dataset of my figures"""
	def __init__(self):
		self.paths = [DATA_PATH + DATA_DIR + '/' + name for name in listdir(DATA_PATH+DATA_DIR)]

	def __getitem__(self, i):
		return Image.open(self.path[i])
		
	def __len__(self):
		return len(self.paths)


dataset = GalaxyLens()
print(len(dataset))

df = get_figures(save=False)
print(df.head())

b = ImageFolder(DATA_PATH)
print(b)
