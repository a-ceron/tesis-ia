from utils import tools
from models import VGG
def VGG16():
    vgg = VGG.VGG16()

    vgg.model()
    vgg.train()    
    


def main():
    VGG16()

if __name__ == '__main__':
    main()