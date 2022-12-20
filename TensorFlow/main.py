from models import VGG

def VGG16():
    vgg = VGG.My_VGG16()

    vgg.model()
    vgg.train()    
    


def main():
    VGG16()

if __name__ == '__main__':
    main()