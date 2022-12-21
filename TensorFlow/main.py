from models import VGG, ResNet

def VGG16():
    vgg = VGG.My_VGG16()

    vgg.model()
    vgg.train()    
    
def ResNet50():
    resnet = ResNet.My_ResNet()

    resnet.model()
    resnet.train()
    

def main():
   ResNet50()

if __name__ == '__main__':
    main()