def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def CNN():
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    batch_size = 32
    K = len(classes)
    epochs = 5

    # Paso 1. Obtenemos los datos
    trainset = torchvision.datasets.CIFAR10(root='./data', 
                                        train=True,
                                        download=True)
    trainloader = torch.utils.data.DataLoader(trainset,                 
                                            batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data',   
                                        train=False,
                                        download=True)
    testloader = torch.utils.data.DataLoader(testset, 
                                            batch_size=batch_size,
                                            shuffle=False, num_workers=2)
    
    # Paso 2. Creamos el modelo
    model = models.MyCNN(K)

    print(model)
    # Paso 3. Entrenamos el modelo
    epocas = 5

    train_losses = [] # almacenamos la perdida al entrenar
    test_losses = []

    train_correct = []  # Almacenamos el númer de elementos correctamente identificados
    test_correct = []
    for epoca in range(epochs):
        trn_corr = 0
        tst_corr = 0
        for index, (X_train, y_train) in enumerate(trainloader):
            y_pred = model(X_train)
            loss = 
