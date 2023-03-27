# Unsipervised Representation Learning With Deep Convolutional Generative Adversarial Networks. Una revisión en español.

## Resumen
(2016) Para este años, el aprendizaje supervisado con redes convolucionales (CNNs) se habian adoptado para aplicaciones de visión por computadora. De forma comparada, el aprendizaje not supervisado con CNNs ha tenido menos atención. El artículo pretende introducir una clase de CNNs llamada Red Generativa adversaria de convolución profunda o Deep Convolutional Generative Adversarial Networks, DCGANs por sus siglas en ingles. Entrenado en varios conjuntos de imagenes se muestra evidencia convincente de que aprende una gerarqui de representaciones de objetos part to scenes in both the generator and discriminator.

## 1. Introducción
El artículo propone una forma de contruir una buena representación de images al entrenar Redes Adversarial Generarivas (GANs) y después reusar las partes de la redes generadores y el discriminadores como features para tareas supervisadas.  El artículo hace las siguientes contribuciones:
- We propose and evaluate a set of constraints on the architecture topology of Convolutional GANs that make them stable to train in most settings. We name this class of architectures Deep Convolutional GANS (DCGAN)
- We use the trained discriminators for image classification tasks, showing competitive permormance with other unsupervised algorithms. 
- We visualize the filters learnt by GANs and empirically show that specific filterls have learned to draw specific objects.
- We show that the generators have intersting vector arithmetiv properties allowing gor easu manipulation of many semantic qualities of generated samples.

## 3. Aproximación y arquitectura de modelo.
Existen dificultades al convertir una rarquitecturaed GAN fully conected a una arquitectura GAN convolucional, metodo usual al modificar una arquitectura. Sin embargo hasta este punto existen algunos procedimientos que han mostrado cómo cambiar una arquitectura convolucional a una arquitectura GAN. 

El primer métdo consiste en remplazar funciones espaciales como las funciones de pooling, con convoluciones stride (compresión de imágenes). El segundo método consiste en remover las capas completamente conectadas. La primera capa de la arquitectura GAN, la cual toma un ruido uniforme de la distribución Z como entrada, puede ser llamada una capa fully conectes slo como una multripicación de matrices, pero el resultado debe ser modificado para que tome las dimensiones de una dentos de 4 dimensiones y se use al unicio de la cola convolucional. Para el disciminador, la última convolución es aplandada y después alimenta a una salida sigmoide simple. El tercer paso es el uso de BatchNormalization y ReLU; con la excepción de usar como última capa de activación una función Tangente hiperbólica.