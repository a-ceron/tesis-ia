import matplotlib.pyplot as plt
import torch

def plot_batch(generator, device, batch_size, noise_dim, deprocess):
    n_images = 64
    generator.eval()
    noise = torch.randn(batch_size, noise_dim, device=device)

    gen_batch = generator(noise).detach().cpu()
    plt.figure(figsize=(16, 4))
    
    plot_batch = deprocess(gen_batch)
    for i in range(n_images):
        plt.subplot(4, 16, i+1)
        plt.imshow(np.transpose(plot_batch[i], [1, 2, 0]).numpy().astype("uint8"))
        plt.axis('off')
    plt.show()
    