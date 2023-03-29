"""
###########################################
Biblioteca contenedora de las funciones de 
pérdida

autor: Ariel Cerón González
fecha: Marzo 14, 2023

Bajo la tutoria del Dr. Gibran Fuentes
IIMAS, UNAM
###########################################
"""

class Losses:
    def minmax_loss(self, real, fake):
        """
        Función de pérdida para el discriminador
        """
        return -torch.mean(torch.log(real + 1e-8) + torch.log(1 - fake + 1e-8))
    
    def generator_loss(self, fake):
        """
        Función de pérdida para el generador
        """
        return -torch.mean(torch.log(fake + 1e-8))
    
    def gradient_penalty(self, real, fake, discriminator, device):
        """
        Función de pérdida para el generador
        """
        BATCH_SIZE, C, H, W = real.shape
        epsilon = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
        interpolated_images = real * epsilon + fake * (1 - epsilon)
        # Calculate critic scores
        mixed_scores = discriminator(interpolated_images)
        # Take the gradient of the scores with respect to the images
        gradient = torch.autograd.grad(
            inputs=interpolated_images,
            outputs=mixed_scores,
            grad_outputs=torch.ones_like(mixed_scores),
            create_graph=True,
            retain_graph=True,
        )[0]
        gradient = gradient.view(gradient.shape[0], -1)
        gradient_norm = gradient.norm(2, dim=1)
        gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
        return gradient_penalty
    
    def wasserstein_loss(self, real, fake):
        """
        Función de pérdida para el generador
        """
        return torch.mean(fake) - torch.mean(real)
    
    def wasserstein_gradient_penalty(self, real, fake, discriminator, device):
        """
        Función de pérdida para el generador
        """
        BATCH_SIZE, C, H, W = real.shape
        epsilon = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
        interpolated_images = real * epsilon + fake * (1 - epsilon)
        # Calculate critic scores
        mixed_scores = discriminator(interpolated_images)
        # Take the gradient of the scores with respect to the images
        gradient = torch.autograd.grad(
            inputs=interpolated_images,
            outputs=mixed_scores,
            grad_outputs=torch.ones_like(mixed_scores),
            create_graph=True,
            retain_graph=True,
        )[0]
        gradient = gradient.view(gradient.shape[0], -1)
        gradient_norm = gradient.norm(2, dim=1)
        gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
        return gradient_penalty
    
    