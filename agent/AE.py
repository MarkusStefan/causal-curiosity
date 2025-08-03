


class AutoEncoder2d():

    def __init__(self):
        self.encoder = None
        self.decoder = None
        

    def forward(self, x):
        """
        Forward pass through the autoencoder.
        
        :param x: Input tensor.
        :return: Reconstructed tensor.
        """
        z = self.encoder(x)
        x = self.decoder(z)
        return x