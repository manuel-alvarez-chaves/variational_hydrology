from enum import Enum

import torch
from torch import nn
from torch.distributions.multivariate_normal import MultivariateNormal


class SamplingMode(Enum):
    STANDARD = "standard"
    LEARNED = "learned"

class ErrorMode(Enum):
    PROPORTIONAL = "proportional"
    EXPONENTIAL = "exponential"
    DENSE = "dense"

class VLSTM(nn.Module):
    def __init__(self,
                 num_input: int,
                 num_hidden: int,
                 output_dropout: float,
                 error: ErrorMode = ErrorMode.PROPORTIONAL,
                 **kwargs,
    ):
        super().__init__()
        """
        Variational LSTM (VLSTM) model.

        Parameters:
        -----------
            num_input : int
                Number of input features
            num_hidden : int
                Number of hidden units in the LSTM
            output_dropout : float
                Dropout rate for the output of the encoder
            error : ErrorMode
                Error mode for the VLSTM model (see 'decode')
        """
        self.input_size = num_input
        self.hidden_size = num_hidden
        self.error = error
        self.output_dropout = output_dropout
        
        # Encoder
        self.encoder = nn.LSTM(input_size=num_input, hidden_size=num_hidden, batch_first=True)
        self.dropout = nn.Dropout(output_dropout)
        self._reset_parameters()

        # Decoder
        self.decoder = nn.Linear(num_hidden, 1)

        # Variational Inference
        self.fc_logvar = nn.Linear(num_hidden, num_hidden)
        self.m = MultivariateNormal(torch.zeros(num_hidden), torch.eye(num_hidden))

        # Dense layer for Error Mode
        if self.error == ErrorMode.DENSE:
            num_layers = kwargs.get("num_layers", 1)
            activation = kwargs.get("activation", nn.ReLU())

            layers = []
            for _ in range(num_layers):
                layers.append(nn.Linear(num_hidden, num_hidden))
                layers.append(activation)
            
            self.dense = nn.Sequential(*layers)
                

    def _reset_parameters(self):
        self.encoder.bias_hh_l0.data[self.hidden_size:2 * self.hidden_size] = 3.0
    
    def _reparametrize(self, logvar):
        """Reparameterizes a latent variable sampled from a normal distribution with a mean of 0 and a standard deviation derived from the log-variance.
        
        Parameters
        ----------
        self : VLSTM input to access attributes and methods of the current instance of the class.
        logvar : input parameter for standard deviation (std) calculation.
        
        Returns
        -------
        Scales the random noise by std simulate sampling from the original distribution paratemized by logvar.
        std : Standard deviation calculation with logvar as the input. 
        eps : Generates random noise sampled from a standard normal distribution (mean = 0  , variance = 1).
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std).requires_grad_(False)
        return eps * std
    
    def encode(self, x):
        """
        Compresses input sequence data into a single fixed-size representation.
        
        Parameters
        ----------
        x : Represents the input tensor containing sequential data.
        self : Access the attributes and methods of the class.
        
        Returns
        -------
        The output processed representation of the input sequence, derived from the final hidden state of the encoder.
        """
        _, (h_n, _) = self.encoder(x)
        return self.dropout(h_n[-1]) # many-to-one
    
    def decode(self, encoded, z):
        """
        Applies a tranformation to the encoded input using an error mode specified in matching and before passing it through return.
        
        Parameters
        ---------
        self : Access attributes and methods of the current instance of the class.
        encoded : Represents the fixed-size encoded input data, and holds the compact representation of the original input sequence.
        z : Represents a latent varaible, error term or auxiliary inpur that modifies the encoded data during the decoding process.
        
        Returns
        -------
        Error Mode : 
            Proportional : The encoded input is scaled proportionally before being passed into self.decoder.
            Exponential : The modification is exponential.
            Dense : A learned transformation is applied before decoding.
        """
        match self.error:
            case ErrorMode.PROPORTIONAL:
                return self.decoder(encoded * (1 + z))
            case ErrorMode.EXPONENTIAL:
                return self.decoder(encoded * (1 + z.exp()))
            case ErrorMode.DENSE:
                return self.decoder(encoded * (1 + self.dense(z)))
    
    def forward(self, x):
        """
        Defines the forward pass of a neural network model, likely incorporating variational inference.
        
        Parameters
        ----------
        self : Access attributes and methods of the current instance of the class.
        x : Represents the input tensor containing sequential data and is processed through an encoding function to create a compact representation.
        
        Returns
        -------
        encoded : Condensed representation for feature extraction or embedding.
        decoded : Reconstructed output for evaluating model performance.
        logvar : Quantifies uncertainty or variability in the latent space.
        """
        encoded = self.encode(x)
        logvar = self.fc_logvar(encoded)
        z = self._reparametrize(logvar)
        decoded = self.decode(encoded, z)
        return encoded, decoded, logvar
    
    def generate_samples(self, x, num_samples, mode=SamplingMode.STANDARD):
        """
        Generates multiple samples based on the input tensor and specified sampling mode.
        Utilizes an encoding process and a decoder, with the option to either use a standard sampling approach or a learned distribution.
        
        Parameters
        ----------
        self : Refers to the instance of the class and provides access to methods and attributes.
        x : The input tensor containing sequential data typically of shape.
        num_samples : Specifies the number of samples to generate for each input sequence.
        mode : 
            Type: Sampling Mode
            Determines how the latent variable z is generated:
                STANDARD : Uses standar normal distribution for sampling.
                LEARNED : Samples based on a learned distribution.
        
        Returns
        -------
        The decoded samples generated by combining the encoded input and the latent variable 
        based on the selected sampling mode.
        """
        num_batches = x.shape[0]
        encoded = self.encode(x)
        encoded = encoded.unsqueeze(1).repeat(1, num_samples, 1)
        if mode == SamplingMode.STANDARD:
            z = self.m.sample((num_batches, num_samples)).to(x.device)
        elif mode == SamplingMode.LEARNED:
            logvar = self.fc_logvar(encoded)
            z = self._reparametrize(logvar) 
        else:
            raise ValueError("Invalid Sampling Mode")
        return self.decode(encoded, z)

    def sample(self, x, num_samples, mode=SamplingMode.STANDARD, track_grad=False):
        """
        Generate multiple samples based on an input tensor while offering flexibility in the sampling strategy.
        Performs within a probabilistic framework.
        
        Parameters
        ----------
        self : Refers to the instance of the class, providing access to methods.
        x : Represents the input tensor containing sequential data, typically of shape.
        num_samples : Specifies the number of samples to generate for each input sequence.
        mode : Determines how the latent variable is generated during sampling.
        track_grad : A boolean flag that determines whether gradients should be tracked during sampling:
                        True : gradients are tracked, allowing for back propagation.
                        False : sampling occurs without tracking gradients, saving computation time and memory.
        
        Returns
        -------
        self.generate_samples : The decoded samples generated, combining the encoded input and the latent variable.
        """
        if track_grad:
            return self.generate_samples(x, num_samples, mode)
        else:
            with torch.no_grad():
                return self.generate_samples(x, num_samples, mode)
