import torch
import torch.nn as nn
import torch.optim as optim
from perceiver_pytorch import Perceiver

class MultiModalPerceiver(nn.Module):
    """
    MultiModalPerceiver is a neural network model that uses the Perceiver architecture to process inputs from 
    multiple modalities with different input dimensions. Each modality is first projected into a common embedding 
    space, and modality-specific embeddings are added to distinguish between modalities. The concatenated embeddings 
    are then passed to the Perceiver model, which outputs predictions for a regression task.

    Parameters:
    -----------
    input_dims : list of int
        A list containing the input dimensions for each modality. 
        For example, if there are three modalities with input sizes of 64, 128, and 256, input_dims would be [64, 128, 256].

    projection_dim : int, default=256
        The dimensionality of the projection of each modality.
    
    num_latents : int, default=16
        The number of latent vectors used in the Perceiver. Higher values can capture more detailed interactions 
        but may increase computation.

    latent_dim : int, default=128
        The dimensionality of each latent vector in the Perceiver. Larger latent dimensions allow for more information 
        to be stored within each latent representation.

    depth : int, default=8
        The number of layers in the Perceiver architecture, where each layer consists of cross-attention and 
        self-attention sub-layers. Increasing depth can improve model capacity at the cost of additional computation.

    cross_heads : int, default=1
        The number of attention heads used in the cross-attention layers. This parameter controls how many 
        attention heads are used when the input tokens attend to the latent vectors.

    latent_heads : int, default=1
        The number of attention heads used in the self-attention layers of the latent space. Increasing the number 
        of heads can enhance the model's ability to capture complex relationships.

    cross_dim_head : int, default=64
        The dimensionality of each cross-attention head. This parameter affects the amount of information each head 
        can focus on in the cross-attention mechanism.

    latent_dim_head : int, default=64
        The dimensionality of each self-attention head in the latent space. Adjusting this affects how the model 
        captures details within each attention head in the latent space.

    attn_dropout : float, default=0.1
        Dropout probability applied to the attention layers within the Perceiver. Helps prevent overfitting by 
        randomly dropping some attention scores during training.

    ff_dropout : float, default=0.1
        Dropout probability applied to the feedforward layers within the Perceiver. This dropout helps prevent 
        overfitting in the feedforward stages.

    output_dim : int, default=1
        The output dimensionality of the model, which is typically set to 1 for regression tasks. Adjust this 
        for other types of tasks that may require multiple outputs.

    Attributes:
    -----------
    projections : nn.ModuleList
        A list containing linear projection layers for each modality. Each layer includes a Linear transformation, 
        BatchNorm1d, and ReLU activation to project each modality to a fixed 256-dimensional embedding.

    modality_embeddings : nn.ParameterList
        A list of learnable modality-specific embeddings that are added to the projected input of each modality. 
        These embeddings allow the model to distinguish between the different modalities in a learnable way.

    perceiver : Perceiver
        The Perceiver model that takes the concatenated projections from all modalities as input. The Perceiver 
        processes this data through cross-attention and self-attention layers to output a prediction.
    """

    def __init__(self, 
                 input_dims,          # List of input dimensions for each modality, e.g., [64, 128, 256]
                 input_channels = 1,
                 input_axis = 1,
                 projection_dim = 256,
                 num_latents=16,      
                 latent_dim=128,       
                 depth=8,              
                 cross_heads=8,        
                 latent_heads=8,       
                 cross_dim_head=32,
                 latent_dim_head=32, 
                 attn_dropout=0.1,     
                 ff_dropout=0.0,       
                 output_dim=1,          # Output dimension for regression
                 weight_tie_layers=True, 
                 fourier_encode_data = False,
                 max_freq = 10,
                 num_freq_bands = 4
                ):
        
        super(MultiModalPerceiver, self).__init__()

        self.input_dims = input_dims
        self.projection_dim = projection_dim 
        
        # Define modality-specific learnable projections with 1D BatchNorm
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, self.projection_dim),
                nn.BatchNorm1d(self.projection_dim),
                nn.ReLU()
            ) for dim in input_dims
        ])

        # Learnable modality embeddings to differentiate between modalities
        self.modality_embeddings = nn.ParameterList([
            nn.Parameter(torch.randn(1, self.projection_dim)) for _ in input_dims
        ])

        input_dim=self.projection_dim * len(input_dims)  # Concatenated input dimension

        # Perceiver model
        self.perceiver = Perceiver(
            input_channels = input_channels,
            input_axis = input_axis,
            num_latents=num_latents,
            latent_dim=latent_dim,
            depth=depth,
            cross_heads=cross_heads,
            latent_heads=latent_heads,
            cross_dim_head=cross_dim_head,
            latent_dim_head=latent_dim_head,
            num_classes=output_dim,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            weight_tie_layers = weight_tie_layers, 
            fourier_encode_data = fourier_encode_data,
            num_freq_bands = num_freq_bands,
            max_freq = max_freq )
            

    def forward(self, x):
        # x should be a list of tensors, one for each modality
        assert len(x) == len(self.input_dims), "Expected input for each modality."

        # Project each modality and add modality-specific embeddings
        projected = []
        for i, modality in enumerate(x):
            # Apply the linear projection + BatchNorm
            proj = self.projections[i](modality)
            # Add modality embedding
            proj = proj + self.modality_embeddings[i]
            projected.append(proj)

        # Concatenate all modality projections along the last dimension
        concatenated = torch.cat(projected, dim=-1)

        # Pass through Perceiver model
        output = self.perceiver(concatenated)
        return output