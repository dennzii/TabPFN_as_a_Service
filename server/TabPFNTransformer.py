import torch
import torch.nn as nn

class Transformer_TabPFN(nn.Module):


    """
    We considered only PFN Transformers with 12 layers, embeddings size 512, hidden size 1024 in
    feed-forward layers, and 4-head attention. We used the Adam optimizer (Kingma and Ba, 2015) with
    linear-warmup and cosine annealing (Loshchilov and Hutter, 2017). For each training we tested a set
    of 3 learning rates, {.001, .0003, .0001}, and used the one with the lowest final training loss. The
    resulting model contains 25.82 M parameters.
    """
    

    def __init__(self,d_model = 512, num_layers = 12, num_heads = 4,hidden_size_ff = 1024, max_features = 100, num_classes = 10):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_features = max_features
        self.hidden_size_ff = hidden_size_ff
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

         # N, (features + 1) @( features +1),d_model
        self.embedding = nn.Linear(self.max_features + 1,self.d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_size_ff,
            batch_first=True,
            activation='gelu'
        )

        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer,num_layers=self.num_layers)
        
        self.classifier = nn.Linear(self.d_model,self.num_classes)


    def forward(self,x,n_train):

        attn_mask = torch.zeros((x.shape[1],x.shape[1]),device=x.device)
        attn_mask[:,n_train:] = float('-inf')

        x = self.embedding(x)
        x = self.encoder(x,mask = attn_mask)
        

        x_test = x[:,n_train:,:]#only compute the representation of test samples.

        out = self.classifier(x_test)

        return out
#loss 152.stepte 1.85
