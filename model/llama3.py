import math
from typing import Optional
from dataclasses import dataclass
import torch
from torch import Tensor
from torch import exp
from torch import arange, outer
from torch import polar, ones_like
from torch import zeros, repeat_interleave
from torch import view_as_complex, view_as_real
from torch import ones, rsqrt
from torch import hstack
from torch.nn import Module
from torch.nn import Parameter
from torch.nn import ModuleList
from torch.nn import Parameter
from torch.nn import Linear
from torch.nn import Embedding
from torch.nn import Sequential, SiLU
from torch.nn.functional import scaled_dot_product_attention


class RMSNorm(Module):
    def __init__(self, model_dimension: int, epsilon: float = 1e-5):
        super().__init__()
        self.epsilon = epsilon
        self.weight = Parameter(ones(model_dimension))

    def norm(self, input: Tensor) -> Tensor:
        return input * rsqrt(input.pow(2).mean(-1, keepdim=True) + self.epsilon)

    def forward(self, input: Tensor) -> Tensor:
        output = self.norm(input.float()).type_as(input)
        return output * self.weight
    
def precompute_complex_positional_embeddings(model_dimension: int, sequence_lenght_limit: int, scaling_factor: float = 10000.0) -> Tensor:
    frequencies = Tensor(sequence_lenght_limit, model_dimension // 2)
    frequencies = exp(- arange(0, model_dimension, 2) * math.log(scaling_factor) / model_dimension)
    frequencies = outer(arange(sequence_lenght_limit), frequencies)
    return polar(ones_like(frequencies), frequencies)

def split(sequence: Tensor, number_of_heads: int) -> Tensor:
    batch_size, sequence_length, model_dimension = sequence.shape
    sequence = sequence.view(batch_size, sequence_length, number_of_heads, model_dimension // number_of_heads)
    sequence = sequence.transpose(1, 2)
    assert sequence.dim() == 4
    return sequence

def concat(sequence: Tensor) -> Tensor:
    batch_size, number_of_heads, sequence_lenght, heads_dimension = sequence.shape
    sequence = sequence.transpose(1, 2)
    sequence = sequence.reshape(batch_size, sequence_lenght, heads_dimension* number_of_heads)
    return sequence

def apply_rotatory_embeddings(sequence: Tensor, rotatory_embeddings: Tensor) -> Tensor:
    batch_size, number_of_heads, sequence_lenght, heads_dimension = sequence.shape
    sequence = sequence.view(batch_size, number_of_heads, sequence_lenght, heads_dimension // 2, 2)
    sequence = view_as_complex(sequence)
    sequence = sequence * rotatory_embeddings
    sequence = view_as_real(sequence)
    return sequence.view(batch_size, number_of_heads, sequence_lenght, heads_dimension)


class Cache(Module):
    def __init__(self, batch_size_limit: int, sequence_lenght_limit: int, number_of_heads: int, heads_dimension: int):
        super().__init__()
        self.sequence_cache = Parameter(data=zeros(batch_size_limit, number_of_heads, sequence_lenght_limit, heads_dimension), requires_grad=False)

    def forward(self, sequence: Tensor, start_position: int) -> Tensor:
        batch_size, sequence_lenght = sequence.size(0), sequence.size(2)
        self.sequence_cache[:batch_size, :, start_position: start_position+sequence_lenght]
        return self.sequence_cache[:batch_size, :, :start_position+sequence_lenght]

class Attention(Module):
    def __init__(self, model_dimension: int, number_of_heads: int, number_of_kv_heads: int,batch_size_limit: int, sequence_lenght_limit: int):
        super().__init__()
        self.model_dimension = model_dimension
        self.heads_dimension = model_dimension // number_of_heads
        self.number_of_heads = number_of_heads
        self.number_of_kv_heads = number_of_kv_heads
        self.repeats = self.number_of_heads // self.number_of_kv_heads

        self.q_projector = Linear(model_dimension, self.heads_dimension * self.number_of_heads, bias=False)
        self.k_projector = Linear(model_dimension, self.heads_dimension * self.number_of_kv_heads, bias=False)
        self.v_projector = Linear(model_dimension, self.heads_dimension * self.number_of_kv_heads, bias=False)
        self.output_projector = Linear(self.number_of_heads * self.heads_dimension, model_dimension, bias=False)

        self.k_cache = Cache(batch_size_limit, sequence_lenght_limit, self.number_of_kv_heads, self.heads_dimension)
        self.v_cache = Cache(batch_size_limit, sequence_lenght_limit, self.number_of_kv_heads, self.heads_dimension)
        
    def forward(self, sequence: Tensor, rotatory_embeddings: Tensor, start_position: int, mask: Optional[Tensor] = None) -> Tensor:
        query, key, value = self.q_projector(sequence), self.k_projector(sequence), self.v_projector(sequence)        
        query, key, value = split(query, self.number_of_heads), split(key, self.number_of_kv_heads), split(value, self.number_of_kv_heads)
        query, key = apply_rotatory_embeddings(query, rotatory_embeddings), apply_rotatory_embeddings(query, rotatory_embeddings)
        key, value = self.k_cache(key, start_position), self.v_cache(value, start_position)
        key, value = repeat_interleave(key, self.repeats, 1), repeat_interleave(value, self.repeats, 1)
        attention = scaled_dot_product_attention(query, key, value, mask)
        attention = concat(attention)
        return self.output_projector(attention)
    
class FeedForward(Module):
    def __init__(self, model_dimension: int, hidden_dimension: int, multiple_of: int, multiplier: Optional[int] = None):
        super().__init__()
        self.model_dimension = model_dimension
        self.hidden_dimension = hidden_dimension
        if multiplier:
            self.hidden_dimension = int(multiplier * hidden_dimension)
        self.hidden_dimension = multiple_of * ((hidden_dimension + multiple_of - 1) // multiple_of)

        self.activation = SiLU()
        self.input_layer = Linear(model_dimension, self.hidden_dimension, bias=False)
        self.output_layer = Linear(self.hidden_dimension, model_dimension, bias=False)
        self.gate_layer = Linear(model_dimension, self.hidden_dimension, bias=False)
       
    def forward(self, input: Tensor) -> Tensor:
        output = self.activation(self.input_layer(input)) * self.gate_layer(input)
        return self.output_layer(output)


@dataclass
class Settings:        
    number_of_layers: int
    vocabular_size: int
    model_dimension: int
    hidden_dimension: int
    number_of_heads: int
    number_of_kv_heads: int
    batch_size_limit: int
    sequence_lenght_limit: int
    multiple_of: int
    positional_encoding_scaling_factor: float = 10000.0
    multiplier: Optional[int] = None


class Decoder(Module):
    def __init__(self, id: int, settings: Settings):
        super().__init__()
        self.id = id
        self.attention = Attention(
            model_dimension=settings.model_dimension,
            number_of_heads=settings.number_of_heads,
            number_of_kv_heads=settings.number_of_kv_heads,
            batch_size_limit=settings.batch_size_limit,
            sequence_lenght_limit=settings.sequence_lenght_limit
        )
        
        self.attention_norm = RMSNorm(settings.model_dimension)

        self.ffn = FeedForward(
            settings.model_dimension, 
            settings.hidden_dimension, 
            settings.multiple_of, 
            settings.multiplier
        )

        self.ffn_norm = RMSNorm(settings.model_dimension)
        

    def forward(self, input: Tensor, start_position: int, rotatory_embeddings: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        output = input + self.attention(self.attention_norm(input), rotatory_embeddings, start_position, mask)
        return output + self.ffn(self.ffn_norm(output))


class Transformer(Module):
    def __init__(self, settings: Settings):
        super().__init__()
        self.embeddings = Embedding(settings.vocabular_size, settings.model_dimension)
        self.layers = ModuleList([ Decoder(id=layer,settings=settings) for layer in range(settings.number_of_layers) ])
        self.output_layer = Linear(settings.model_dimension, settings.vocabular_size, bias=False)
        self.positional_embeddings = Parameter(data=precompute_complex_positional_embeddings(
            settings.model_dimension // settings.number_of_heads, 
            settings.sequence_lenght_limit,
            settings.positional_encoding_scaling_factor
        ), requires_grad=False)
        self.norm = RMSNorm(settings.model_dimension)

    def forward(self, tokens: Tensor, start_position: int) -> Tensor:
        batch_size, sequence_lenght = tokens.shape
        sequence = self.embeddings(tokens)
        positional_embeddings = self.positional_embeddings[start_position: start_position + sequence_lenght]

        mask = None
        if sequence_lenght > 1:
            mask = ones(sequence_lenght, sequence_lenght)
            mask = mask.triu(diagonal=1)
            mask = hstack([zeros(sequence_lenght, start_position), mask]).to(tokens.device)
            
        for layer in self.layers:
            sequence = layer(sequence, start_position, positional_embeddings, mask)
        
        sequence = self.norm(sequence)
        return self.output_layer(sequence)