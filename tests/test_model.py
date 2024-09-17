import pytest

### Testing using the llama3 model from the https://github.com/meta-llama/llama3 repository

import torch
import fairscale.nn.model_parallel.initialize as fs_init
import torch.distributed as dist
import os

from model.original_llama3 import ModelArgs, Attention, RMSNorm, FeedForward, Transformer, precompute_freqs_cis
from model.original_llama3 import TransformerBlock 

from model.llama3 import Attention as IAttention, apply_rotatory_embeddings
from model.llama3 import precompute_complex_positional_embeddings
from model.llama3 import RMSNorm as IRMSNorm
from model.llama3 import FeedForward as IFeedForward
from model.llama3 import Decoder as ITransformerBlock
from model.llama3 import Transformer as ITransformer
from model.llama3 import Settings
    

@pytest.fixture(scope='session')
def fairscale_init():
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['RANK'] = '0'
    dist.init_process_group(backend='gloo')
    fs_init.initialize_model_parallel(1)
    yield
    dist.destroy_process_group()


def test_attention(fairscale_init):
    with torch.no_grad():

        args = ModelArgs(
            dim=4096,
            n_layers=1,
            n_heads=32,
            n_kv_heads=8,
            vocab_size=-1,
            multiple_of=256,
            ffn_dim_multiplier=None,
            norm_eps=1e-5,
            rope_theta=500000,
            max_batch_size=32,
            max_seq_len=2048,
        )
        

        attention = Attention(args)
        attention.eval()

        x = torch.randn(2, 2048, 4096)

        freqs_cis = precompute_freqs_cis(4096, 2048, theta=args.rope_theta)
        freqs_cis = freqs_cis[:, :64]

        iattention = IAttention(
            model_dimension=4096,
            number_of_heads=32,
            number_of_kv_heads=8,
            batch_size_limit=32,
            sequence_lenght_limit=2048,
        )

        attention.cache_k.random_()
        attention.cache_v.random_()

        iattention.k_cache.sequence_cache.copy_(attention.cache_k)
        iattention.v_cache.sequence_cache.copy_(attention.cache_v)
        iattention.q_projector.weight.copy_(attention.wq.weight)
        iattention.k_projector.weight.copy_(attention.wk.weight)
        iattention.v_projector.weight.copy_(attention.wv.weight)
        iattention.output_projector.weight.copy_(attention.wo.weight)

        iattention.eval()

        output2 = iattention(x, 0)        
        output = attention(x, 0, freqs_cis)

        assert output.shape == output2.shape
        assert torch.allclose(output, output2, atol=1e-5)
        
        output2 = iattention(x, 0)        
        output = attention(x, 0, freqs_cis)
        assert torch.allclose(output, output2, atol=1e-5)

def test_norm():
    with torch.no_grad():
        norm = RMSNorm(4096, eps=1e-6)

        inorm = IRMSNorm(4096, 1e-6)

        x = torch.randn(2, 2048, 4096)

        output = norm(x)
        output2 = inorm(x)

        assert output.shape == output2.shape
        assert torch.allclose(output, output2, atol=1e-5)


def test_complex_positional_embeddings():
    with torch.no_grad():
        theta = 500000
        freqs_cis = precompute_freqs_cis(256, 512, theta=theta)
        postional_embeddings = precompute_complex_positional_embeddings(256, 512,theta)
        assert torch.allclose(freqs_cis, postional_embeddings, atol=1e-3)


def test_feed_forward():
    with torch.no_grad():
        hidden_dim = 4096
        ffn = FeedForward(4096, hidden_dim, 256, None)

        hidden_dim = int(2 * hidden_dim / 3)
        iffn = IFeedForward(4096, hidden_dim, 256, None)
        
        iffn.gate_layer.weight.copy_(ffn.w3.weight)
        iffn.input_layer.weight.copy_(ffn.w1.weight)
        iffn.output_layer.weight.copy_(ffn.w2.weight)

        x = torch.randn(2, 2048, 4096)

        output = ffn(x)
        output2 = iffn(x)

        assert output.shape == output2.shape
        assert torch.allclose(output, output2, atol=1e-5)


def test_decoder():
    with torch.no_grad():
        transformer_block = TransformerBlock(0, ModelArgs(
            dim=4096,
            n_layers=1,
            n_heads=32,
            n_kv_heads=8,
            vocab_size=-1,
            multiple_of=256,
            ffn_dim_multiplier=None,
            norm_eps=1e-5,
            rope_theta=500000,
            max_batch_size=32,
            max_seq_len=2048,
        ))

        itransformer_block = ITransformerBlock(0, Settings(
            number_of_layers=1,
            model_dimension=4096,
            hidden_dimension=int(4 *2 * 4096 / 3),
            number_of_heads=32,
            number_of_kv_heads=8,
            batch_size_limit=32,
            sequence_lenght_limit=2048,
            multiple_of=256,
            multiplier=None,
            positional_encoding_scaling_factor=500000,
            vocabular_size=-1
            )
        )


        transformer_block.attention.cache_k.random_()
        transformer_block.attention.cache_v.random_()

        transformer_block.eval()

        itransformer_block.attention.k_cache.sequence_cache.copy_(transformer_block.attention.cache_k)
        itransformer_block.attention.v_cache.sequence_cache.copy_(transformer_block.attention.cache_v)
        itransformer_block.attention.q_projector.weight.copy_(transformer_block.attention.wq.weight)
        itransformer_block.attention.k_projector.weight.copy_(transformer_block.attention.wk.weight)
        itransformer_block.attention.v_projector.weight.copy_(transformer_block.attention.wv.weight)
        itransformer_block.attention.output_projector.weight.copy_(transformer_block.attention.wo.weight)

        itransformer_block.ffn.input_layer.weight.copy_(transformer_block.feed_forward.w1.weight)
        itransformer_block.ffn.output_layer.weight.copy_(transformer_block.feed_forward.w2.weight)
        itransformer_block.ffn.gate_layer.weight.copy_(transformer_block.feed_forward.w3.weight)

        x = torch.randn(2, 2048, 4096)

        freqs_cis = precompute_freqs_cis(4096, 2048 * 2, 500000)
        freqs_cis = freqs_cis[:2048, :64]

        output = transformer_block(x, 0, freqs_cis, None)
        output2 = itransformer_block(x, 0)

        assert output.shape == output2.shape
        assert torch.allclose(output, output2, atol=1e-4)

def test_transformer(fairscale_init):
    with torch.no_grad():
        args = ModelArgs(
            dim=4096,
            n_layers=1,
            n_heads=32,
            n_kv_heads=8,
            vocab_size=4000,
            multiple_of=256,
            ffn_dim_multiplier=None,
            norm_eps=1e-5,
            rope_theta=500000,
            max_batch_size=32,
            max_seq_len=2048,
        )

        settings = Settings(
            number_of_layers=1,
            model_dimension=4096,
            hidden_dimension=int(4 *2 * 4096 / 3),
            number_of_heads=32,
            number_of_kv_heads=8,
            batch_size_limit=32,
            sequence_lenght_limit=2048,
            multiple_of=256,
            multiplier=None,
            positional_encoding_scaling_factor=500000,
            vocabular_size=4000
        )

        transformer = Transformer(args)
        itransformer = ITransformer(settings)

        transformer.tok_embeddings.weight.random_()
        itransformer.embeddings.weight.copy_(transformer.tok_embeddings.weight)

        transformer.output.weight.random_()
        itransformer.output_layer.weight.copy_(transformer.output.weight)


        for i, layer in enumerate(transformer.layers):
            for j, ilayer in enumerate(itransformer.layers):
                if i == j:
                    layer.attention.cache_k.random_()
                    layer.attention.cache_v.random_()
                    ilayer.attention.k_cache.sequence_cache.copy_(layer.attention.cache_k)
                    ilayer.attention.v_cache.sequence_cache.copy_(layer.attention.cache_v)
                    ilayer.attention.q_projector.weight.copy_(layer.attention.wq.weight)

                    ilayer.attention.k_projector.weight.copy_(layer.attention.wk.weight)
                    ilayer.attention.v_projector.weight.copy_(layer.attention.wv.weight)
                    ilayer.attention.output_projector.weight.copy_(layer.attention.wo.weight)
                    
                    ilayer.ffn.input_layer.weight.copy_(layer.feed_forward.w1.weight)
                    ilayer.ffn.output_layer.weight.copy_(layer.feed_forward.w2.weight)
                    ilayer.ffn.gate_layer.weight.copy_(layer.feed_forward.w3.weight)



        transformer.eval()

        itransformer.eval()

        x = torch.randint(1, 4000, (2, 2048))

        output = transformer(x, 0)
        output2 = itransformer(x, 0)

        assert output.shape == output2.shape
        assert torch.allclose(output, output2, atol=1e-4)