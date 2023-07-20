import numpy as np
import math
import torch as t
import torch.nn as nn
import torch.nn.functional as F

from jukebox.transformer.ops import filter_logits, discretized_mix_logistic_loss, sample_mol
#from jukebox.transformer.transformer import Transformer
from fast_transformers.builders import TransformerEncoderBuilder
from fast_transformers.builders import RecurrentEncoderBuilder
from fast_transformers.masking import TriangularCausalMask
from jukebox.utils.logger import get_range
from jukebox.utils.torch_utils import empty_cache

def get_normal(*shape, std=0.01):
    w = t.empty(shape)
    nn.init.normal_(w, std=std)
    return w

def roll(x, n):
    return t.cat((x[:, -n:], x[:, :-n]), dim=1)

def split_chunks(length, chunk_size):
    n_passes = (length + chunk_size - 1) // chunk_size
    chunk_sizes = [*[chunk_size] * (n_passes - 1), (length - 1) % chunk_size + 1]
    assert sum(chunk_sizes) == length
    return chunk_sizes


class PositionEmbedding(nn.Module):
    def __init__(self, input_shape, width, init_scale=1.0, pos_init=False):
        super().__init__()
        self.input_shape = input_shape
        self.input_dims = input_dims = np.prod(input_shape)
        self.pos_init = pos_init
        if pos_init:
            self.register_buffer('pos', t.tensor(get_pos_idx(input_shape)).long())
            self._pos_embs = nn.ModuleList()
            for i in range(len(input_shape)):
                emb = nn.Embedding(input_shape[i], width)
                nn.init.normal_(emb.weight, std=0.02)
                self._pos_embs.append(emb)
        else:
            self.pos_emb = nn.Parameter(get_normal(input_dims, width, std=0.01 * init_scale))

    def forward(self):
        if self.pos_init:
            pos_emb = sum([self._pos_embs[i](self.pos[:,i]) for i in range(len(self.input_shape))])
        else:
            pos_emb = self.pos_emb
        return pos_emb


class ConditionalAutoregressive2D(nn.Module):
    def __init__(self, input_shape, bins,
                 width=128, depth=2, heads=1,
                 attn_dropout=0.0, resid_dropout=0.0, emb_dropout=0.0, mask=True,
                 zero_out=False, pos_init=False, x_cond=False, y_cond=False,
                 encoder_dims=0, only_encode=False, merged_decoder=False, 
                 prime_len=None, num_mixtures=10,
                 m_attn=0.25, m_mlp=1, init_scale=1.0, checkpoint_res=0, train=False):
        super().__init__()
        self.input_shape = input_shape
        self.input_dims = input_dims = np.prod(input_shape)
        self.encoder_dims = encoder_dims
        self.bins = bins
        self.width = width
        self.depth = depth

        self.x_emb = nn.Embedding(bins, width)
        nn.init.normal_(self.x_emb.weight, std=0.02 * init_scale)
        self.x_emb_dropout = nn.Dropout(emb_dropout)
        self.y_cond = y_cond
        self.x_cond = x_cond
        if not y_cond:
            self.start_token = nn.Parameter(get_normal(1, width, std=0.01 * init_scale))

        self.pos_emb = PositionEmbedding(input_shape=input_shape, width=width, init_scale=init_scale, pos_init=pos_init)
        self.pos_emb_dropout = nn.Dropout(emb_dropout)

        
        if train:
            self.transformer = TransformerEncoderBuilder.from_kwargs(
                n_layers= depth,
                n_heads= heads,
                feed_forward_dimensions=int(m_mlp * width),
                model_dimensions=width,
                query_dimensions= int(m_attn*width)//heads,
                value_dimensions= int(m_attn*width)//heads,
                activation='gelu',
                dropout=attn_dropout,
                attention_type="causal-linear",
            ).get()
        else:
            # encoder (inference)
            print(' [o] using RNN backend.')
            self.transformer = RecurrentEncoderBuilder.from_kwargs(
                n_layers= depth,
                n_heads= heads,
                model_dimensions=width,
                feed_forward_dimensions=int(m_mlp * width),
                query_dimensions= int(m_attn*width)//heads,
                value_dimensions= int(m_attn*width)//heads,
                dropout=attn_dropout,
                activation='gelu',
                attention_type="causal-linear",
            ).get()

        self.only_encode = only_encode
        self.prime_len = prime_len
        if merged_decoder:
            # Merged piped model uses this setup
            self.add_cond_after_transformer = False
            #self.share_x_emb_x_out = False
        else:
            self.add_cond_after_transformer = True
            #self.share_x_emb_x_out = True

        if not only_encode:
            self.x_out = nn.Linear(width, 3*num_mixtures, bias=False)
            #if self.share_x_emb_x_out:
            #    self.x_out.weight = self.x_emb.weight
            #self.dmlloss = discretized_mix_logistic_loss()
            #self.loss = t.nn.CrossEntropyLoss()

    def preprocess(self, x):
        # Input: x is NHWC and uint8. Converted to NL and long
        # Can include stuff like bitpacking, reordering here.
        N = x.shape[0]
        return x.view(N, -1).long()

    def postprocess(self, x, sample_tokens=None):
        # Convert back from NL and long to NHWC
        N = x.shape[0]
        assert (0 <= x).all() and (x < self.bins).all()
        if sample_tokens is None or sample_tokens==self.input_dims:
            return x.view(N, *self.input_shape)
        else:
            return x.view(N, -1)

    def forward(self, x, x_cond=None, y_cond=None, encoder_kv=None, fp16=False, loss_full=False,
                encode=False, get_preds=False, get_acts=False, get_sep_loss=False):
        # Preprocess.
        with t.no_grad():
            x = self.preprocess(x)

        N, D = x.shape
        assert isinstance(x, t.cuda.LongTensor)
        assert (0 <= x).all() and (x < self.bins).all()

        if self.y_cond:
            assert y_cond is not None
            assert y_cond.shape == (N, 1, self.width)
        else:
            assert y_cond is None

        if self.x_cond:
            assert x_cond is not None
            assert x_cond.shape == (N, D, self.width) or x_cond.shape == (N, 1, self.width), f"{x_cond.shape} != {(N, D, self.width)} nor {(N, 1, self.width)}. Did you pass the correct --sample_length?"
        else:
            assert x_cond is None
            x_cond = t.zeros((N, 1, self.width), device=x.device, dtype=t.float)

        x_t = x # Target
        x = self.x_emb(x) # X emb
        x = roll(x, 1) # Shift by 1, and fill in start token
        if self.y_cond:
            x[:,0] = y_cond.view(N, self.width)
        else:
            x[:,0] = self.start_token

        x = self.x_emb_dropout(x) + self.pos_emb_dropout(self.pos_emb()) + x_cond # Pos emb and dropout
        
        triangular_mask = TriangularCausalMask(x.shape[1], device=x.device)

        x = self.transformer(x, attn_mask=triangular_mask) # Transformer
        if self.add_cond_after_transformer: # Piped doesnt add x_cond
            x = x + x_cond

        acts = x
        if self.only_encode:
            return x
        x = self.x_out(x) # Predictions

        if get_sep_loss:
            assert self.prime_len is not None
            x_prime = x[:, :self.prime_len].reshape(-1, self.bins)
            x_gen = x[:, self.prime_len:].reshape(-1, self.bins)

            prime_loss = F.cross_entropy(x_prime, x_t[:, :self.prime_len].reshape(-1)) / np.log(2.)
            gen_loss = F.cross_entropy(x_gen, x_t[:, self.prime_len:].reshape(-1)) / np.log(2.)

            loss = (prime_loss, gen_loss) # Note order! Prime is first
        else:
            loss = discretized_mix_logistic_loss(x.permute(0, 2, 1).contiguous(), x_t.view(N, D, 1),num_classes=self.bins)  / np.log(2.) 

        if get_preds:
            return loss, x
        elif get_acts:
            return loss, acts
        else:
            return loss, None

    def get_emb(self, sample_t, n_samples, x, x_cond, y_cond):
        N, D = n_samples, self.input_dims
        if sample_t == 0:
            # Fill in start token
            x = t.empty(n_samples, 1, self.width).cuda()
            if self.y_cond:
                x[:, 0] = y_cond.view(N, self.width)
            else:
                x[:, 0] = self.start_token
        else:
            assert isinstance(x, t.cuda.LongTensor)
            assert (0 <= x).all() and (x < self.bins).all()
            x = self.x_emb(x)
        assert x.shape == (n_samples, 1, self.width)
        if x_cond.shape == (N, D, self.width):
            cond = x_cond[:, sample_t:sample_t + 1, :]
        else:
            cond = x_cond
        x = x + self.pos_emb()[sample_t:sample_t + 1] + cond  # Pos emb, dropout is identity at eval time
        assert x.shape == (n_samples, 1, self.width)
        return x, cond

    def sample(self, n_samples, x_cond=None, y_cond=None, encoder_kv=None, fp16=False, temp=1.0, top_k=0, top_p=0.0,
               get_preds=False, sample_tokens=None):
        assert self.training == False

        if sample_tokens is None: sample_tokens=self.input_dims//4
        N, D = n_samples, self.input_dims
        if self.y_cond:
            assert y_cond is not None
            assert y_cond.shape == (N, 1, self.width)
        else:
            assert y_cond is None

        if self.x_cond:
            assert x_cond is not None
            assert x_cond.shape == (N, D, self.width) or x_cond.shape == (N, 1, self.width), f"Got {x_cond.shape}, expected ({N}, {D}/{1}, {self.width})"
        else:
            assert x_cond is None
            x_cond = t.zeros((N, 1, self.width), dtype=t.float).cuda()

        with t.no_grad():
            xs, x_emb, x = [], [], None
            if get_preds:
                preds = []
            for sample_t in get_range(range(0, sample_tokens)):
                x, cond = self.get_emb(sample_t, n_samples, x, x_cond, y_cond)
                #self.transformer.check_cache(n_samples, sample_t, fp16)
                x_emb.append(x)
                x = t.cat(x_emb, dim=1)
                triangular_mask = TriangularCausalMask(x.shape[1], device=x.device)

                x = self.transformer(x, attn_mask=triangular_mask)[:,-1:,:] # Transformer
                
                if self.add_cond_after_transformer:
                    x = x + cond
                assert x.shape == (n_samples, 1, self.width)
                x = self.x_out(x) # Predictions
                if get_preds:
                    preds.append(x.clone())
                # Adjust logits
                x = x / temp
                x = filter_logits(x, top_k=top_k, top_p=top_p)
                x = t.distributions.Categorical(logits=x).sample() # Sample and replace x
                assert x.shape == (n_samples, 1)
                xs.append(x.clone())

            del x, x_emb
            
            #self.transformer.del_cache()

            x = t.cat(xs, dim=1)
            if get_preds:
                preds = t.cat(preds, dim=1)
            x = self.postprocess(x, sample_tokens)
        if get_preds:
            return x, preds
        else:
            return x  
    
    def sample_recurrent(self, n_samples, x_cond=None, y_cond=None, encoder_kv=None, fp16=False, temp=1.0, top_k=0, top_p=0.0,
               get_preds=False, sample_tokens=None):
        assert self.training == False
        memory = None

        if sample_tokens is None: sample_tokens=self.input_dims
        N, D = n_samples, self.input_dims
        if self.y_cond:
            assert y_cond is not None
            assert y_cond.shape == (N, 1, self.width)
        else:
            assert y_cond is None

        if self.x_cond:
            assert x_cond is not None
            assert x_cond.shape == (N, D, self.width) or x_cond.shape == (N, 1, self.width), f"Got {x_cond.shape}, expected ({N}, {D}/{1}, {self.width})"
        else:
            assert x_cond is None
            x_cond = t.zeros((N, 1, self.width), dtype=t.float).cuda()

        with t.no_grad():
            xs, x = [], None
            if get_preds:
                preds = []
            for sample_t in get_range(range(0, sample_tokens)):
                x, cond = self.get_emb(sample_t, n_samples, x, x_cond, y_cond)
                #self.transformer.check_cache(n_samples, sample_t, fp16)
                x, memory = self.transformer(x[:,-1,:], memory) # Transformer
                x = t.unsqueeze(x,1)
                if self.add_cond_after_transformer:
                    x = x + cond
                assert x.shape == (n_samples, 1, self.width)
                x = self.x_out(x) # Predictions
                if get_preds:
                    preds.append(x.clone())
                # Adjust logits
                x = x / temp
                #x = filter_logits(x, top_k=top_k, top_p=top_p)
                x = sample_mol(x[:,-1,:], num_classes=self.bins) # Sample and replace x
                assert x.shape == (n_samples, 1)
                xs.append(x.clone())

            del x
            #self.transformer.del_cache()

            x = t.cat(xs, dim=1)
            if get_preds:
                preds = t.cat(preds, dim=1)
            x = self.postprocess(x, sample_tokens)
        if get_preds:
            return x, preds
        else:
            return x, memory

    def primed_sample(self, n_samples, x, memory, x_cond=None, y_cond=None, encoder_kv=None, fp16=False, temp=1.0, top_k=0,
                      top_p=0.0, get_preds=False, chunk_size=None, sample_tokens=None):
        assert self.training == False
        
        if sample_tokens is None: sample_tokens=self.input_dims
        # Preprocess.
        with t.no_grad():
            x = self.preprocess(x)
        assert isinstance(x, t.cuda.LongTensor)
        assert (0 <= x).all() and (x < self.bins).all()
        assert x.shape[0] == n_samples
        xs = t.split(x, 1, dim=1)
        xs = list(xs)
        assert len(xs) < sample_tokens
        

        N, D = n_samples, self.input_dims
        if self.y_cond:
            assert y_cond is not None
            assert y_cond.shape == (N, 1, self.width)
        else:
            assert y_cond is None

        if self.x_cond:
            assert x_cond is not None
            assert x_cond.shape == (N, D, self.width) or x_cond.shape == (N, 1, self.width), f"Got {x_cond.shape}, expected ({N}, {D}/{1}, {self.width})"
        else:
            assert x_cond is None
            x_cond = t.zeros((N, 1, self.width), dtype=t.float).cuda()

        with t.no_grad():
            if get_preds:
                preds = []

            x = xs[-1]
            assert x.shape == (n_samples, 1)
            empty_cache()
            for sample_t in get_range(range(len(xs), sample_tokens)):
                x, cond = self.get_emb(sample_t, n_samples, x, x_cond, y_cond)
                #self.transformer.check_cache(n_samples, sample_t, fp16)
                x, memory = self.transformer(x[:,-1,:], memory) # Transformer
                x = t.unsqueeze(x,1)
                if self.add_cond_after_transformer:
                    x = x + cond
                assert x.shape == (n_samples, 1, self.width)
                x = self.x_out(x) # Predictions
                if get_preds:
                    preds.append(x)
                # Adjust logits
                x = x / temp
                #x = filter_logits(x, top_k=top_k, top_p=top_p)
                #x = t.distributions.Categorical(logits=x).sample() # Sample and replace x
                x = sample_mol(x[:,-1,:], num_classes=self.bins)
                assert x.shape == (n_samples, 1)
                xs.append(x.clone())

            del x
            #self.transformer.del_cache()

            x = t.cat(xs, dim=1)
            if get_preds:
                preds = t.cat(preds, dim=1)
            x = self.postprocess(x, sample_tokens)
        if get_preds:
            return x, preds
        else:
            return x, memory

    def check_sample(self, chunk_size):
        bs, l, d = (4, self.input_dims, self.width)
        prime = int(self.input_dims//8*7)
        enc_l = self.encoder_dims
        with t.no_grad():
            y_cond = t.randn(bs, 1, d).cuda() if self.y_cond else None
            x_cond = t.randn(bs, l, d).cuda() if self.x_cond else None
            encoder_kv = t.randn(bs, enc_l, d).cuda()

            x, preds_sample = self.sample(bs, x_cond, y_cond, encoder_kv, get_preds=True)
            loss, preds_forw = self.forward(x, x_cond, y_cond, encoder_kv, get_preds=True)
            max_err = t.max(t.abs(preds_sample - preds_forw))
            assert max_err <= 1e-6, f"Max err is {max_err} {[i for i in range(l) if t.max(t.abs(preds_sample - preds_forw)[:, i, :]) > 1e-6]}"

            x_prime = x.view(bs, -1)[:,:prime]
            # unchunked
            x, preds_sample = self.primed_sample(bs, x_prime.clone(), x_cond, y_cond, encoder_kv, get_preds=True)
            assert (x.view(bs, -1)[:,:prime] == x_prime).all(), "Priming samples don't match"
            loss, preds_forw = self.forward(x, x_cond, y_cond, encoder_kv, get_preds=True)
            max_err = t.max(t.abs(preds_sample - preds_forw))
            assert max_err <= 1e-6, f"Max err is {max_err} {[i for i in range(l) if t.max(t.abs(preds_sample - preds_forw)[:, i, :]) > 1e-6]}"

            # chunked
            x, preds_sample = self.primed_sample(bs, x_prime.clone(), x_cond, y_cond, encoder_kv, get_preds=True, chunk_size=chunk_size)
            assert (x.view(bs, -1)[:,:prime] == x_prime).all(), "Priming samples don't match"
            loss, preds_forw = self.forward(x, x_cond, y_cond, encoder_kv, get_preds=True)
            max_err = t.max(t.abs(preds_sample - preds_forw))
            assert max_err <= 1e-6, f"Max err is {max_err} {[i for i in range(l) if t.max(t.abs(preds_sample - preds_forw)[:, i, :]) > 1e-6]}"


def test_prior(input_shape, encoder_dims, blocks, heads, chunk_size):
    bins = 512
    width = 32
    depth = 2
    prime_len = encoder_dims
    for x_cond in [True, False]:
        for y_cond in [True, False]:
            for attn_order in [0,2,6,12]:
                prior = ConditionalAutoregressive2D(input_shape, bins,
                                                    width=width, depth=depth, heads=heads,
                                                    attn_order=attn_order, blocks=blocks,
                                                    x_cond=x_cond, y_cond=y_cond,
                                                    encoder_dims=encoder_dims, prime_len=prime_len).cuda()
                prior.training = False
                prior.check_sample(chunk_size)
                print(f"Checked x_cond: {x_cond}, y_cond: {y_cond}, attn_order: {attn_order}")
            # prior.apply(_convert_mlp_traced)
            # prior.check_sample()
            # print(f"Checked traced x_cond: {x_cond}, y_cond: {y_cond}")


if __name__ == '__main__':
    from jukebox.utils.dist_utils import setup_dist_from_mpi
    setup_dist_from_mpi(port=29600)
    test_cases = [
        ((6144,), 384, 64, 2, 23),
        ((6144,), 384, 64, 2, 8),
        ((8192,), 512, 128, 2, 16),
    ]
    for test_case in test_cases:
        test_prior(*test_case)
