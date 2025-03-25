from torch.utils.data import DataLoader
import itertools
import matplotlib.pyplot as plt
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import os
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from collections import defaultdict

#############################################
# Utility: Configuration Node
#############################################


class CfgNode:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def to_dict(self):
        return self.__dict__

    def merge_from_dict(self, d):
        self.__dict__.update(d)


def save(path, model):
    torch.save(model.state_dict(), path)


def load(path, model, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(path, map_location=device))
#############################################
# Basic and Bio-Inspired Modules
#############################################


class NonlinearPlasticLayer(nn.Module):
    """
    A biologically inspired plastic layer that applies a linear transformation
    followed by a Naka–Rushton–like nonlinearity.

    Response:
      out = R_max * (activation^n / (activation^n + I50^n)) * (1 + plasticity_rate)
    """

    def __init__(self, in_features, out_features, R_max=1.0, I50=0.5, n=2.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.plasticity_rate = nn.Parameter(torch.zeros(1))
        self.R_max = R_max
        self.I50 = I50
        self.n = n

    def forward(self, x):
        linear_out = self.linear(x)
        activation = F.softplus(linear_out)
        epsilon = 1e-6
        activation_n = activation ** self.n
        denominator = activation_n + (self.I50 ** self.n) + epsilon
        nonlinear_output = self.R_max * (activation_n / denominator)
        return nonlinear_output * (1 + self.plasticity_rate)


class HomeostaticNorm(nn.Module):
    """Normalizes activations to a target mean and std."""

    def __init__(self, target_mean=0.0, target_std=1.0):
        super().__init__()
        self.target_mean = target_mean
        self.target_std = target_std

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return (x - mean) / (std + 1e-6) * self.target_std + self.target_mean


class FeedbackRNN(nn.Module):
    """
    Gated feedback module using a GRU combined with a neuromodulatory gate.
    Simulates dopamine-like modulation to regulate feedback strength.
    """

    def __init__(self, n_embd, hidden_size=None):
        super().__init__()
        hidden_size = hidden_size or n_embd
        self.gru = nn.GRU(input_size=n_embd,
                          hidden_size=hidden_size, batch_first=True)
        self.modulator = nn.Linear(n_embd, n_embd)

    def forward(self, x, hidden=None):
        out, hidden = self.gru(x, hidden)
        gate = torch.sigmoid(self.modulator(x))
        return out * gate, hidden

#############################################
# Additional Bio-Inspired Modules
#############################################


class DendriticIntegration(nn.Module):
    """
    Simulates dendritic computation via multiple nonlinear branches and integration.
    """

    def __init__(self, n_embd, num_branches=3):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Linear(n_embd, n_embd), nn.ReLU())
            for _ in range(num_branches)
        ])
        self.integrator = nn.Linear(n_embd * num_branches, n_embd)

    def forward(self, x):
        branch_outs = [branch(x) for branch in self.branches]
        combined = torch.cat(branch_outs, dim=-1)
        return self.integrator(combined)


class RefinedDynamicNeuromodulator(nn.Module):
    """
    Enhances neuromodulation by combining temporal convolution (STDP window)
    with a meta-plasticity gain.
    """

    def __init__(self, n_embd, context_dim=None, kernel_size=3):
        super().__init__()
        self.context_dim = context_dim
        input_dim = n_embd + context_dim if context_dim is not None else n_embd
        self.fc1 = nn.Linear(input_dim, n_embd)
        self.fc2 = nn.Linear(n_embd, n_embd)
        self.fc3 = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(0.1)
        self.temporal_conv = nn.Conv1d(
            1, 1, kernel_size=kernel_size, padding=kernel_size//2)
        self.stdp_factor = nn.Parameter(torch.tensor(1.0))
        self.meta_gain = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, context=None, prev_activation=None):
        x_input = torch.cat(
            [x, context], dim=-1) if (self.context_dim is not None and context is not None) else x
        x_mod = F.silu(self.fc1(x_input))
        x_mod = self.dropout(x_mod)
        x_mod = F.silu(self.fc2(x_mod))
        x_mod = self.dropout(x_mod)
        gating = torch.sigmoid(self.fc3(x_mod))
        if prev_activation is not None:
            delta = torch.abs(x - prev_activation)  # [B, T, n_embd]
            B, T, D = delta.shape
            delta_reshaped = delta.transpose(1, 2).reshape(B * D, 1, T)
            temporal_effect = self.temporal_conv(delta_reshaped)
            temporal_effect = temporal_effect.view(B, D, T).transpose(1, 2)
            stdp_term = torch.exp(-temporal_effect) * self.stdp_factor
            gating = gating * stdp_term
        gating = gating * self.meta_gain
        return gating


def adaptive_lateral_inhibition(x, k_ratio=0.5):
    """
    Applies adaptive lateral inhibition by retaining top k activations and normalizing.
    """
    k = max(1, int(x.size(-1) * k_ratio))
    topk_values, _ = torch.topk(x, k, dim=-1)
    threshold = topk_values[..., -1].unsqueeze(-1)
    inhibited = torch.where(x >= threshold, x, torch.zeros_like(x))
    norm_factor = inhibited.sum(dim=-1, keepdim=True) + 1e-6
    return inhibited / norm_factor


class OscillatoryModulator(nn.Module):
    """
    Implements a Gabor-like modulation: cosine modulated by a Gaussian envelope.
    """

    def __init__(self, n_embd):
        super().__init__()
        self.amplitude = nn.Parameter(torch.ones(1))
        self.frequency = nn.Parameter(torch.ones(1) * 0.5)
        self.phase = nn.Parameter(torch.zeros(1))
        self.mu = nn.Parameter(torch.zeros(1))
        self.sigma = nn.Parameter(torch.ones(1))
        self.fc = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        mod = self.amplitude * \
            torch.exp(-((x - self.mu)**2) / (2 * self.sigma**2))
        mod *= torch.cos(2 * math.pi * self.frequency *
                         (x - self.mu) + self.phase)
        return self.fc(mod)


class BioPositionalEncoding(nn.Module):
    """
    Hierarchical, learnable positional encoding simulating multi-scale organization.
    """

    def __init__(self, d_model, max_len=512, num_scales=4):
        super().__init__()
        self.num_scales = num_scales
        self.scale_embeddings = nn.ParameterList(
            [nn.Parameter(torch.randn(max_len, d_model)) for _ in range(num_scales)])
        self.aggregator = nn.Linear(num_scales * d_model, d_model)

    def forward(self, pos_ids):
        embeddings = [emb[pos_ids] for emb in self.scale_embeddings]
        combined = torch.cat(embeddings, dim=-1)
        return self.aggregator(combined)

#############################################
# Frequency-Domain Multi-Head Latent Self-Attention
#############################################


class FreqDomainLatentSelfAttention(nn.Module):
    """
    Multi-head latent self-attention with FFT-based adaptive scaling, relative bias,
    latent projection and fusion.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "Embedding dimension must be divisible by number of heads."
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.relative_bias = nn.Parameter(torch.zeros(
            self.n_head, config.block_size, config.block_size))
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.latent_proj = nn.Linear(self.head_dim, self.head_dim)
        self.fuse_layer = nn.Linear(2 * self.head_dim, self.head_dim)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = torch.split(qkv, C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        qk = torch.stack([q, k], dim=0)
        qk_fft = torch.fft.rfft(qk.float(), dim=3)
        q_fft, k_fft = qk_fft[0], qk_fft[1]
        freqs = torch.fft.rfftfreq(T, d=1.0).to(x.device)
        scaling_factors = torch.exp(-torch.abs(freqs) * T * self.alpha)
        scaling_factors = scaling_factors.view(1, 1, -1, 1)
        q_fft_mod = q_fft * scaling_factors
        k_fft_mod = k_fft * scaling_factors
        q_mod = torch.fft.irfft(q_fft_mod, n=T, dim=2)
        k_mod = torch.fft.irfft(k_fft_mod, n=T, dim=2)
        att_scores = torch.matmul(q_mod, k_mod.transpose(-2, -1)) * self.scale
        att_scores = att_scores + self.relative_bias[:, :T, :T].unsqueeze(0)
        att_weights = F.softmax(att_scores, dim=-1)
        att_weights = self.attn_dropout(att_weights)
        local_out = torch.matmul(att_weights, v)
        latent = v.mean(dim=2)
        latent = self.latent_proj(latent)
        latent_expanded = latent.unsqueeze(2).expand(-1, -1, T, -1)
        fused = torch.cat([local_out, latent_expanded], dim=-1)
        fused = self.fuse_layer(fused)
        fused = fused.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.c_proj(fused))
        return out

#############################################
# Attention-Based Fusion Layer
#############################################


class AttentionFusion(nn.Module):
    """
    Fuses low-level and high-level representations using an attention mechanism.
    Computes query from low-level and key/value from high-level to produce a weighted sum.
    """

    def __init__(self, n_embd):
        super().__init__()
        self.query = nn.Linear(n_embd, n_embd)
        self.key = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.out_proj = nn.Linear(n_embd, n_embd)

    def forward(self, low, high):
        Q = self.query(low)
        K = self.key(high)
        V = self.value(high)
        attn_scores = torch.matmul(
            Q, K.transpose(-2, -1)) / math.sqrt(low.size(-1))
        attn_weights = F.softmax(attn_scores, dim=-1)
        fused = torch.matmul(attn_weights, V)
        fused = self.out_proj(fused)
        return fused

#############################################
# Enhanced Transformer Block with Improved Feedback & Plasticity
#############################################


class EnhancedBlock(nn.Module):
    def __init__(self, config, use_feedback=True):
        super().__init__()
        self.layerdrop = getattr(config, 'layerdrop', 0.0)
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = FreqDomainLatentSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            NonlinearPlasticLayer(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            NonlinearPlasticLayer(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop)
        )
        self.feedback_proj = nn.Linear(config.n_embd, config.n_embd)
        self.dendritic = DendriticIntegration(config.n_embd, num_branches=3)
        self.dynamic_neuromod = RefinedDynamicNeuromodulator(
            config.n_embd, context_dim=config.n_embd, kernel_size=3)
        self.oscillatory_mod = OscillatoryModulator(config.n_embd)
        self.homeostatic_norm = HomeostaticNorm()
        self.use_feedback = use_feedback
        if self.use_feedback:
            self.feedback_rnn = FeedbackRNN(config.n_embd)

    def forward(self, x, feedback=None, feedback_hidden=None, prev_activation=None):
        attn_input = self.ln_1(x)
        if feedback is not None:
            attn_input = attn_input + feedback
        x_attn = self.attn(attn_input)
        x = x + x_attn
        dendritic_out = self.dendritic(x)
        x = x + dendritic_out
        if self.use_feedback and feedback is not None:
            feedback_processed, feedback_hidden = self.feedback_rnn(
                feedback, feedback_hidden)
        else:
            feedback_processed = torch.zeros_like(x)
        mlp_input = self.ln_2(x)
        mlp_out = self.mlp(mlp_input)
        gating = self.dynamic_neuromod(
            mlp_input, context=feedback_processed, prev_activation=prev_activation)
        mlp_out = mlp_out * gating
        oscillation = self.oscillatory_mod(mlp_input)
        mlp_out = mlp_out + oscillation
        mlp_out = adaptive_lateral_inhibition(mlp_out, k_ratio=0.5)
        mlp_out = self.homeostatic_norm(mlp_out)
        x = x + mlp_out
        block_feedback = self.feedback_proj(x)
        return x, block_feedback, feedback_hidden

#############################################
# Hierarchical Enhanced GPT Model with Two Levels and Attention Fusion
#############################################


class HierarchicalEnhancedGPT(nn.Module):
    @staticmethod
    def get_default_config():
        C = CfgNode()
        C.model_type = 'gpt'
        C.n_layer = None
        C.n_head = None
        C.n_embd = None
        C.vocab_size = None
        C.block_size = None
        C.embd_pdrop = 0.15
        C.resid_pdrop = 0.20
        C.attn_pdrop = 0.20
        C.layerdrop = 0.1
        C.label_smoothing = 0.1
        C.low_level_ratio = 0.6
        return C

    def __init__(self, config):
        super().__init__()
        self.config = config
        # Use a dictionary of default configurations based on model type.
        default_configs = {
            'openai-gpt':   dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600),
            'gopher-44m':   dict(n_layer=8, n_head=16, n_embd=512),
            'gpt-mini':     dict(n_layer=6, n_head=6, n_embd=192),
            'gpt-micro':    dict(n_layer=4, n_head=4, n_embd=128),
            'gpt-nano':     dict(n_layer=3, n_head=3, n_embd=48),
        }
        # If essential parameters are missing, fill them in with defaults.
        if config.n_layer is None or config.n_head is None or config.n_embd is None:
            if config.model_type in default_configs:
                config.merge_from_dict(default_configs[config.model_type])
                print("Using default configuration for model type:",
                      config.model_type)
                print("Configuration:\n", config.to_dict())
            else:
                raise ValueError(
                    f"Model type {config.model_type} not recognized.")
        self.block_size = config.block_size
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        low_ratio = config.low_level_ratio if hasattr(
            config, "low_level_ratio") else 0.6
        n_low = math.ceil(config.n_layer * low_ratio)
        n_high = config.n_layer - n_low
        self.low_blocks = nn.ModuleList(
            [EnhancedBlock(config, use_feedback=True) for _ in range(n_low)]
        )
        self.high_blocks = nn.ModuleList(
            [EnhancedBlock(config, use_feedback=True) for _ in range(n_high)]
        )
        # Use attention-based fusion to integrate low and high-level outputs.
        self.attn_fusion = AttentionFusion(config.n_embd)
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = BioPositionalEncoding(
            config.n_embd, max_len=config.block_size, num_scales=4
        )
        self.emb_drop = nn.Dropout(config.embd_pdrop)
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        n_params = sum(p.numel() for p in self.parameters())
        print("Number of parameters: %.2fM" % (n_params / 1e6))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_emb = self.wte(idx)
        pos_emb = self.pos_emb(torch.arange(
            T, device=idx.device).unsqueeze(0).expand(B, T))
        x = self.emb_drop(token_emb + pos_emb)
        feedback = torch.zeros_like(x, device=x.device)
        feedback_hidden = None
        prev_activation = None
        # Process low-level blocks.
        x_low = x
        for block in self.low_blocks:
            x_low, feedback, feedback_hidden = block(
                x_low, feedback, feedback_hidden, prev_activation)
            prev_activation = x_low.detach()
        # Process high-level blocks.
        x_high = x_low
        for block in self.high_blocks:
            x_high, feedback, feedback_hidden = block(
                x_high, feedback, feedback_hidden, prev_activation)
            prev_activation = x_high.detach()
        # Fuse low and high-level outputs using attention-based fusion.
        x_merge = self.attn_fusion(x_low, x_high)
        x_out = self.ln_f(x_merge)
        logits = self.lm_head(x_out)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1),
                                   label_smoothing=self.config.label_smoothing)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(
                1) <= self.block_size else idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

#############################################
# Trainer Class with Improved LR Regulation and Specialized Updates
#############################################


class Trainer:
    @staticmethod
    def get_default_config():
        C = CfgNode(
            device='auto',
            num_workers=4,
            max_iters=7500,
            batch_size=16,
            learning_rate=8e-5,
            betas=(0.9, 0.98),
            weight_decay=0.05,
            grad_norm_clip=1.0,
            use_amp=True,
            checkpoint_path='./checkpoints',
            save_interval=1,
            lr_scheduler='linear_warmup_decay',
            warmup_iters=6000,
            decay_iters=300000,
            min_lr=1e-6,
            early_stopping_patience=10,
            gradient_accumulation_steps=2
        )
        return C

    def __init__(self, config, model, train_dataset, val_dataset=None):
        # Assign key attributes
        self.config = config
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        print(
            f"Trainer initialized with model: {type(model)} and train dataset: {type(train_dataset)}")
        if config.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = config.device
        self.model.to(self.device)
        print("Training on device:", self.device)

        # Specialized parameter grouping:
        low_params = list(self.model.low_blocks.parameters())
        high_params = list(self.model.high_blocks.parameters())
        other_params = list(self.model.wte.parameters()) + list(self.model.pos_emb.parameters()) + \
            list(self.model.emb_drop.parameters()) + list(self.model.ln_f.parameters()) + \
            list(self.model.lm_head.parameters())
        optimizer_groups = [
            {"params": low_params, "lr": self.config.learning_rate,
             "group_type": "low", "initial_lr": self.config.learning_rate},
            {"params": high_params, "lr": self.config.learning_rate * 0.8,
             "group_type": "high", "initial_lr": self.config.learning_rate * 0.8},
            {"params": other_params, "lr": self.config.learning_rate,
             "group_type": "base", "initial_lr": self.config.learning_rate}
        ]
        self.optimizer = torch.optim.AdamW(
            optimizer_groups,
            betas=self.config.betas,
            weight_decay=self.config.weight_decay
        )
        if self.config.lr_scheduler == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config.decay_iters, eta_min=self.config.min_lr
            )
        elif self.config.lr_scheduler == 'plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=2
            )
        elif self.config.lr_scheduler == 'linear_warmup_decay':
            def lr_lambda(step):
                if step < self.config.warmup_iters:
                    return step / self.config.warmup_iters
                else:
                    decay = max(self.config.min_lr / self.config.learning_rate,
                                1 - (step - self.config.warmup_iters) / self.config.decay_iters)
                    return decay
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda
            )
        else:
            self.scheduler = None
        self.scaler = torch.amp.GradScaler(
            device='cuda', enabled=self.config.use_amp)
        self.checkpoint_path = self.config.checkpoint_path
        os.makedirs(self.checkpoint_path, exist_ok=True)
        self.best_val_loss = float('inf')
        self.no_improvement_epochs = 0
        self.global_step = 0
        self.current_epoch = 0
        self.iter_num = 0
        self.iter_time = time.time()

    def save_checkpoint(self, epoch, val_loss=None):
        # Save the model checkpoint along with optimizer and scaler states.
        if val_loss is None:
            ckpt_name = f'epoch{epoch:03d}.pt'
        else:
            ckpt_name = f'epoch{epoch:03d}_valloss{val_loss:.4f}.pt'
        ckpt_path = os.path.join(self.checkpoint_path, ckpt_name)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'global_step': self.global_step
        }, ckpt_path)
        print(f"Checkpoint saved at {ckpt_path}")

    def update_specialized_learning_rates(self, epoch, val_loss):
        # Update learning rates based on epoch instead of global_step.
        for group in self.optimizer.param_groups:
            if group.get("group_type", "base") == "high":
                new_lr = group["initial_lr"] * (0.98 ** epoch)
                group["lr"] = max(new_lr, self.config.min_lr)
            elif group.get("group_type", "base") == "low":
                new_lr = group["initial_lr"] * (0.99 ** epoch)
                group["lr"] = max(new_lr, self.config.min_lr)
            else:  # For base group, apply a similar decay or leave it to the scheduler
                new_lr = group["initial_lr"] * (1 - (
                    epoch * self.config.max_iters - self.config.warmup_iters) / self.config.decay_iters)
                group["lr"] = max(new_lr, self.config.min_lr)

    def log_lr_and_gradients(self):
        # Log the effective learning rates and average gradient norms for each parameter group.
        print("----- Diagnostic Logging -----")
        print("Effective Learning Rates:")
        for i, group in enumerate(self.optimizer.param_groups):
            lr = group["lr"]
            group_type = group.get("group_type", "base")
            print(f"  Group {i} ({group_type}): lr = {lr:.2e}")
        print("Average Gradient Norms:")
        for i, group in enumerate(self.optimizer.param_groups):
            norms = [p.grad.data.norm(2).item()
                     for p in group["params"] if p.grad is not None]
            avg_norm = sum(norms) / len(norms) if norms else 0.0
            group_type = group.get("group_type", "base")
            print(
                f"  Group {i} ({group_type}): avg grad norm = {avg_norm:.2e}")
        print("------------------------------")

    def run(self, num_epochs=50):
        model, config = self.model, self.config
        train_loader = DataLoader(
            self.train_dataset,
            sampler=torch.utils.data.RandomSampler(
                self.train_dataset, replacement=True, num_samples=int(1e10)
            ),
            shuffle=False,
            pin_memory=True,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )
        train_loss_history = []
        val_loss_history = []
        iter_per_epoch = config.max_iters  # 7500 iterations per epoch
        clip_dict = {'base': 1.0, 'neuromod': 0.5,
                     'plastic': 1.2, 'oscillatory': 0.5}
        grad_accum_steps = config.gradient_accumulation_steps if hasattr(
            config, 'gradient_accumulation_steps') else 1

        for epoch in range(num_epochs):
            print(f"Starting Epoch {epoch + 1}")
            model.train()
            epoch_loss = 0.0
            epoch_start_time = time.time()
            data_iter = iter(train_loader)
            self.optimizer.zero_grad(set_to_none=True)
            with tqdm(total=iter_per_epoch, desc=f"Epoch {epoch + 1}/{num_epochs}", unit='batch', ncols=100) as pbar:
                for i in range(iter_per_epoch):
                    try:
                        batch = next(data_iter)
                    except StopIteration:
                        data_iter = iter(train_loader)
                        batch = next(data_iter)
                    batch = [t.to(self.device) for t in batch]
                    x, y = batch
                    with torch.amp.autocast(device_type='cuda', enabled=config.use_amp):
                        logits, loss = model(x, y)
                    loss = loss / grad_accum_steps  # Normalize loss for accumulation
                    self.scaler.scale(loss).backward()
                    epoch_loss += loss.item() * grad_accum_steps
                    if (i + 1) % grad_accum_steps == 0:
                        self.scaler.unscale_(self.optimizer)
                        for group in self.optimizer.param_groups:
                            torch.nn.utils.clip_grad_norm_(group["params"],
                                                           clip_dict.get(group.get("group_type", "base"), 1.0))
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad(set_to_none=True)
                    if hasattr(self, 'trigger_callbacks'):
                        self.trigger_callbacks('on_batch_end')
                    self.iter_num += 1
                    self.global_step += 1
                    tnow = time.time()
                    self.iter_dt = tnow - self.iter_time
                    self.iter_time = tnow
                    pbar.set_postfix({'loss': f'{epoch_loss / (i + 1):.3f}'})
                    pbar.update(1)
            avg_train_loss = epoch_loss / iter_per_epoch
            train_loss_history.append(avg_train_loss)
            train_time = time.time() - epoch_start_time

            avg_val_loss = None
            if self.val_dataset is not None:
                model.eval()
                val_loss = 0.0
                num_batches = 0
                val_loader = DataLoader(
                    self.val_dataset,
                    batch_size=config.batch_size,
                    shuffle=False,
                    num_workers=config.num_workers,
                    pin_memory=True
                )
                with torch.no_grad():
                    for batch in val_loader:
                        batch = [t.to(self.device) for t in batch]
                        x_val, y_val = batch
                        _, loss_val = model(x_val, y_val)
                        val_loss += loss_val.item()
                        num_batches += 1
                avg_val_loss = val_loss / num_batches if num_batches > 0 else 0.0
                val_loss_history.append(avg_val_loss)
                model.train()
                print(f"\nEpoch {epoch + 1}/{num_epochs} in {self._format_time(train_time)}: "
                      f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            else:
                print(f"\nEpoch {epoch + 1}/{num_epochs} in {self._format_time(train_time)}: "
                      f"Train Loss: {avg_train_loss:.4f}")

            # Update learning rates per epoch
            if self.val_dataset is not None:
                if avg_val_loss < self.best_val_loss:
                    self.best_val_loss = avg_val_loss
                    self.no_improvement_epochs = 0
                    self.save_checkpoint(epoch + 1, avg_val_loss)
                else:
                    self.no_improvement_epochs += 1
                    if self.no_improvement_epochs >= config.early_stopping_patience:
                        print("Early stopping triggered.")
                        break
                if hasattr(self, 'update_specialized_learning_rates'):
                    self.update_specialized_learning_rates(epoch, avg_val_loss)
            else:
                if (epoch + 1) % config.save_interval == 0:
                    self.save_checkpoint(epoch + 1)
            # Log diagnostic information
            self.log_lr_and_gradients()

            self.current_epoch = epoch + 1
            torch.cuda.empty_cache()
            gc.collect()

        plt.figure(figsize=(10, 6))
        epochs_range = range(1, len(train_loss_history) + 1)
        plt.plot(epochs_range, train_loss_history,
                 label='Training Loss', marker='o')
        if self.val_dataset is not None:
            plt.plot(epochs_range, val_loss_history,
                     label='Validation Loss', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss over Epochs')
        plt.legend()
        plt.grid(True)
        plt.show(block=False)

    def _format_time(self, seconds):
        m, s = divmod(int(seconds), 60)
        return f"{m:02d}:{s:02d}"
