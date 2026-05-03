# model.py — GPCAHazDesNet 
#
# GPCAHazDesNet: Gaussian Process Channel Attention Haze Detection Network
#
# Architecture overview:
#   Input image
#     └─ MultiScaleStem          (multi-receptive-field feature extraction)
#         └─ EncoderStage x3     (progressive downsampling + GPCA attention)
#             └─ Bottleneck      (deepest feature representation)
#                 └─ DecoderBlock x3  (skip-connection upsampling)
#                     └─ MapRefinement → haze_map  (spatial haze density)
#                     └─ Classifier   → class_logits (Clear/Low/Moderate/Heavy)


import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════
#  BUILDING BLOCKS
# ══════════════════════════════════════════════

class ConvBNAct(nn.Module):
    """
    Basic Conv → BatchNorm → Activation block used throughout the network.

    Using bias=False in Conv because BatchNorm already handles the bias term,
    so including it would be redundant and waste parameters.

    Supports relu / gelu / silu / none activations to allow flexibility
    at different stages (e.g. residual branches need 'none' before the skip add).
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        activation: str = "relu",
    ):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels,
                      kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
        ]
        if activation == "relu":
            layers.append(nn.ReLU(inplace=True))
        elif activation == "gelu":
            layers.append(nn.GELU())
        elif activation == "silu":
            layers.append(nn.SiLU())
        elif activation == "none":
            # No activation — used before residual additions to avoid
            # double-activating the identity path
            pass
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    """
    Standard post-activation residual block: act(F(x) + x).

    Two 3×3 convolutions with a skip connection. The second conv uses
    activation='none' so we apply the activation AFTER the residual add,
    which is the standard post-activation residual formulation.

    Identity shortcut (no projection) — works because in/out channels
    are always equal when this block is used.
    """
    def __init__(self, channels: int, activation: str = "relu"):
        super().__init__()
        self.conv1 = ConvBNAct(channels, channels, 3, 1, 1, activation=activation)
        # Second conv has no activation; activation applied after the residual add below
        self.conv2 = ConvBNAct(channels, channels, 3, 1, 1, activation="none")
        if activation == "relu":
            self.act = nn.ReLU(inplace=True)
        elif activation == "gelu":
            self.act = nn.GELU()
        else:
            self.act = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply activation after adding residual so the skip path is clean
        return self.act(x + self.conv2(self.conv1(x)))


# ══════════════════════════════════════════════
#  MULTI-SCALE STEM
# ══════════════════════════════════════════════

class MultiScaleStem(nn.Module):
    """
    Multi-receptive-field stem that processes the input image at three scales
    simultaneously before fusing.

    Why three scales?
    - 3×3: captures fine local texture (edges, fine haze particles)
    - 5×5: captures medium-scale patterns (haze gradients)
    - 7×7: captures coarse-scale context (global haze distribution)

    All branches output base_channels feature maps, then a 1×1 conv
    fuses the concatenated 3*base_channels back down to base_channels.
    This is cheaper than a single large kernel while covering more context.
    """
    def __init__(self, in_channels: int = 3, base_channels: int = 32):
        super().__init__()
        self.branch3 = ConvBNAct(in_channels, base_channels, 3, 1, 1, activation="relu")
        self.branch5 = ConvBNAct(in_channels, base_channels, 5, 1, 2, activation="relu")
        self.branch7 = ConvBNAct(in_channels, base_channels, 7, 1, 3, activation="relu")
        # 1×1 fusion conv: merges multi-scale info with no spatial mixing
        self.fuse    = ConvBNAct(base_channels * 3, base_channels, 1, 1, 0, activation="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Run all three branches in parallel, cat along channel dim, fuse
        return self.fuse(torch.cat([
            self.branch3(x), self.branch5(x), self.branch7(x)
        ], dim=1))


# ══════════════════════════════════════════════
#  TRUE GPCA — Gaussian Process Channel Attention
# ══════════════════════════════════════════════

class TrueGPCA(nn.Module):
    """
    Gaussian Process Channel Attention (GPCA).

    Standard channel attention (SE-Net style) just learns a direct mapping
    from pooled features to channel weights. GPCA instead models the
    uncertainty over channel importance using a variational + GP framework:

      1. A small MLP encodes pooled features into a Gaussian distribution
         over channel importance (mu, logvar).
      2. A deterministic sigmoid-Gaussian approximation converts (mu, logvar)
         to a probability — avoiding train/inference mismatch from MC sampling.
      3. A sparse GP (inducing point approximation) smooths the channel
         importance estimates, capturing inter-channel correlations via an
         RBF kernel over learned channel embeddings.
      4. A final MLP + sigmoid produces the channel attention weights.

    The GP posterior enforces smooth, correlated channel weighting rather
    than treating each channel independently — important because haze-relevant
    channels are often correlated (e.g. blue-channel haze scattering).
    """
    def __init__(
        self,
        channels: int,
        hidden_ratio: float = 0.5,   # MLP bottleneck size relative to channels
        embed_dim: int = 32,         # dimensionality of the GP inducing point embeddings
        jitter: float = 1e-4,        # numerical stability jitter added to GP kernel diagonal
        mc_samples: int = 4,         # kept for API compatibility; sampling is disabled (see below)
        temperature: float = 1.0,    # unused currently; reserved for future logit scaling
    ):
        super().__init__()
        self.channels    = channels
        self.jitter      = jitter
        self.mc_samples  = mc_samples
        self.temperature = temperature

        hidden_dim = max(8, int(channels * hidden_ratio))

        # Variational encoder: pooled features → (mu, logvar) over channel importance
        self.pre         = nn.Sequential(nn.Linear(channels, hidden_dim), nn.ReLU(inplace=True))
        self.mu_head     = nn.Linear(hidden_dim, channels)
        self.logvar_head = nn.Linear(hidden_dim, channels)

        # GP inducing points: each of the `channels` has a learned embedding vector
        # in embed_dim-dimensional space. The RBF kernel is computed between these.
        self.inducing_embed  = nn.Parameter(torch.randn(channels, embed_dim) * 0.02)
        self.query_embed     = nn.Parameter(torch.randn(channels, embed_dim) * 0.02)

        # Learnable GP hyperparameters (log-parameterised for positivity)
        self.log_lengthscale = nn.Parameter(torch.zeros(1))          # RBF kernel lengthscale
        self.log_outputscale = nn.Parameter(torch.zeros(1))          # RBF kernel output scale
        self.log_noise       = nn.Parameter(torch.full((1,), -4.0))  # GP observation noise (small init)

        # Post-GP MLP: maps GP posterior mean back to channel attention weights
        self.post = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, channels),
        )

    def _rbf_kernel(self, xa, xb):
        """
        Computes the RBF (squared exponential) kernel matrix between xa and xb.
        K(a, b) = output_scale * exp(-0.5 * ||a - b||^2 / lengthscale^2)
        Both xa and xb have shape (channels, embed_dim).
        Returns a (channels, channels) kernel matrix.
        """
        diff  = xa.unsqueeze(1) - xb.unsqueeze(0)    # (C, C, embed_dim)
        dist2 = (diff ** 2).sum(dim=-1)               # (C, C) squared distances
        ell2  = torch.exp(2.0 * self.log_lengthscale) + 1e-12  # lengthscale^2
        return torch.exp(self.log_outputscale) * torch.exp(-0.5 * dist2 / ell2)

    def _sample_sigmoid_gaussian(self, mu, logvar):
        """
        Deterministic sigmoid-Gaussian approximation.

        Instead of Monte Carlo sampling z ~ N(mu, var) and applying sigmoid(z),
        we use the analytic approximation:
            E[sigmoid(z)] ≈ sigmoid(mu / sqrt(1 + pi/8 * var))

        This eliminates train/inference mismatch that would occur if we sampled
        during training but used the mean at inference. Always deterministic.
        """
        denom = torch.sqrt(1.0 + (math.pi / 8.0) * torch.exp(logvar))
        return torch.sigmoid(mu / denom)

    def _gp_posterior_mean(self, y):
        """
        Computes the GP posterior mean at query points given observations y.

        Uses the standard sparse GP posterior formula:
            alpha = (Kuu + noise * I)^{-1} y^T
            posterior_mean = Kuq^T @ alpha

        where Kuu is the kernel between inducing points and Kuq is the
        cross-kernel between inducing and query points.

        torch.linalg.solve is used instead of explicit matrix inversion for
        numerical stability (avoids computing and storing the full inverse).

        Args:
            y: (B, C) — channel importance estimates from the variational encoder
        Returns:
            (B, C) — GP-smoothed channel importance
        """
        Xu    = self.inducing_embed                    # (C, embed_dim)
        Xq    = self.query_embed                       # (C, embed_dim)
        Kuu   = self._rbf_kernel(Xu, Xu)              # (C, C)
        Kuq   = self._rbf_kernel(Xu, Xq)              # (C, C)
        noise = torch.exp(self.log_noise) + self.jitter
        eye   = torch.eye(self.channels, device=y.device, dtype=y.dtype)
        # Solve (Kuu + noise*I) @ alpha = y^T  →  alpha shape: (C, B)
        alpha = torch.linalg.solve(Kuu + noise * eye, y.T)
        # Posterior mean: (C, C)^T @ (C, B) → transpose → (B, C)
        return (Kuq.transpose(0, 1) @ alpha).T

    def forward(self, x, return_aux=False):
        """
        Args:
            x          : (B, C, H, W) feature map
            return_aux : if True, returns a dict of intermediates for debugging/visualisation

        Returns:
            out : (B, C, H, W) channel-attended feature map
            aux : dict with mu, logvar, channel_var, gp_mean, attention  [if return_aux]
        """
        b, c, _, _ = x.shape

        # Global average pool → (B, C): summarise spatial info per channel
        pooled      = F.adaptive_avg_pool2d(x, 1).flatten(1)

        # Variational encoder: pooled features → mu and logvar
        feat        = self.pre(pooled)
        mu          = self.mu_head(feat)
        logvar      = self.logvar_head(feat).clamp(-8.0, 4.0)  # clamp for numerical stability

        # Deterministic sigmoid-Gaussian: (B, C) channel importance in [0, 1]
        channel_var = self._sample_sigmoid_gaussian(mu, logvar)

        # GP posterior smoothing: captures inter-channel correlations
        gp_mean     = self._gp_posterior_mean(channel_var)

        # Final attention weights: sigmoid → [0,1], reshape for spatial broadcasting
        attn        = torch.sigmoid(self.post(gp_mean)).view(b, c, 1, 1)

        # Apply channel attention: rescale each channel of the feature map
        out         = x * attn

        if return_aux:
            return out, {"mu": mu, "logvar": logvar,
                         "channel_var": channel_var, "gp_mean": gp_mean, "attention": attn}
        return out


# ══════════════════════════════════════════════
#  ENCODER / DECODER
# ══════════════════════════════════════════════

class EncoderStage(nn.Module):
    """
    One encoder stage: stride-2 downsampling → 2× residual blocks → GPCA attention.

    GPCA is placed after the residual blocks (not before) so that attention
    operates on fully-processed features at this scale, not on the raw
    downsampled input.

    Produces skip-connection features that are reused by the corresponding DecoderBlock.
    """
    def __init__(self, in_channels, out_channels, gpca_embed_dim=32):
        super().__init__()
        # Stride-2 conv halves spatial resolution and increases channel depth
        self.down = ConvBNAct(in_channels, out_channels, 3, 2, 1, activation="relu")
        self.res1 = ResidualBlock(out_channels, activation="relu")
        self.res2 = ResidualBlock(out_channels, activation="relu")
        # Smaller embed_dim at early encoder stages (fewer channels → less GP capacity needed)
        self.gpca = TrueGPCA(out_channels, embed_dim=gpca_embed_dim)

    def forward(self, x):
        return self.gpca(self.res2(self.res1(self.down(x))))


class DecoderBlock(nn.Module):
    """
    One decoder stage: bilinear upsample → concat skip connection → 2× conv.

    Skip connections from the encoder restore fine-grained spatial detail
    lost during downsampling — critical for producing accurate spatial haze maps.

    Bilinear interpolation (not transposed conv) avoids checkerboard artifacts
    that transposed convolutions commonly introduce.
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        # First conv processes the concatenated (upsampled + skip) features together
        self.conv1 = ConvBNAct(in_channels + skip_channels, out_channels, 3, 1, 1, activation="relu")
        self.conv2 = ConvBNAct(out_channels, out_channels, 3, 1, 1, activation="relu")

    def forward(self, x, skip):
        # Upsample to match skip connection's spatial size (handles odd sizes gracefully)
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv2(self.conv1(torch.cat([x, skip], dim=1)))


class MapRefinement(nn.Module):
    """
    Lightweight head that converts decoder features to a single-channel haze density map.

    Two 3×3 convs refine spatial features before a 1×1 conv projects to 1 channel.
    Sigmoid is applied externally in the main forward pass (not here), so this
    module outputs raw logits.
    """
    def __init__(self, channels=32):
        super().__init__()
        self.refine = nn.Sequential(
            ConvBNAct(channels, channels, 3, 1, 1, activation="relu"),
            ConvBNAct(channels, channels, 3, 1, 1, activation="relu"),
            nn.Conv2d(channels, 1, kernel_size=1),  # project to single-channel density map
        )

    def forward(self, x):
        return self.refine(x)


# ══════════════════════════════════════════════
#  FOCAL LOSS
# ══════════════════════════════════════════════

class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017) — focuses training on hard/misclassified examples.

        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    The (1 - p_t)^gamma term down-weights easy examples (high p_t) and
    up-weights hard ones (low p_t). gamma=2 is the standard default.

    Why Focal Loss here?
    The Heavy haze class is both rare and visually hard to distinguish from Moderate.
    Standard cross-entropy treats all examples equally, so the model tends to
    over-optimise for the easy Clear/Low majority. Focal loss fixes this by
    concentrating gradient signal on the hard Heavy examples.

    Label smoothing (0.1) is also applied to prevent the model from becoming
    overconfident at ambiguous boundaries between adjacent haze levels.
    """
    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor = None,       # per-class weights, combined with focal term
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.gamma           = gamma
        self.weight          = weight
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(1)

        # --- Label smoothing ---
        # Distribute a small probability mass (label_smoothing / (C-1)) across
        # all non-target classes instead of using hard one-hot targets.
        # Prevents the model from driving logits to ±∞ for easy/clear samples.
        with torch.no_grad():
            smooth_targets = torch.full_like(logits, self.label_smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)

        log_prob = F.log_softmax(logits, dim=1)
        prob     = log_prob.exp()

        # Cross-entropy with smooth targets: -sum(q * log_p) over classes
        ce_loss  = -(smooth_targets * log_prob).sum(dim=1)

        # Focal weight: (1 - p_t)^gamma, where p_t = predicted prob of the TRUE class.
        # High p_t (easy/correct) → weight ≈ 0 → loss suppressed.
        # Low p_t (hard/wrong)    → weight ≈ 1 → full loss gradient.
        p_t      = prob.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_wt = (1.0 - p_t) ** self.gamma

        loss = focal_wt * ce_loss

        # Apply optional per-class weights (e.g. extra penalty for missing Heavy)
        if self.weight is not None:
            w    = self.weight.to(logits.device)
            loss = loss * w[targets]

        return loss.mean()


# ══════════════════════════════════════════════
#  MAIN MODEL — GPCAHazDesNet v3
# ══════════════════════════════════════════════

class GPCAHazDesNet(nn.Module):
    """
    GPCAHazDesNet v3 — Gaussian Process Channel Attention Haze Detection Network.

    A dual-output U-Net style architecture that simultaneously predicts:
      (a) global_density  : continuous haze density in [0, 1]
                            = 0.7 * map_mean + 0.3 * map_max
                            (the max term sharpens Heavy class separation)
      (b) class_logits    : 4-class logits (Clear / Low / Moderate / Heavy)
                            from bottleneck features via a small MLP head
      (c) haze_map        : (optional) full spatial haze density map (B, 1, H, W)
      (d) consistency_loss: (optional) scalar loss enforcing density↔class agreement

    Channel progression:
        stem → 32  |  enc1 → 64  |  enc2 → 128  |  enc3 → 256
        bottleneck → 256
        dec3 → 128  |  dec2 → 64  |  dec1 → 32
        map_head → 1     classifier: 256 → 128 → 4
    """

    # Density thresholds for bucketing into class indices (used by consistency loss)
    # [0, 0.25) = Clear, [0.25, 0.50) = Low, [0.50, 0.75) = Moderate, [0.75, 1.0] = Heavy
    DENSITY_BOUNDARIES = [0.0, 0.25, 0.50, 0.75, 1.0]

    def __init__(self, in_channels=3, base_channels=32, num_classes=4):
        super().__init__()
        self.num_classes = num_classes

        # --- Encoder ---
        self.stem = MultiScaleStem(in_channels=in_channels, base_channels=base_channels)
        self.enc1 = EncoderStage(base_channels, 64,  gpca_embed_dim=16)  # smaller embed at early stage
        self.enc2 = EncoderStage(64,  128, gpca_embed_dim=32)
        self.enc3 = EncoderStage(128, 256, gpca_embed_dim=32)

        # --- Bottleneck ---
        # Deepest representation: one conv + one residual block at full 256-channel depth
        self.bottleneck = nn.Sequential(
            ConvBNAct(256, 256, 3, 1, 1, activation="relu"),
            ResidualBlock(256, activation="relu"),
        )

        # --- Decoder ---
        # Each block upsamples and fuses with the corresponding encoder skip connection
        self.dec3 = DecoderBlock(256, 128, 128)
        self.dec2 = DecoderBlock(128, 64,  64)
        self.dec1 = DecoderBlock(64,  base_channels, base_channels)

        # --- Output heads ---
        # Spatial haze map: base_channels → 1 channel (sigmoid applied in forward)
        self.map_head = MapRefinement(base_channels)
        # Light 5×5 average pooling to smooth out pixel-level noise in the haze map
        self.smooth   = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)

        # Classification head: operates on global-average-pooled bottleneck features.
        # Input is 256-dim (bottleneck only).
        #
        # NOTE: haze_stats (mean, std, max from the map) were previously concatenated
        # here to give the classifier explicit density cues (making it 259-dim), but
        # this caused over-reliance on map quality early in training and hurt stability.
        # Reverted to bottleneck-only input. Kept as comments for reference:
        #   haze_stats   = torch.stack([haze_mean, haze_std, haze_max], dim=1)  # (B, 3)
        #   combined     = torch.cat([pooled, haze_stats], dim=1)               # (B, 259)
        #   class_logits = self.classifier(combined)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),           # dropout regularises the small classification head
            nn.Linear(128, num_classes),
        )

    def forward(self, x, return_map=False, return_consistency=False):
        """
        Args:
            x                   : (B, 3, H, W) input image tensor
            return_map          : if True, also return the spatial haze_map
            return_consistency  : if True, also return consistency_loss

        Returns (depending on flags):
            global_density   : (B,) continuous haze density per image in [0, 1]
            class_logits     : (B, num_classes) raw classification logits
            haze_map         : (B, 1, H, W)  [only if return_map=True]
            consistency_loss : scalar tensor  [only if return_consistency=True]
        """
        # --- Encoder ---
        x0 = self.stem(x)     # (B, 32,  H,    W)
        x1 = self.enc1(x0)    # (B, 64,  H/2,  W/2)
        x2 = self.enc2(x1)    # (B, 128, H/4,  W/4)
        x3 = self.enc3(x2)    # (B, 256, H/8,  W/8)

        bottleneck_feat = self.bottleneck(x3)   # (B, 256, H/8, W/8)

        # --- Decoder (with skip connections from encoder) ---
        z = self.dec3(bottleneck_feat, x2)      # (B, 128, H/4, W/4)
        z = self.dec2(z, x1)                    # (B, 64,  H/2, W/2)
        z = self.dec1(z, x0)                    # (B, 32,  H,   W)

        # --- Haze map ---
        haze_map  = torch.sigmoid(self.map_head(z))   # (B, 1, H, W) values in [0, 1]
        haze_map  = self.smooth(haze_map)              # mild spatial smoothing

        # Spatial statistics of the haze map
        haze_mean = haze_map.mean(dim=(1, 2, 3))    # (B,) — average haze level
        haze_std  = haze_map.std(dim=(1, 2, 3))     # (B,) — spatial haze variability (unused)
        haze_max  = haze_map.amax(dim=(1, 2, 3))    # (B,) — peak haze intensity

        # Global density: weighted mean + max combination.
        # The 0.3 * max term boosts signal for Heavy images where local
        # peaks are more discriminative than the spatial average alone.
        global_density = 0.7 * haze_mean + 0.3 * haze_max   # (B,)

        # --- Classification head ---
        pooled       = F.adaptive_avg_pool2d(bottleneck_feat, 1).flatten(1)  # (B, 256)
        class_logits = self.classifier(pooled)   # (B, num_classes)

        # --- Consistency loss (optional) ---
        # Converts the regression density into a pseudo class label using fixed
        # density boundaries, then measures cross-entropy against class_logits.
        # Softly couples regression and classification so they don't contradict
        # each other (e.g. density=0.85 predicting Clear would be penalised).
        # density.detach() prevents the consistency gradient from flowing into
        # the map branch — only the classifier is updated by this term.
        if return_consistency:
            pred_class_from_density = torch.bucketize(
                global_density.detach(),
                torch.tensor(self.DENSITY_BOUNDARIES[1:-1], device=x.device)
            ).clamp(0, self.num_classes - 1)
            consistency_loss = F.cross_entropy(class_logits, pred_class_from_density)

            if return_map:
                return global_density, class_logits, haze_map, consistency_loss
            return global_density, class_logits, consistency_loss

        if return_map:
            return global_density, class_logits, haze_map
        return global_density, class_logits


# ══════════════════════════════════════════════
#  HAZE LOSS v3 — Focal + Label Smoothing
# ══════════════════════════════════════════════

class HazeLoss(nn.Module):
    """
    Combined multi-task loss for GPCAHazDesNet.

        total = reg_weight    * SmoothL1(global_density, density_gt)
              + cls_weight    * FocalLoss(class_logits, class_gt)
              + cons_weight   * consistency_loss          [optional]
              + smooth_weight * TV(haze_map)              [optional]

    Component breakdown:
      - SmoothL1 regression: behaves like L2 near zero (stable) and L1 for
        large errors (robust to outliers). Supervises the continuous density output.
      - FocalLoss classification: focuses training on hard Heavy examples.
        Label smoothing (0.1) prevents overconfidence at haze class boundaries.
      - Consistency loss: enforces density↔class prediction agreement.
        Weight increased 0.1 → 0.2 in v3 for stronger coupling.
      - Total variation (TV) smoothness: penalises spatially noisy haze maps,
        encouraging physically plausible smooth haze distributions.

    Default class weights: Clear=1.0, Low=1.0, Moderate=1.2, Heavy=2.0
    Heavy increased 1.5 → 2.0 in v3 — it's the rarest and most error-prone class.
    """

    def __init__(
        self,
        reg_weight:      float = 1.0,
        cls_weight:      float = 1.0,
        cons_weight:     float = 0.2,        # increased from 0.1 in v3
        smooth_weight:   float = 0.05,       # small — only lightly regularises the spatial map
        class_weights            = None,     # override default [1.0, 1.0, 1.2, 2.0]
        focal_gamma:     float = 2.0,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.reg_weight    = reg_weight
        self.cls_weight    = cls_weight
        self.cons_weight   = cons_weight
        self.smooth_weight = smooth_weight

        if class_weights is not None:
            w = torch.tensor(class_weights, dtype=torch.float32)
        else:
            # Clear and Low are well-represented and easy — weight 1.0
            # Moderate is harder at the boundary — slight up-weight 1.2
            # Heavy is rare and hardest — strongest up-weight 2.0
            w = torch.tensor([1.0, 1.0, 1.2, 2.0])

        # register_buffer: moves with .to(device) but is not a learnable parameter
        self.register_buffer("class_weights", w)

        self.cls_loss_fn = FocalLoss(
            gamma           = focal_gamma,
            weight          = w,
            label_smoothing = label_smoothing,
        )
        self.reg_loss_fn = nn.SmoothL1Loss()

    def forward(
        self,
        density_pred:     torch.Tensor,          # (B,)      predicted density
        density_gt:       torch.Tensor,          # (B,)      ground-truth density
        class_logits:     torch.Tensor,          # (B, C)    classification logits
        class_gt:         torch.Tensor,          # (B,)      ground-truth class indices
        consistency_loss: torch.Tensor = None,   # scalar    from model(return_consistency=True)
        haze_map:         torch.Tensor = None,   # (B,1,H,W) from model(return_map=True)
    ) -> torch.Tensor:

        reg_loss = self.reg_loss_fn(density_pred, density_gt)
        cls_loss = self.cls_loss_fn(class_logits, class_gt)

        total = self.reg_weight * reg_loss + self.cls_weight * cls_loss

        if consistency_loss is not None:
            total = total + self.cons_weight * consistency_loss

        # Total variation (TV) smoothness loss on the spatial haze map.
        # Penalises abrupt pixel-to-pixel changes in both spatial directions.
        # Encourages the model to produce coherent, spatially smooth haze regions
        # rather than noisy per-pixel predictions.
        if haze_map is not None:
            dh    = torch.abs(haze_map[:, :, 1:, :] - haze_map[:, :, :-1, :]).mean()   # vertical
            dw    = torch.abs(haze_map[:, :, :, 1:] - haze_map[:, :, :, :-1]).mean()   # horizontal
            total = total + self.smooth_weight * (dh + dw)

        return total


# ══════════════════════════════════════════════
#  UTILITIES
# ══════════════════════════════════════════════

def count_parameters(model: nn.Module) -> int:
    """Returns the total number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_model(checkpoint_path: str = "checkpoints/best.pt", device: str = None):
    """
    Loads a GPCAHazDesNet checkpoint robustly, handling:
      - DataParallel 'module.' prefix (stripped automatically)
      - Shape mismatches between checkpoint and current model (skipped with warning)
      - Missing / unexpected keys (reported but non-fatal via strict=False)

    Args:
        checkpoint_path : path to the .pt checkpoint file
        device          : 'cuda' or 'cpu' — defaults to cuda if available

    Returns:
        model : GPCAHazDesNet in eval() mode on the target device
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = GPCAHazDesNet(num_classes=4)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Support checkpoints saved as plain state_dict or wrapped in a dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint   # assume raw state dict

    # Strip DataParallel 'module.' prefix if checkpoint was saved with nn.DataParallel
    cleaned = {
        k.replace("module.", "", 1) if k.startswith("module.") else k: v
        for k, v in state_dict.items()
    }

    model_state    = model.state_dict()
    filtered_state = {}
    skipped        = []

    for k, v in cleaned.items():
        if k not in model_state:
            skipped.append((k, "not in current model"))
            continue
        # Skip keys whose tensor shape changed (e.g. after architecture modifications)
        if model_state[k].shape != v.shape:
            skipped.append((k, f"checkpoint {tuple(v.shape)} != model {tuple(model_state[k].shape)}"))
            continue
        filtered_state[k] = v

    # strict=False: missing keys are left at their random initialisation
    missing, unexpected = model.load_state_dict(filtered_state, strict=False)

    if skipped:
        print("Skipped incompatible keys:")
        for k, reason in skipped:
            print(f"  {k}: {reason}")
    if missing:
        print("Missing keys (randomly initialised):", missing)
    if unexpected:
        print("Unexpected keys (ignored):", unexpected)

    model.to(device)
    model.eval()
    return model


# ══════════════════════════════════════════════
#  SMOKE TEST
# ══════════════════════════════════════════════

if __name__ == "__main__":
    model = GPCAHazDesNet(num_classes=4)
    x     = torch.randn(2, 3, 224, 224)

    # Test all four forward pass flag combinations
    density, logits                       = model(x)
    density, logits, haze_map             = model(x, return_map=True)
    density, logits, cons_loss            = model(x, return_consistency=True)
    density, logits, haze_map, cons_loss  = model(x, return_map=True, return_consistency=True)

    # Test loss with all optional terms enabled
    criterion = HazeLoss()
    loss = criterion(
        density_pred     = density,
        density_gt       = torch.rand(2),
        class_logits     = logits,
        class_gt         = torch.randint(0, 4, (2,)),
        consistency_loss = cons_loss,
        haze_map         = haze_map,
    )

    print("density      :", density.shape)
    print("logits       :", logits.shape)
    print("haze_map     :", haze_map.shape)
    print("cons_loss    :", cons_loss.item())
    print("total loss   :", loss.item())
    print("parameters   :", f"{count_parameters(model):,}")
