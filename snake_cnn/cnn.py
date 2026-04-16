"""
cnn.py – Homemade CNN built entirely with NumPy.

No PyTorch / TensorFlow. Everything is explicit.

Supported layer types (pass as arch list to HomemadeCNN):
  {"type": "conv",    "out_ch": int, "kernel": int, "activation": fn|None}
  {"type": "pool",    "size": int}
  {"type": "flatten"}
  {"type": "dense",   "out_size": int, "activation": fn|None}

Default architecture (2 conv + 2 dense – minimum to fulfill "2+ layers to decide"):
  Input  20×20×3
  Conv   3→8   3×3  ReLU
  Conv   8→16  3×3  ReLU
  Flatten → 4096
  Dense  4096→64  ReLU
  Dense  64→4     (raw logits)
  → softmax in agent
"""
import numpy as np

# ── activations ──────────────────────────────────

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / (e.sum() + 1e-9)


# ── fast im2col via stride tricks ─────────────────

def _im2col(x: np.ndarray, kH: int, kW: int) -> np.ndarray:
    """
    Extract every kH×kW patch from x (H, W, C) into rows.
    Returns (out_H * out_W, kH * kW * C).

    Uses np.lib.stride_tricks.as_strided – zero-copy until reshape.
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    H, W, C = x.shape
    out_H = H - kH + 1
    out_W = W - kW + 1
    strides = (x.strides[0], x.strides[1],   # slide along H, W
               x.strides[0], x.strides[1],   # kernel rows, cols
               x.strides[2])                 # channels
    patches = np.lib.stride_tricks.as_strided(
        x,
        shape=(out_H, out_W, kH, kW, C),
        strides=strides,
    )
    return patches.reshape(out_H * out_W, kH * kW * C)


# ── layer classes ─────────────────────────────────

class ConvLayer:
    """
    2-D convolution, no padding, stride 1.
    Input  (H, W, in_ch)  →  Output (H-k+1, W-k+1, out_ch)
    """

    def __init__(self, in_ch: int, out_ch: int,
                 kernel_size: int = 3, activation=relu):
        k = kernel_size
        self.k      = k
        self.out_ch = out_ch
        self.activation = activation
        # He initialization (good default for ReLU networks)
        scale  = np.sqrt(2.0 / (in_ch * k * k))
        self.W = (np.random.randn(out_ch, in_ch * k * k) * scale).astype(np.float32)
        self.b = np.zeros(out_ch, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        cols  = _im2col(x, self.k, self.k)          # (N, k*k*C)
        out_H = x.shape[0] - self.k + 1
        out_W = x.shape[1] - self.k + 1
        out   = (cols @ self.W.T + self.b).reshape(out_H, out_W, self.out_ch)
        return self.activation(out) if self.activation else out


class MaxPool2D:
    """
    2-D max pooling (no overlap, stride = pool size).
    Input (H, W, C)  →  Output (H//s, W//s, C)
    """

    def __init__(self, size: int = 2):
        self.s = size

    def forward(self, x: np.ndarray) -> np.ndarray:
        H, W, C = x.shape
        s = self.s
        oH, oW = H // s, W // s
        # Crop to exact multiple then reshape + max – no loops
        return x[:oH * s, :oW * s].reshape(oH, s, oW, s, C).max(axis=(1, 3))


class Flatten:
    """Flattens (H, W, C) → (H*W*C,)."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x.ravel().astype(np.float32)


class DenseLayer:
    """
    Fully-connected layer.
    Input (in_size,)  →  Output (out_size,)
    """

    def __init__(self, in_size: int, out_size: int, activation=relu):
        scale      = np.sqrt(2.0 / in_size)
        self.W     = (np.random.randn(in_size, out_size) * scale).astype(np.float32)
        self.b     = np.zeros(out_size, dtype=np.float32)
        self.activation = activation

    def forward(self, x: np.ndarray) -> np.ndarray:
        out = x @ self.W + self.b
        return self.activation(out) if self.activation else out


# ── default architecture ──────────────────────────

DEFAULT_ARCH = [
    {"type": "conv",    "out_ch": 8,  "kernel": 3},          # layer 1
    {"type": "conv",    "out_ch": 16, "kernel": 3},          # layer 2
    {"type": "flatten"},
    {"type": "dense",   "out_size": 64},
    {"type": "dense",   "out_size": 4, "activation": None},  # raw logits
]

# Example minimal 1-conv architecture:
ARCH_1CONV = [
    {"type": "conv",    "out_ch": 8, "kernel": 3},
    {"type": "flatten"},
    {"type": "dense",   "out_size": 4, "activation": None},
]

# Example deeper architecture:
ARCH_DEEP = [
    {"type": "conv",    "out_ch": 8,  "kernel": 3},
    {"type": "conv",    "out_ch": 16, "kernel": 3},
    {"type": "pool",    "size": 2},
    {"type": "conv",    "out_ch": 32, "kernel": 3},
    {"type": "flatten"},
    {"type": "dense",   "out_size": 128},
    {"type": "dense",   "out_size": 64},
    {"type": "dense",   "out_size": 4, "activation": None},
]


# ── configurable CNN ──────────────────────────────

class HomemadeCNN:
    """
    Build a CNN from a list of layer config dicts.

    Usage:
        cnn = HomemadeCNN()                         # default 2-conv arch
        cnn = HomemadeCNN(arch=ARCH_1CONV)          # 1 conv layer
        cnn = HomemadeCNN(arch=ARCH_DEEP)           # deeper network

        logits = cnn.forward(state)                 # state: (20, 20, 3)
        # logits shape: (4,) – one score per action

    Weight access (for evolutionary training):
        flat = cnn.get_weights()
        cnn.set_weights(flat)
    """

    def __init__(self, in_ch: int = 3, grid_size: int = 20, arch=None):
        self.arch   = arch or DEFAULT_ARCH
        self.layers = []
        self._build(in_ch, grid_size)

    def _build(self, in_ch: int, grid_size: int):
        ch   = in_ch
        h    = grid_size       # tracks spatial dimension (assumes square)
        flat = None            # size after Flatten

        for cfg in self.arch:
            t = cfg["type"]

            if t == "conv":
                out_ch = cfg["out_ch"]
                k      = cfg.get("kernel", 3)
                act    = cfg.get("activation", relu)
                self.layers.append(ConvLayer(ch, out_ch, k, act))
                h  = h - k + 1
                ch = out_ch

            elif t == "pool":
                s = cfg.get("size", 2)
                self.layers.append(MaxPool2D(s))
                h = h // s

            elif t == "flatten":
                self.layers.append(Flatten())
                flat = h * h * ch       # spatial_h == spatial_w (square grids)

            elif t == "dense":
                in_s  = flat
                out_s = cfg["out_size"]
                act   = cfg.get("activation", relu)
                self.layers.append(DenseLayer(in_s, out_s, act))
                flat  = out_s

    # ── forward pass ─────────────────────────────

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Full forward pass.
        x     : (H, W, C) float32
        return: (4,) float32 raw logits
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x

    # ── weight serialization (for evolution) ─────

    def get_weights(self) -> np.ndarray:
        """Return a flat 1-D array of every W and b in the network."""
        parts = []
        for layer in self.layers:
            if hasattr(layer, "W"):
                parts.append(layer.W.ravel())
                parts.append(layer.b.ravel())
        return np.concatenate(parts).astype(np.float32)

    def set_weights(self, flat_w: np.ndarray):
        """Load a flat weight vector back into the network layers."""
        idx = 0
        for layer in self.layers:
            if hasattr(layer, "W"):
                n = layer.W.size
                layer.W = flat_w[idx: idx + n].reshape(layer.W.shape).astype(np.float32)
                idx += n
                m = layer.b.size
                layer.b = flat_w[idx: idx + m].astype(np.float32)
                idx += m

    def layer_summary(self) -> list[str]:
        """Human-readable list of layer descriptions."""
        lines = []
        for layer in self.layers:
            if isinstance(layer, ConvLayer):
                lines.append(f"Conv  kernel={layer.k}  out_ch={layer.out_ch}  act={'ReLU' if layer.activation else 'none'}")
            elif isinstance(layer, MaxPool2D):
                lines.append(f"MaxPool  size={layer.s}")
            elif isinstance(layer, Flatten):
                lines.append("Flatten")
            elif isinstance(layer, DenseLayer):
                lines.append(f"Dense  {layer.W.shape[0]}->{layer.W.shape[1]}  act={'ReLU' if layer.activation else 'none'}")
        return lines


