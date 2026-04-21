//! CLI definition using clap derive — subcommands, typed args, auto-generated help.
//!
//! Main dispatches on the parsed command. Each subcommand's logic lives in its
//! own module, not here. This file is pure interface definition.

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "wave-engine")]
#[command(about = "Research platform for wave-coherent neural architectures")]
#[command(version)]
pub struct Cli {
    /// Rayon thread pool size (default: half available cores)
    #[arg(long, global = true)]
    pub threads: Option<usize>,

    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand)]
pub enum Command {
    /// Standard token-based training
    Train(TrainArgs),

    /// Wave-space training from KWDS dataset
    TrainWaves(TrainWavesArgs),

    /// Token-based text generation
    Generate(GenerateArgs),

    /// Wave-space generation from wave-trained checkpoints
    WaveGenerate(WaveGenerateArgs),

    /// Phase encoding: encode tokens, relate pairs, scan vocabulary
    Encode(EncodeArgs),

    /// Analyze KWMF wave memory files
    ScanMemory(ScanMemoryArgs),

    /// Geometric analysis of a trained checkpoint
    GalaxyScan(GalaxyScanArgs),

    /// Junction monitor verification suite
    Verify(VerifyArgs),

    /// Checkpoint analysis and diagnostics
    Analyze(AnalyzeArgs),

    /// ODE per-band magnitude/phase inspection
    OdeMonitor(OdeMonitorArgs),

    /// Phase decode diagnostic (compare phase-native vs lm_head)
    PhaseDecode(PhaseDecodeArgs),

    /// Convert dataset to KWDS/KWMF format
    ConvertDataset(ConvertDatasetArgs),

    /// Architecture recommendations based on config
    Recommend(RecommendArgs),

    /// Scale checkpoint to different dimensions
    ScaleCheckpoint(ScaleCheckpointArgs),

    /// Start inference server (requires --features serve)
    #[cfg(feature = "serve")]
    Serve(ServeArgs),
}

// ─── Shared arg groups ───

#[derive(clap::Args, Clone)]
pub struct ModelArgs {
    /// Number of frequency bands
    #[arg(long, default_value_t = 84)]
    pub n_bands: usize,

    /// Number of attention heads
    #[arg(long, default_value_t = 4)]
    pub n_head: usize,

    /// Number of transformer layers
    #[arg(long, default_value_t = 4)]
    pub layers: usize,

    /// Maestro bottleneck dimension
    #[arg(long, default_value_t = 16)]
    pub maestro_dim: usize,

    /// Kerr-ODE alpha (damping)
    #[arg(long, default_value_t = 0.1)]
    pub alpha: f32,

    /// Kerr-ODE beta (nonlinearity)
    #[arg(long, default_value_t = 0.2)]
    pub beta: f32,

    /// Out-proj groups (1 = dense)
    #[arg(long, default_value_t = 1)]
    pub out_proj_groups: usize,

    /// Vocabulary size
    #[arg(long, default_value_t = 15)]
    pub vocab: usize,
}

#[derive(clap::Args, Clone)]
pub struct CheckpointArgs {
    /// Path to checkpoint file
    #[arg(long)]
    pub resume: String,
}

// ─── Per-subcommand args ───

#[derive(clap::Args)]
pub struct TrainArgs {
    /// Training data file
    pub data: String,

    #[command(flatten)]
    pub model: ModelArgs,

    /// Training iterations
    #[arg(long, default_value_t = 10000)]
    pub iters: usize,

    /// Learning rate
    #[arg(long, default_value_t = 3e-4)]
    pub lr: f32,

    /// Sequence length
    #[arg(long, default_value_t = 128)]
    pub seq: usize,

    /// Batch size
    #[arg(long, default_value_t = 4)]
    pub batch: usize,

    /// Resume from checkpoint
    #[arg(long)]
    pub resume: Option<String>,

    /// Use phase-native loss
    #[arg(long)]
    pub phase_native: bool,

    /// Use BPE tokenizer
    #[arg(long)]
    pub bpe: bool,

    /// BPE tokenizer path
    #[arg(long, default_value = "data/tokenizer.json")]
    pub tokenizer: String,

    /// Use Candle backend
    #[arg(long)]
    pub candle: bool,

    /// Use CUDA kernel (implies --candle)
    #[arg(long)]
    pub cuda_kernel: bool,

    /// Use Candle CustomOp fallback (no CUDA toolkit required; implied by --cuda-kernel)
    #[arg(long)]
    pub custom_op: bool,

    /// GPU duty cycle percentage (1-100, default 100 = no sleep between iters)
    #[arg(long, default_value_t = 100)]
    pub gpu_duty: usize,

    /// Enable FWM (four-wave mixing). Alias: --fwm-strength.
    #[arg(long, alias = "fwm-strength", default_value_t = 0.0)]
    pub chi: f32,

    /// Candle per-layer NaN detection (~6x slower — diagnostic only)
    #[arg(long)]
    pub debug_nan: bool,

    /// Checkpoint output name
    #[arg(long)]
    pub checkpoint_name: Option<String>,

    /// Disable curriculum training (default: on)
    #[arg(long)]
    pub no_curriculum: bool,

    /// Enable curriculum training (kept for explicit opt-in; default behaviour is ON)
    #[arg(long)]
    pub curriculum: bool,

    /// Split-band ODE integration (freeze-and-decouple). Phase A requires chi=0.
    #[arg(long)]
    pub split_band: bool,

    /// Train attention weights (phase_proj, v_proj, out_proj, harmonic_raw).
    /// Content projection stays frozen by design. Default: frozen attention.
    #[arg(long)]
    pub learnable_attn: bool,

    /// Train on a KWDS wave dataset with L2 loss on ODE output states instead
    /// of cross-entropy on token logits. The positional DATA arg becomes the
    /// KWDS file path. All other flags (split-band, monitors, GPU, etc.) still
    /// apply — wave training now shares the main training loop.
    #[arg(long)]
    pub wave_loss: bool,

    /// Use wgpu GPU backend
    #[arg(long)]
    pub gpu: bool,

    /// Enable pipeline monitor
    #[arg(long)]
    pub monitor: bool,

    /// Custom training log filename
    #[arg(long)]
    pub log_name: Option<String>,

    // ─── Architecture / encoding ───

    /// Tied embeddings (wte reused as lm_head)
    #[arg(long)]
    pub tied_embeddings: bool,

    /// Low-rank lm_head factorization (0 = full rank)
    #[arg(long, default_value_t = 0)]
    pub lm_rank: usize,

    /// Wave-decode mode
    #[arg(long)]
    pub wave_decode: bool,

    /// Train phase offsets as learnable parameters
    #[arg(long)]
    pub unfreeze_phases: bool,

    /// Freeze ODE (identity shortcut — degrades gradients, for A/B only)
    #[arg(long)]
    pub freeze_ode: bool,

    /// Pythagorean sphere encoding
    #[arg(long)]
    pub pythagorean: bool,

    /// Custom modulus m1 for dual-modulus encoding
    #[arg(long)]
    pub m1: Option<usize>,

    /// Custom modulus m2 for dual-modulus encoding
    #[arg(long)]
    pub m2: Option<usize>,

    // ─── Training schedule / regularisation ───

    /// Health-sample interval in iters (0 = disabled)
    #[arg(long, default_value_t = 0)]
    pub health_interval: usize,

    /// Head LR floor for hypergradient (0 = disabled)
    #[arg(long, default_value_t = 0.0)]
    pub head_lr_floor: f32,

    /// Phase-native loss temperature
    #[arg(long, default_value_t = 1.0)]
    pub phase_temp: f32,

    /// AGC ceiling override (None = derive from alpha)
    #[arg(long)]
    pub agc_ceiling: Option<f32>,

    /// Spring constant for dynamic params (0 = no spring)
    #[arg(long, default_value_t = 0.1)]
    pub spring: f32,

    /// First N layers active at eq=1.0, rest dormant at eq=0.0
    #[arg(long)]
    pub active_layers: Option<usize>,

    // ─── DynParam flags (off | dyn | CSV values) ───

    /// Per-layer residual scaling: off | dyn | v1,v2,…
    #[arg(long, default_value = "off")]
    pub layer_scale: crate::cpu::train::DynParam,

    /// Per-group LR scaling: off | dyn | v1,v2,…
    #[arg(long, default_value = "off")]
    pub lr_scale: crate::cpu::train::DynParam,

    /// Per-layer RK4 combination weights: off | dyn
    #[arg(long, default_value = "off")]
    pub rk4_weights: crate::cpu::train::DynParam,

    /// Weight decay: off | dyn | v1,v2,…
    #[arg(long, default_value = "off")]
    pub wd: crate::cpu::train::DynParam,

    /// Learnable harmonic numbers: off | dyn | v1,v2,…
    #[arg(long, default_value = "off")]
    pub harmonics: crate::cpu::train::DynParam,

    /// AGC headroom: off | dyn | v1,v2,…
    #[arg(long, default_value = "off")]
    pub agc_headroom: crate::cpu::train::DynParam,

    /// Corrector plate: dyn | off (default: dyn)
    #[arg(long, default_value = "dyn")]
    pub corrector: crate::cpu::train::DynParam,

    /// Legacy alias: equivalent to --corrector off
    #[arg(long)]
    pub no_corrector: bool,

    // ─── Pathway flags (default on; --no-* disables for A/B) ───

    /// ODE pathway: frozen ODE uses real Jacobian for gradient flow (default on)
    #[arg(long, default_value_t = true)]
    pub ode_pathway: bool,

    /// Attention pathway: attention backward contributes to d_normed (default on)
    #[arg(long, default_value_t = true)]
    pub attention_pathway: bool,

    /// Disable ODE pathway (identity shortcut, for A/B comparison only)
    #[arg(long)]
    pub no_ode_pathway: bool,

    /// Disable attention pathway (for A/B comparison only)
    #[arg(long)]
    pub no_attention_pathway: bool,
}

#[derive(clap::Args)]
pub struct TrainWavesArgs {
    /// KWDS dataset file
    pub kwds: String,

    #[command(flatten)]
    pub model: ModelArgs,

    /// Training iterations
    #[arg(long, default_value_t = 10000)]
    pub iters: usize,

    /// Learning rate
    #[arg(long, default_value_t = 3e-4)]
    pub lr: f32,

    /// Sequence length
    #[arg(long, default_value_t = 64)]
    pub seq: usize,

    /// Checkpoint output name
    #[arg(long, default_value = "wave_trained.bin")]
    pub checkpoint_name: String,

    /// Resume from checkpoint
    #[arg(long)]
    pub resume: Option<String>,
}

#[derive(clap::Args)]
pub struct GenerateArgs {
    #[command(flatten)]
    pub checkpoint: CheckpointArgs,

    #[command(flatten)]
    pub model: ModelArgs,

    /// Training data file (for char vocabulary; ignored with --bpe)
    #[arg(long, default_value = "data/input.txt")]
    pub data: String,

    /// Prompt text
    #[arg(long, default_value = "The ")]
    pub prompt: String,

    /// Maximum tokens to generate
    #[arg(long, default_value_t = 200)]
    pub max_tokens: usize,

    /// Use BPE tokenizer
    #[arg(long)]
    pub bpe: bool,

    /// BPE tokenizer path
    #[arg(long, default_value = "data/tokenizer.json")]
    pub tokenizer: String,

    /// Sampling temperature (0 = greedy)
    #[arg(long, default_value_t = 0.0)]
    pub temperature: f32,

    /// Use phase-native decode
    #[arg(long)]
    pub phase_native: bool,

    /// Wave memory file path
    #[arg(long)]
    pub memory: Option<String>,
}

#[derive(clap::Args)]
pub struct WaveGenerateArgs {
    #[command(flatten)]
    pub checkpoint: CheckpointArgs,

    #[command(flatten)]
    pub model: ModelArgs,

    /// Training data file (for char vocabulary)
    #[arg(long, default_value = "data/input.txt")]
    pub data: String,

    /// Prompt text
    #[arg(long, default_value = "3+4=")]
    pub prompt: String,

    /// Maximum tokens to generate
    #[arg(long, default_value_t = 10)]
    pub max_tokens: usize,

    /// Use BPE tokenizer
    #[arg(long)]
    pub bpe: bool,

    /// BPE tokenizer path
    #[arg(long, default_value = "data/tokenizer.json")]
    pub tokenizer: String,

    /// Sampling temperature (0 = greedy)
    #[arg(long, default_value_t = 0.0)]
    pub temperature: f32,

    /// Run per-band phase/magnitude diagnostic
    #[arg(long)]
    pub wave_diagnose: bool,

    /// Run teacher-forced accuracy test with KWDS file
    #[arg(long)]
    pub teacher_force: Option<String>,
}

#[derive(clap::Args)]
pub struct EncodeArgs {
    /// Checkpoint to load (omit with --blank for untrained model)
    #[arg(long)]
    pub resume: Option<String>,

    #[command(flatten)]
    pub model: ModelArgs,

    /// Data file for char vocabulary
    #[arg(long, default_value = "data/input.txt")]
    pub data: String,

    /// Use untrained (blank) model instead of checkpoint
    #[arg(long)]
    pub blank: bool,

    /// Multi-grid modulus 1
    #[arg(long, default_value_t = 5)]
    pub m1: usize,

    /// Multi-grid modulus 2
    #[arg(long, default_value_t = 7)]
    pub m2: usize,

    /// Layer to inject encoding at
    #[arg(long, default_value_t = 0)]
    pub inject_layer: usize,

    /// Run galaxy scan on encode output
    #[arg(long)]
    pub scan: bool,

    /// Run relate-vocab: full vocabulary relationship scan
    #[arg(long)]
    pub relate_vocab: bool,

    /// Relate items pairwise (repeat for each item: --relate "a" --relate "b")
    #[arg(long)]
    pub relate: Vec<String>,

    /// Relate a number (used with --relate)
    #[arg(long)]
    pub relate_number: Vec<u64>,

    /// Relate a catalog spec (used with --relate)
    #[arg(long)]
    pub relate_catalog: Vec<String>,

    /// Encode text (single item)
    #[arg(long)]
    pub encode: Option<String>,

    /// Encode a number
    #[arg(long)]
    pub encode_number: Option<u64>,

    /// Encode from catalog specification
    #[arg(long)]
    pub encode_catalog: Option<String>,

    /// Encode raw phases
    #[arg(long)]
    pub encode_phases: Option<String>,

    /// Output file for relate-vocab JSON
    #[arg(long, default_value = "vocab_relations.json")]
    pub output: String,
}

#[derive(clap::Args)]
pub struct ScanMemoryArgs {
    /// KWMF memory file to analyze
    pub file: String,

    /// Write JSON scan to this path
    #[arg(long)]
    pub output: Option<String>,
}

#[derive(clap::Args)]
pub struct GalaxyScanArgs {
    #[command(flatten)]
    pub checkpoint: CheckpointArgs,

    #[command(flatten)]
    pub model: ModelArgs,

    /// Corpus file for scan positions
    #[arg(long)]
    pub scan_corpus: Option<String>,

    /// Multi-grid modulus 1
    #[arg(long, default_value_t = 5)]
    pub m1: usize,

    /// Multi-grid modulus 2
    #[arg(long, default_value_t = 7)]
    pub m2: usize,
}

#[derive(Subcommand)]
pub enum VerifyCommand {
    /// J1: Gradient correctness (analytical vs finite-difference)
    Grad(VerifyGradArgs),

    /// J10: Tier parity (CPU vs wgpu forward, section-by-section diff)
    TierParity(VerifyTierParityArgs),
    // Future: Param, Roundtrip, VectorLength, LiveGradient, etc.
}

#[derive(clap::Args)]
pub struct VerifyTierParityArgs {
    #[command(flatten)]
    pub model: ModelArgs,

    /// Tier to compare against CPU: wgpu | candle
    #[arg(long, default_value = "wgpu")]
    pub tier: String,

    /// Checkpoint to load (omit for random-init model)
    #[arg(long)]
    pub resume: Option<String>,

    /// Sequence length for the forward pass
    #[arg(long, default_value_t = 16)]
    pub seq: usize,

    /// Number of forward-pass iterations (deterministic; >1 to average timing noise)
    #[arg(long, default_value_t = 1)]
    pub iters: usize,

    /// Print every section's diff even on pass (not just violations)
    #[arg(long)]
    pub verbose: bool,

    /// Run both tiers with split-band ODE integration (Phase A; chi=0 required)
    #[arg(long)]
    pub split_band: bool,
}

#[derive(clap::Args)]
pub struct VerifyArgs {
    #[command(subcommand)]
    pub command: VerifyCommand,
}

#[derive(clap::Args)]
pub struct VerifyGradArgs {
    /// Training mode to check
    #[arg(default_value = "phase-native")]
    pub mode: String,

    /// Tier to check: cpu | candle. Candle requires --features candle-backend.
    #[arg(long, default_value = "cpu")]
    pub tier: String,

    #[command(flatten)]
    pub model: ModelArgs,

    /// Check scope: tiny, sampled, exhaustive
    #[arg(long, default_value = "sampled")]
    pub scope: String,

    /// Perturbation epsilon
    #[arg(long, default_value_t = 1e-4)]
    pub eps: f32,

    /// Pass tolerance
    #[arg(long, default_value_t = 0.01)]
    pub tol: f32,

    /// Print every parameter check
    #[arg(long)]
    pub verbose: bool,

    /// Attention pathway: attention backward contributes to d_normed (default on)
    #[arg(long, default_value_t = true)]
    pub attention_pathway: bool,

    /// Enable learnable ODE (ODE backward computes real Jacobian + trains ODE params)
    #[arg(long)]
    pub learnable_ode: bool,

    /// ODE pathway: frozen ODE uses real Jacobian for gradient flow (default on)
    #[arg(long, default_value_t = true)]
    pub ode_pathway: bool,

    /// Disable attention pathway (for baseline comparison)
    #[arg(long)]
    pub no_attention_pathway: bool,

    /// Disable ODE pathway (for baseline comparison)
    #[arg(long)]
    pub no_ode_pathway: bool,

    /// Split-band ODE integration (freeze-and-decouple). Phase A requires chi=0.
    #[arg(long)]
    pub split_band: bool,

    /// Include attention weights in the grad check (requires learnable attention build)
    #[arg(long)]
    pub learnable_attn: bool,
}

#[derive(clap::Args)]
pub struct AnalyzeArgs {
    #[command(flatten)]
    pub checkpoint: CheckpointArgs,

    #[command(flatten)]
    pub model: ModelArgs,

    /// Data file
    #[arg(long, default_value = "data/input.txt")]
    pub data: String,

    /// Use BPE tokenizer
    #[arg(long)]
    pub bpe: bool,

    /// BPE tokenizer path
    #[arg(long, default_value = "data/tokenizer.json")]
    pub tokenizer: String,

    /// Enable sub-harmonic diagnostic
    #[arg(long)]
    pub sub_harmonic: bool,
}

#[derive(clap::Args)]
pub struct OdeMonitorArgs {
    #[command(flatten)]
    pub checkpoint: CheckpointArgs,

    #[command(flatten)]
    pub model: ModelArgs,

    /// Data file
    #[arg(long, default_value = "data/input.txt")]
    pub data: String,

    /// Prompt for ODE inspection
    #[arg(long)]
    pub prompt: Option<String>,

    /// Compare two prompts
    #[arg(long)]
    pub compare: Vec<String>,
}

#[derive(clap::Args)]
pub struct PhaseDecodeArgs {
    #[command(flatten)]
    pub checkpoint: CheckpointArgs,

    #[command(flatten)]
    pub model: ModelArgs,

    /// Data file
    #[arg(long, default_value = "data/input.txt")]
    pub data: String,
}

#[derive(clap::Args)]
pub struct ConvertDatasetArgs {
    /// Input data file
    pub data: String,

    #[command(flatten)]
    pub model: ModelArgs,

    /// Output file path
    #[arg(long)]
    pub output: String,

    /// Per-position wave storage (KWDS format). Without this flag, writes aggregate
    /// KWMF by running tokens through a model in block-size chunks.
    #[arg(long)]
    pub per_position: bool,

    /// Use BPE tokenizer
    #[arg(long)]
    pub bpe: bool,

    /// BPE tokenizer path
    #[arg(long, default_value = "data/tokenizer.json")]
    pub tokenizer: String,

    /// Convert through a trained model (KWMF aggregate mode). Without this, uses
    /// an untrained random-init model.
    #[arg(long)]
    pub resume: Option<String>,
}

#[derive(clap::Args)]
pub struct RecommendArgs {
    /// Data file to analyze for recommendations
    pub data: String,
}

#[derive(clap::Args)]
pub struct ScaleCheckpointArgs {
    #[command(flatten)]
    pub checkpoint: CheckpointArgs,

    /// Source bands
    #[arg(long)]
    pub src_bands: usize,

    /// Target bands. Alias: --target-bands.
    #[arg(long, alias = "target-bands", default_value_t = 128)]
    pub tgt_bands: usize,

    /// Target attention heads
    #[arg(long, default_value_t = 8)]
    pub target_head: usize,

    /// Target layers (None = keep source layer count)
    #[arg(long)]
    pub target_layers: Option<usize>,

    /// Out-proj groups for target
    #[arg(long, default_value_t = 1)]
    pub out_proj_groups: usize,

    /// Output checkpoint path
    #[arg(long, default_value = "scaled_checkpoint.bin")]
    pub output: String,
}

#[cfg(feature = "serve")]
#[derive(clap::Args)]
pub struct ServeArgs {
    #[command(flatten)]
    pub checkpoint: CheckpointArgs,

    #[command(flatten)]
    pub model: ModelArgs,

    /// Server port
    #[arg(long, default_value_t = 8080)]
    pub port: u16,

    /// Bind address
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,

    /// Model name advertised via /v1/models
    #[arg(long, default_value = "wave-engine")]
    pub model_name: String,

    /// Bearer auth token (alias: --api-key)
    #[arg(long, alias = "api-key")]
    pub token: Option<String>,

    /// Use phase-native decode (for phase-native checkpoints)
    #[arg(long)]
    pub phase_native: bool,

    /// Use BPE tokenizer
    #[arg(long)]
    pub bpe: bool,

    /// BPE tokenizer path
    #[arg(long, default_value = "data/tokenizer.json")]
    pub tokenizer: String,

    /// Training data file (required for char-level vocab; ignored with --bpe)
    #[arg(long)]
    pub data: Option<String>,

    /// Wave memory file
    #[arg(long)]
    pub memory: Option<String>,
}
