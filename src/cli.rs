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

    /// Enable FWM (four-wave mixing)
    #[arg(long, default_value_t = 0.0)]
    pub chi: f32,

    /// Enable dynamic harmonics
    #[arg(long)]
    pub harmonics_dyn: bool,

    /// Checkpoint output name
    #[arg(long)]
    pub checkpoint_name: Option<String>,

    /// Enable curriculum training
    #[arg(long)]
    pub curriculum: bool,
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

    #[command(flatten)]
    pub model: ModelArgs,
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
    // Future: Param, Roundtrip, VectorLength, LiveGradient, etc.
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

    /// Enable attention pathway (fixes d_normed bug #6)
    #[arg(long)]
    pub attention_pathway: bool,

    /// Enable learnable ODE (ODE backward computes real Jacobian)
    #[arg(long)]
    pub learnable_ode: bool,

    /// Enable ODE pathway (frozen ODE uses real Jacobian for gradient flow)
    #[arg(long)]
    pub ode_pathway: bool,
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

    /// Per-position wave storage (KWDS format)
    #[arg(long)]
    pub per_position: bool,

    /// Use BPE tokenizer
    #[arg(long)]
    pub bpe: bool,

    /// BPE tokenizer path
    #[arg(long, default_value = "data/tokenizer.json")]
    pub tokenizer: String,
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

    /// Target bands
    #[arg(long)]
    pub tgt_bands: usize,

    /// Output checkpoint path
    #[arg(long)]
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

    /// Bearer auth token
    #[arg(long)]
    pub token: Option<String>,

    /// Use BPE tokenizer
    #[arg(long)]
    pub bpe: bool,

    /// BPE tokenizer path
    #[arg(long, default_value = "data/tokenizer.json")]
    pub tokenizer: String,

    /// Wave memory file
    #[arg(long)]
    pub memory: Option<String>,
}
