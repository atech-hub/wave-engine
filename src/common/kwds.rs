//! KWDS (Kerr Wave Dataset) — per-position wave storage for train-from-waves.
//!
//! Stores embedding+positional waves for every position in a dataset,
//! plus target waves (next-position) for loss computation.
//! Memory-mappable for efficient training I/O.

use std::io::{Write, Read, Seek, SeekFrom};

const MAGIC: u32 = 0x4B574453; // "KWDS"
const VERSION: u32 = 1;

/// KWDS file header.
#[derive(Debug, Clone)]
pub struct KwdsHeader {
    pub n_positions: u64,
    pub n_bands: u32,
    pub n_embd: u32, // n_bands * 2
}

impl KwdsHeader {
    pub fn record_bytes(&self) -> u64 {
        self.n_embd as u64 * 4 // one wave = n_embd floats * 4 bytes
    }

    pub fn input_offset(&self, pos: u64) -> u64 {
        32 + pos * self.record_bytes()
    }

    pub fn target_offset(&self, pos: u64) -> u64 {
        32 + self.n_positions * self.record_bytes() + pos * self.record_bytes()
    }

    pub fn file_size(&self) -> u64 {
        32 + 2 * self.n_positions * self.record_bytes() // header + inputs + targets
    }
}

/// Write KWDS header.
pub fn write_header(f: &mut impl Write, header: &KwdsHeader) -> std::io::Result<()> {
    f.write_all(&MAGIC.to_le_bytes())?;
    f.write_all(&VERSION.to_le_bytes())?;
    f.write_all(&header.n_positions.to_le_bytes())?;
    f.write_all(&header.n_bands.to_le_bytes())?;
    f.write_all(&header.n_embd.to_le_bytes())?;
    f.write_all(&[0u8; 8])?; // reserved
    Ok(())
}

/// Read KWDS header.
pub fn read_header(f: &mut impl Read) -> std::io::Result<KwdsHeader> {
    let mut buf4 = [0u8; 4];
    let mut buf8 = [0u8; 8];

    f.read_exact(&mut buf4)?;
    let magic = u32::from_le_bytes(buf4);
    assert_eq!(magic, MAGIC, "Not a KWDS file");

    f.read_exact(&mut buf4)?;
    let _version = u32::from_le_bytes(buf4);

    f.read_exact(&mut buf8)?;
    let n_positions = u64::from_le_bytes(buf8);

    f.read_exact(&mut buf4)?;
    let n_bands = u32::from_le_bytes(buf4);

    f.read_exact(&mut buf4)?;
    let n_embd = u32::from_le_bytes(buf4);

    let mut _reserved = [0u8; 8];
    f.read_exact(&mut _reserved)?;

    Ok(KwdsHeader { n_positions, n_bands, n_embd })
}

/// Write a single wave record (input or target).
pub fn write_wave(f: &mut impl Write, wave: &[f32]) -> std::io::Result<()> {
    for &v in wave {
        f.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

/// Read a single wave record at a specific file offset.
pub fn read_wave_at(f: &mut (impl Read + Seek), offset: u64, n_embd: usize) -> std::io::Result<Vec<f32>> {
    f.seek(SeekFrom::Start(offset))?;
    let mut wave = vec![0.0f32; n_embd];
    for v in &mut wave {
        let mut buf = [0u8; 4];
        f.read_exact(&mut buf)?;
        *v = f32::from_le_bytes(buf);
    }
    Ok(wave)
}

/// Read a contiguous window of input waves (for training batch sampling).
pub fn read_input_window(
    f: &mut (impl Read + Seek),
    header: &KwdsHeader,
    start_pos: u64,
    window_len: usize,
) -> std::io::Result<Vec<Vec<f32>>> {
    let n_embd = header.n_embd as usize;
    let mut result = Vec::with_capacity(window_len);
    let offset = header.input_offset(start_pos);
    f.seek(SeekFrom::Start(offset))?;

    for _ in 0..window_len {
        let mut wave = vec![0.0f32; n_embd];
        for v in &mut wave {
            let mut buf = [0u8; 4];
            f.read_exact(&mut buf)?;
            *v = f32::from_le_bytes(buf);
        }
        result.push(wave);
    }
    Ok(result)
}

/// Read a contiguous window of target waves.
pub fn read_target_window(
    f: &mut (impl Read + Seek),
    header: &KwdsHeader,
    start_pos: u64,
    window_len: usize,
) -> std::io::Result<Vec<Vec<f32>>> {
    let n_embd = header.n_embd as usize;
    let mut result = Vec::with_capacity(window_len);
    let offset = header.target_offset(start_pos);
    f.seek(SeekFrom::Start(offset))?;

    for _ in 0..window_len {
        let mut wave = vec![0.0f32; n_embd];
        for v in &mut wave {
            let mut buf = [0u8; 4];
            f.read_exact(&mut buf)?;
            *v = f32::from_le_bytes(buf);
        }
        result.push(wave);
    }
    Ok(result)
}

/// Convert a tokenised dataset to KWDS.
/// Stores post-embedding waves (no ODE) as inputs, next-token embeddings as targets.
pub fn convert_tokens_to_kwds(
    path: &str,
    tokens: &[usize],
    wte: &[Vec<f32>],
    wpe: &[Vec<f32>],
    n_bands: usize,
) -> std::io::Result<()> {
    let n_embd = n_bands * 2;
    let n_positions = (tokens.len() - 1) as u64; // last token has no target

    let header = KwdsHeader {
        n_positions,
        n_bands: n_bands as u32,
        n_embd: n_embd as u32,
    };

    println!("  Writing KWDS: {} positions, {} bands, {:.1} MB",
        n_positions, n_bands,
        header.file_size() as f64 / (1024.0 * 1024.0));

    let mut f = std::fs::File::create(path)?;
    write_header(&mut f, &header)?;

    // Write input waves (embedding + positional, no ODE)
    let block_size = wpe.len();
    for i in 0..n_positions as usize {
        let tok = tokens[i];
        let pos = i % block_size;
        let mut wave = vec![0.0f32; n_embd];
        if tok < wte.len() && pos < wpe.len() {
            for j in 0..n_embd {
                wave[j] = wte[tok][j] + wpe[pos][j];
            }
        }
        write_wave(&mut f, &wave)?;
    }

    // Write target waves (next token's PURE embedding — no positional encoding)
    // This way the model learns to predict token identity, not position-specific patterns
    for i in 0..n_positions as usize {
        let next_tok = tokens[i + 1];
        let mut wave = vec![0.0f32; n_embd];
        if next_tok < wte.len() {
            for j in 0..n_embd {
                wave[j] = wte[next_tok][j]; // pure token embedding, no positional
            }
        }
        write_wave(&mut f, &wave)?;
    }

    println!("  KWDS written: {}", path);
    Ok(())
}
