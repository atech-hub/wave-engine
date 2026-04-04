//! Unified data loader — handles .txt, .jsonl, and directories.
//!
//! Auto-detects format from file extension and path type.
//! All formats produce the same output: (Vec<usize>, vocab_size).
//! Supports both char-level and BPE tokenization.
//! Token cache integration: encode once, load instantly on repeat runs.

use std::collections::{BTreeSet, HashMap};
use std::io::{BufRead, BufReader};

use crate::common::bpe;
use crate::common::token_cache;

/// Load and tokenize data from any supported source.
///
/// Format detection:
/// - Directory → concatenate all .txt/.jsonl files sorted by name
/// - .jsonl    → stream line-by-line, extract "text" field
/// - otherwise → plain text (existing behavior)
///
/// Returns (tokens, vocab_size).
pub fn load_data(path: &str, use_bpe: bool, tokenizer_path: Option<&str>) -> (Vec<usize>, usize) {
    // Check token cache first (any format)
    if let Some(cached) = token_cache::load_cache(path, use_bpe, tokenizer_path) {
        return cached;
    }

    let p = std::path::Path::new(path);

    // Gather raw text from the appropriate source
    let raw_text = if p.is_dir() {
        load_directory_text(path)
    } else {
        let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("");
        match ext {
            "jsonl" => load_jsonl_text(path),
            _ => std::fs::read_to_string(path).expect("Failed to read data file"),
        }
    };

    // Tokenize
    let (tokens, vocab_size) = if use_bpe {
        let tok_path = tokenizer_path.unwrap_or("data/tokenizer.json");
        let tokenizer = bpe::BpeTokenizer::from_file(tok_path);
        let t = tokenizer.encode(&raw_text);
        let v = tokenizer.vocab_size;
        println!("  BPE tokens: {}, vocab: {}", t.len(), v);
        (t, v)
    } else {
        tokenize_chars(&raw_text)
    };

    // Save to cache for instant reload next time
    token_cache::save_cache(path, use_bpe, tokenizer_path, &tokens, vocab_size);
    (tokens, vocab_size)
}

/// Load raw text from all supported files in a directory.
/// Files are sorted by name for deterministic ordering.
fn load_directory_text(dir_path: &str) -> String {
    let mut entries: Vec<std::path::PathBuf> = std::fs::read_dir(dir_path)
        .unwrap_or_else(|e| { eprintln!("Failed to read directory {}: {}", dir_path, e); std::process::exit(1); })
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension().map_or(false, |ext| {
                ext == "txt" || ext == "jsonl"
            })
        })
        .collect();

    entries.sort();

    if entries.is_empty() {
        eprintln!("No .txt or .jsonl files found in {}", dir_path);
        std::process::exit(1);
    }

    println!("  [dir] Found {} data files in {}", entries.len(), dir_path);

    let mut all_text = String::new();
    for path in &entries {
        let ext = path.extension().unwrap().to_str().unwrap();
        let path_str = path.to_str().unwrap();
        let file_text = match ext {
            "jsonl" => load_jsonl_text(path_str),
            _ => std::fs::read_to_string(path_str)
                .unwrap_or_else(|e| { eprintln!("  [dir] Failed to read {}: {}", path_str, e); String::new() }),
        };
        let chars = file_text.len();
        println!("  [dir] {} -> {} chars", path.file_name().unwrap().to_str().unwrap(), chars);
        all_text.push_str(&file_text);
    }

    println!("  [dir] Total: {} chars from {} files", all_text.len(), entries.len());
    all_text
}

/// Load raw text from a JSONL file by extracting the "text" field from each line.
fn load_jsonl_text(path: &str) -> String {
    let file = std::fs::File::open(path)
        .unwrap_or_else(|e| { eprintln!("Failed to open JSONL file {}: {}", path, e); std::process::exit(1); });
    let reader = BufReader::with_capacity(64 * 1024, file);

    let mut text = String::new();
    let mut lines_processed = 0usize;
    let mut lines_with_text = 0usize;

    for line in reader.lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => continue,
        };
        if line.is_empty() { continue; }

        if let Some(extracted) = extract_text_field(&line) {
            // Unescape basic JSON escapes
            let unescaped = unescape_json(&extracted);
            if !text.is_empty() {
                text.push('\n'); // separator between documents
            }
            text.push_str(&unescaped);
            lines_with_text += 1;
        }

        lines_processed += 1;
        if lines_processed % 100_000 == 0 {
            eprintln!("  [jsonl] {} lines processed, {} with text, {} chars so far",
                lines_processed, lines_with_text, text.len());
        }
    }

    println!("  [jsonl] Complete: {} lines ({} with text), {} chars",
        lines_processed, lines_with_text, text.len());
    text
}

/// Extract the "text" field from a JSONL line.
/// Minimal JSON extraction — no serde_json dependency.
/// Handles: {"text": "..."} and {"text": "...", "meta": ...}
fn extract_text_field(line: &str) -> Option<String> {
    // Find "text": or "text" :
    let key = "\"text\"";
    let key_pos = line.find(key)?;
    let after_key = &line[key_pos + key.len()..];

    // Skip optional whitespace and colon
    let after_colon = after_key.trim_start().strip_prefix(':')?;
    let after_space = after_colon.trim_start();

    // Must start with a quote
    if !after_space.starts_with('"') { return None; }
    let content = &after_space[1..];

    // Find closing quote (handle escaped quotes properly)
    let mut end = 0;
    let bytes = content.as_bytes();
    while end < bytes.len() {
        if bytes[end] == b'\\' {
            // Skip next character (escaped)
            end += 2;
            continue;
        }
        if bytes[end] == b'"' {
            return Some(content[..end].to_string());
        }
        end += 1;
    }
    None
}

/// Unescape basic JSON string escapes.
fn unescape_json(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('n') => result.push('\n'),
                Some('t') => result.push('\t'),
                Some('r') => result.push('\r'),
                Some('"') => result.push('"'),
                Some('\\') => result.push('\\'),
                Some('/') => result.push('/'),
                Some(other) => { result.push('\\'); result.push(other); }
                None => result.push('\\'),
            }
        } else {
            result.push(c);
        }
    }
    result
}

/// Char-level tokenization: build vocab from all unique chars, then tokenize.
fn tokenize_chars(text: &str) -> (Vec<usize>, usize) {
    let chars: Vec<char> = text.chars().collect();
    let unique: BTreeSet<char> = chars.iter().cloned().collect();
    let vocab: Vec<char> = unique.into_iter().collect(); // BTreeSet is sorted
    let vocab_size = vocab.len();
    let char_to_idx: HashMap<char, usize> = vocab.iter().enumerate().map(|(i, &c)| (c, i)).collect();
    let tokens: Vec<usize> = chars.iter().map(|c| *char_to_idx.get(c).unwrap_or(&0)).collect();
    println!("  Char-level tokens: {}, vocab: {}", tokens.len(), vocab_size);
    (tokens, vocab_size)
}

/// Load raw text from any supported source (for recommend, generate, etc.).
/// Handles .txt, .jsonl, and directories.
pub fn load_text_raw(path: &str) -> String {
    let p = std::path::Path::new(path);
    if p.is_dir() {
        load_directory_text(path)
    } else {
        let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("");
        match ext {
            "jsonl" => load_jsonl_text(path),
            _ => std::fs::read_to_string(path)
                .unwrap_or_else(|e| { eprintln!("Error reading {}: {}", path, e); std::process::exit(1); }),
        }
    }
}

// ─── Large dataset support: .wtok binary + mmap ───────────────────

/// WTOK binary token file format:
/// [4 bytes magic "WTOK"]
/// [4 bytes u32 vocab_size]
/// [8 bytes u64 n_tokens]
/// [n_tokens × 4 bytes u32 token_ids]
const WTOK_MAGIC: &[u8; 4] = b"WTOK";
const WTOK_HEADER_SIZE: usize = 4 + 4 + 8; // 16 bytes

/// Save tokenized data to .wtok binary format.
pub fn save_wtok(path: &str, tokens: &[usize], vocab_size: usize) {
    use std::io::Write;
    let mut f = std::fs::File::create(path)
        .unwrap_or_else(|e| { eprintln!("Failed to create {}: {}", path, e); std::process::exit(1); });
    f.write_all(WTOK_MAGIC).unwrap();
    f.write_all(&(vocab_size as u32).to_le_bytes()).unwrap();
    f.write_all(&(tokens.len() as u64).to_le_bytes()).unwrap();
    for &t in tokens {
        f.write_all(&(t as u32).to_le_bytes()).unwrap();
    }
    let mb = (WTOK_HEADER_SIZE + tokens.len() * 4) as f64 / 1e6;
    println!("  [wtok] Saved {:.1}MB ({} tokens, vocab {}) -> {}", mb, tokens.len(), vocab_size, path);
}

/// Load tokenized data from .wtok via memory mapping.
/// Zero-copy: the OS pages data in as needed. Training accesses random
/// windows, so only the active pages are in physical memory.
pub fn load_wtok_mmap(path: &str) -> (MmapTokens, usize) {
    let file = std::fs::File::open(path)
        .unwrap_or_else(|e| { eprintln!("Failed to open {}: {}", path, e); std::process::exit(1); });
    let mmap = unsafe { memmap2::Mmap::map(&file) }
        .unwrap_or_else(|e| { eprintln!("Failed to mmap {}: {}", path, e); std::process::exit(1); });

    // Validate header
    assert!(&mmap[..4] == WTOK_MAGIC, "Not a WTOK file: {}", path);
    let vocab_size = u32::from_le_bytes(mmap[4..8].try_into().unwrap()) as usize;
    let n_tokens = u64::from_le_bytes(mmap[8..16].try_into().unwrap()) as usize;

    let mb = mmap.len() as f64 / 1e6;
    println!("  [wtok] Mapped {:.1}MB ({} tokens, vocab {}) from {}", mb, n_tokens, vocab_size, path);

    (MmapTokens { mmap, n_tokens }, vocab_size)
}

/// Memory-mapped token data. Provides random access to tokens via OS paging.
pub struct MmapTokens {
    mmap: memmap2::Mmap,
    pub n_tokens: usize,
}

impl MmapTokens {
    /// Read a token at a given index.
    #[inline]
    pub fn token_at(&self, idx: usize) -> usize {
        let offset = WTOK_HEADER_SIZE + idx * 4;
        u32::from_le_bytes(self.mmap[offset..offset + 4].try_into().unwrap()) as usize
    }

    /// Read a window of tokens into a pre-allocated buffer.
    pub fn read_window(&self, start: usize, buf: &mut [usize]) {
        for (i, slot) in buf.iter_mut().enumerate() {
            *slot = self.token_at(start + i);
        }
    }

    /// Read a window as a new Vec (convenience for existing code).
    pub fn window_vec(&self, start: usize, len: usize) -> Vec<usize> {
        let mut buf = vec![0usize; len];
        self.read_window(start, &mut buf);
        buf
    }
}

/// Threshold for switching to mmap mode (bytes of raw text).
/// Below this: load entirely into RAM as Vec<usize>.
/// Above this: tokenize to .wtok binary, then mmap.
const LARGE_FILE_THRESHOLD: u64 = 500 * 1024 * 1024; // 500 MB

/// Check if a data source is large enough to warrant mmap.
fn is_large_source(path: &str) -> bool {
    let p = std::path::Path::new(path);
    if p.is_dir() {
        // Sum file sizes in directory
        let total: u64 = std::fs::read_dir(path).ok()
            .map(|entries| entries.filter_map(|e| e.ok())
                .filter_map(|e| e.metadata().ok().map(|m| m.len()))
                .sum())
            .unwrap_or(0);
        total > LARGE_FILE_THRESHOLD
    } else {
        p.metadata().map(|m| m.len() > LARGE_FILE_THRESHOLD).unwrap_or(false)
    }
}

/// Load data with automatic mmap for large files.
/// For files > 500MB: tokenize once to .wtok, then mmap for all subsequent runs.
/// Returns either in-memory tokens or a path to the .wtok file for mmap loading.
pub fn load_data_auto(path: &str, use_bpe: bool, tokenizer_path: Option<&str>) -> DataTokens {
    let wtok_path = format!("{}.wtok", path.trim_end_matches('/'));

    // Check if .wtok already exists and is newer than source
    let wtok_exists = std::path::Path::new(&wtok_path).exists();
    let source_newer = if wtok_exists {
        let wtok_mod = std::fs::metadata(&wtok_path).ok().and_then(|m| m.modified().ok());
        let src_mod = std::fs::metadata(path).ok().and_then(|m| m.modified().ok());
        match (wtok_mod, src_mod) {
            (Some(w), Some(s)) => s > w,
            _ => true, // rebuild if can't determine
        }
    } else {
        false
    };

    if wtok_exists && !source_newer {
        // Load from existing .wtok via mmap
        let (mmap_tokens, vocab_size) = load_wtok_mmap(&wtok_path);
        return DataTokens::Mmap { tokens: mmap_tokens, vocab_size };
    }

    // Load normally (in-memory)
    let (tokens, vocab_size) = load_data(path, use_bpe, tokenizer_path);

    // If large, save .wtok for future mmap access
    if is_large_source(path) || tokens.len() > 50_000_000 {
        save_wtok(&wtok_path, &tokens, vocab_size);
        // Reload via mmap to free the Vec
        let (mmap_tokens, vs) = load_wtok_mmap(&wtok_path);
        DataTokens::Mmap { tokens: mmap_tokens, vocab_size: vs }
    } else {
        DataTokens::InMemory { tokens, vocab_size }
    }
}

/// Token data — either in-memory Vec or memory-mapped file.
pub enum DataTokens {
    InMemory { tokens: Vec<usize>, vocab_size: usize },
    Mmap { tokens: MmapTokens, vocab_size: usize },
}

impl DataTokens {
    pub fn vocab_size(&self) -> usize {
        match self { Self::InMemory { vocab_size, .. } | Self::Mmap { vocab_size, .. } => *vocab_size }
    }

    pub fn total_tokens(&self) -> usize {
        match self {
            Self::InMemory { tokens, .. } => tokens.len(),
            Self::Mmap { tokens, .. } => tokens.n_tokens,
        }
    }

    /// Get a slice of tokens (in-memory) or a window (mmap).
    pub fn get_window(&self, start: usize, len: usize) -> Vec<usize> {
        match self {
            Self::InMemory { tokens, .. } => tokens[start..start + len].to_vec(),
            Self::Mmap { tokens, .. } => tokens.window_vec(start, len),
        }
    }
}

/// Scan data source for vocab size and corpus size without full tokenization.
/// Used by --recommend for fast analysis of any format.
pub fn analyze_data_source(path: &str) -> (usize, usize) {
    let p = std::path::Path::new(path);

    let text = if p.is_dir() {
        load_directory_text(path)
    } else {
        let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("");
        match ext {
            "jsonl" => load_jsonl_text(path),
            _ => std::fs::read_to_string(path)
                .unwrap_or_else(|e| { eprintln!("Error reading {}: {}", path, e); std::process::exit(1); }),
        }
    };

    let corpus_chars = text.len();
    let vocab_size: usize = text.chars().collect::<BTreeSet<char>>().len();
    (corpus_chars, vocab_size)
}
