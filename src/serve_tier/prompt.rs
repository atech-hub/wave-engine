//! Chat formatting and tokenization for serving.

use crate::bpe::BpeTokenizer;
use super::api_types::ChatMessage;

/// Vocabulary for serving — encode/decode text to/from token IDs.
pub struct Vocab {
    pub vocab_size: usize,
    pub bpe: Option<BpeTokenizer>,
    /// Char-level fallback: char→index mapping
    pub char_map: Vec<char>,
}

impl Vocab {
    pub fn from_bpe(bpe: BpeTokenizer, vocab_size: usize) -> Self {
        Self { vocab_size, bpe: Some(bpe), char_map: Vec::new() }
    }

    pub fn from_chars(vocab_size: usize) -> Self {
        let mut chars: Vec<char> = (0..128u8).filter_map(|b| {
            let c = b as char;
            if c.is_ascii() { Some(c) } else { None }
        }).collect();
        chars.sort();
        chars.dedup();
        chars.truncate(vocab_size);
        Self { vocab_size, bpe: None, char_map: chars }
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        if let Some(ref bpe) = self.bpe {
            bpe.encode(text)
        } else {
            text.chars()
                .filter_map(|c| self.char_map.iter().position(|&ch| ch == c))
                .collect()
        }
    }

    pub fn decode(&self, tokens: &[usize]) -> String {
        if let Some(ref bpe) = self.bpe {
            bpe.decode(tokens)
        } else {
            tokens.iter()
                .map(|&t| if t < self.char_map.len() { self.char_map[t].to_string() } else { "?".to_string() })
                .collect::<Vec<_>>()
                .join("")
        }
    }
}

/// Format chat messages into a token sequence.
/// Separator "\n\n" goes BETWEEN messages, not after the last one — the last
/// message should sit at the cursor so the model continues directly from it
/// (so completion-style single-user prompts like "88-65=" work without the
/// formatter appending tokens that put the model in an off-training-distribution
/// state).
pub fn format_chat(messages: &[ChatMessage], vocab: &Vocab) -> Vec<usize> {
    let mut text = String::new();
    let mut first = true;
    for msg in messages {
        if !matches!(msg.role.as_str(), "system" | "user" | "assistant") {
            continue;
        }
        if !first {
            text.push_str("\n\n");
        }
        text.push_str(&msg.content);
        first = false;
    }
    vocab.encode(&text)
}
