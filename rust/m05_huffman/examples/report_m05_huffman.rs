//! Prints the numbers cited in this slice's README.
//! Run: cargo run -p m05_huffman --example report_m05_huffman

use m05_huffman::{average_length, build_codes, decode, encode, entropy, frequencies};

const CORPUS: &str = "the quick brown fox jumps over the lazy dog. \
the dog barks and the fox runs away. \
compression is prediction and prediction is intelligence. \
a prefix code assigns shorter codewords to more frequent symbols.";

fn report(label: &str, data: &[u8]) {
    let freqs = frequencies(data);
    let codes = build_codes(&freqs);
    let bits = encode(data, &codes);
    let ok = decode(&bits, &codes) == data;

    let total: u64 = freqs.values().sum();
    let probs: Vec<f64> = freqs.values().map(|&f| f as f64 / total as f64).collect();
    let h = entropy(&probs);
    let l = average_length(&codes, &freqs);

    println!("== {label} ==");
    println!("  symbols            : {}", freqs.len());
    println!("  bytes              : {}", data.len());
    println!("  entropy H          : {h:.4} bit/symbol");
    println!("  Huffman L          : {l:.4} bit/symbol");
    println!("  H <= L < H + 1     : {}", l >= h && l < h + 1.0);
    println!("  fixed 8-bit        : 8.0000 bit/symbol");
    println!("  ratio vs fixed     : {:.2}%", 100.0 * l / 8.0);
    println!("  roundtrip          : {ok}");
    println!("  longest codeword   : {}", codes.values().map(|c| c.len()).max().unwrap_or(0));
    println!();
}

fn main() {
    report("sample English corpus", CORPUS.as_bytes());

    // Skewed 5-symbol source: H is far below the 3 bits a fixed-width code needs.
    let mut freqs = std::collections::BTreeMap::new();
    for (sym, f) in [(b'a', 900u64), (b'b', 50), (b'c', 30), (b'd', 15), (b'e', 5)] {
        freqs.insert(sym, f);
    }
    let codes = build_codes(&freqs);
    let total: u64 = freqs.values().sum();
    let probs: Vec<f64> = freqs.values().map(|&f| f as f64 / total as f64).collect();
    println!("== skewed 5-symbol source ==");
    println!("  entropy H          : {:.4} bit/symbol", entropy(&probs));
    println!("  Huffman L          : {:.4} bit/symbol", average_length(&codes, &freqs));
    println!("  fixed 3-bit        : 3.0000 bit/symbol");
}
