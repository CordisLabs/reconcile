//! Kolmogorov Complexity and Algorithmic Randomness (Shen, Uspensky,
//! Vereshchagin) — Huffman coding, implemented against the definition of an
//! optimal prefix code. Zero dependencies.
//!
//! The point of the tests is not "the file runs" but "the code length is provably
//! pinned": Shannon's source-coding theorem gives H <= L < H + 1 for a Huffman code
//! of a memoryless source, and Kraft's inequality must hold exactly.

#![forbid(unsafe_code)]

use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap, HashMap};

enum Node {
    Leaf(u8),
    Inner(usize, usize),
}

/// Shannon entropy of a distribution, in bits.
pub fn entropy(probs: &[f64]) -> f64 {
    probs
        .iter()
        .filter(|p| **p > 0.0)
        .map(|p| -p * p.log2())
        .sum()
}

pub fn frequencies(data: &[u8]) -> BTreeMap<u8, u64> {
    let mut f = BTreeMap::new();
    for &b in data {
        *f.entry(b).or_insert(0) += 1;
    }
    f
}

/// Build an optimal prefix code. A single-symbol alphabet gets the empty code
/// (length 0), which is the information-theoretically correct answer.
pub fn build_codes(freqs: &BTreeMap<u8, u64>) -> HashMap<u8, Vec<bool>> {
    let mut nodes: Vec<Node> = Vec::new();
    let mut heap: BinaryHeap<Reverse<(u64, usize)>> = BinaryHeap::new();

    for (&sym, &f) in freqs.iter().filter(|(_, f)| **f > 0) {
        nodes.push(Node::Leaf(sym));
        heap.push(Reverse((f, nodes.len() - 1)));
    }

    match nodes.len() {
        0 => return HashMap::new(),
        1 => {
            let mut out = HashMap::new();
            if let Node::Leaf(sym) = nodes[0] {
                out.insert(sym, Vec::new());
            }
            return out;
        }
        _ => {}
    }

    while heap.len() > 1 {
        let Reverse((f1, n1)) = heap.pop().expect("heap non-empty");
        let Reverse((f2, n2)) = heap.pop().expect("heap has two nodes");
        nodes.push(Node::Inner(n1, n2));
        heap.push(Reverse((f1 + f2, nodes.len() - 1)));
    }

    let Reverse((_, root)) = heap.pop().expect("one root remains");
    let mut codes = HashMap::new();
    assign(&nodes, root, &mut Vec::new(), &mut codes);
    codes
}

fn assign(nodes: &[Node], idx: usize, prefix: &mut Vec<bool>, out: &mut HashMap<u8, Vec<bool>>) {
    match nodes[idx] {
        Node::Leaf(sym) => {
            out.insert(sym, prefix.clone());
        }
        Node::Inner(left, right) => {
            prefix.push(false);
            assign(nodes, left, prefix, out);
            prefix.pop();
            prefix.push(true);
            assign(nodes, right, prefix, out);
            prefix.pop();
        }
    }
}

pub fn encode(data: &[u8], codes: &HashMap<u8, Vec<bool>>) -> Vec<bool> {
    let mut bits = Vec::new();
    for &b in data {
        bits.extend_from_slice(&codes[&b]);
    }
    bits
}

pub fn decode(bits: &[bool], codes: &HashMap<u8, Vec<bool>>) -> Vec<u8> {
    let lookup: HashMap<(usize, u64), u8> = codes
        .iter()
        .map(|(&sym, code)| {
            let value = code.iter().fold(0u64, |acc, &b| (acc << 1) | b as u64);
            ((code.len(), value), sym)
        })
        .collect();

    let max_len = codes.values().map(|c| c.len()).max().unwrap_or(0);
    assert!(max_len <= 63, "u64 packing supports at most 63-bit codes");

    let mut out = Vec::new();
    let mut acc = 0u64;
    let mut len = 0usize;
    for &b in bits {
        acc = (acc << 1) | b as u64;
        len += 1;
        if let Some(&sym) = lookup.get(&(len, acc)) {
            out.push(sym);
            acc = 0;
            len = 0;
        } else if len > max_len {
            panic!("bitstream is not a valid code sequence");
        }
    }
    assert_eq!(len, 0, "bitstream ends mid-code (truncated)");
    out
}

/// Average code length in bits per symbol under the given frequencies.
pub fn average_length(codes: &HashMap<u8, Vec<bool>>, freqs: &BTreeMap<u8, u64>) -> f64 {
    let total: u64 = freqs.values().sum();
    let mut bits = 0u64;
    for (&sym, &f) in freqs {
        if f > 0 {
            bits += f * codes[&sym].len() as u64;
        }
    }
    bits as f64 / total as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn probs_from(freqs: &BTreeMap<u8, u64>) -> Vec<f64> {
        let total: u64 = freqs.values().sum();
        freqs.values().map(|&f| f as f64 / total as f64).collect()
    }

    fn sample_freqs() -> BTreeMap<u8, u64> {
        let mut f = BTreeMap::new();
        f.insert(b'a', 900);
        f.insert(b'b', 50);
        f.insert(b'c', 30);
        f.insert(b'd', 15);
        f.insert(b'e', 5);
        f
    }

    #[test]
    fn kraft_inequality_is_tight() {
        let freqs = sample_freqs();
        let codes = build_codes(&freqs);
        let kraft: f64 = codes.values().map(|c| 0.5f64.powi(c.len() as i32)).sum();
        assert!((kraft - 1.0).abs() < 1e-12, "kraft sum = {kraft}");
    }

    #[test]
    fn prefix_free() {
        let codes = build_codes(&sample_freqs());
        let all: Vec<(u8, &Vec<bool>)> = codes.iter().map(|(&s, c)| (s, c)).collect();
        for i in 0..all.len() {
            for j in 0..all.len() {
                if i == j {
                    continue;
                }
                let (_, a) = all[i];
                let (_, b) = all[j];
                assert!(
                    !(a.len() <= b.len() && b[..a.len()] == a[..]),
                    "code for {} is a prefix of {}",
                    all[i].0 as char,
                    all[j].0 as char
                );
            }
        }
    }

    #[test]
    fn shannon_bound_holds() {
        let freqs = sample_freqs();
        let codes = build_codes(&freqs);
        let h = entropy(&probs_from(&freqs));
        let l = average_length(&codes, &freqs);
        assert!(l >= h - 1e-12, "Huffman beat entropy: L={l} H={h}");
        assert!(l < h + 1.0, "Huffman exceeded H+1: L={l} H={h}");
    }

    #[test]
    fn beats_fixed_width_on_skewed_source() {
        let freqs = sample_freqs();
        let codes = build_codes(&freqs);
        let l = average_length(&codes, &freqs);
        // Naive encoding would spend ceil(log2(5)) = 3 bits on every symbol.
        assert!(l < 3.0, "no gain over fixed width: L={l}");
    }

    #[test]
    fn roundtrip() {
        let text = b"the quick brown fox jumps over the lazy dog, again and again";
        let freqs = frequencies(text);
        let codes = build_codes(&freqs);
        let bits = encode(text, &codes);
        assert_eq!(decode(&bits, &codes), text.to_vec());
    }

    #[test]
    fn single_symbol_alphabet() {
        let mut freqs = BTreeMap::new();
        freqs.insert(b'x', 42);
        let codes = build_codes(&freqs);
        assert_eq!(codes[&b'x'].len(), 0);
        let h = entropy(&probs_from(&freqs));
        assert_eq!(h, 0.0);
        assert_eq!(average_length(&codes, &freqs), 0.0);
    }
}
