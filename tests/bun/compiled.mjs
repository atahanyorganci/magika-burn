// Built into a standalone executable by tests/bun/compile.sh. Importing the
// .wasm file `with { type: "file" }` embeds it in the executable.
import { readFileSync } from "node:fs";
import wasm from "@yorganci/magika-burn/magika_bg.wasm" with { type: "file" };
import { initSync, MagikaModel } from "@yorganci/magika-burn/web";

initSync({ module: readFileSync(wasm) });
const source = `use std::collections::HashMap;

/// Counts how often each word appears in the input.
fn word_counts(text: &str) -> HashMap<&str, usize> {
    let mut counts = HashMap::new();
    for word in text.split_whitespace() {
        *counts.entry(word).or_insert(0) += 1;
    }
    counts
}

fn main() {
    let counts = word_counts("the quick brown fox jumps over the lazy dog");
    for (word, count) in &counts {
        println!("{word}: {count}");
    }
}
`;
console.log(new MagikaModel().identifyBytes(new TextEncoder().encode(source)).output.label);
