// KRAIN AIVM host function implementations -- MNIST MLP forward pass.
//
// Compiled as part of the KRAIN Nitro fork.  The Stylus crate (crates/stylus/)
// calls into this crate from host functions registered under the "krain_aivm"
// WASM import module.
//
// Determinism guarantee: all loops use sequential scalar f32 arithmetic.
// No SIMD intrinsics, no parallel iterators, no FMA.

use std::sync::OnceLock;

// -- Constants -----------------------------------------------------------------

pub const FC1_IN: usize = 784;
pub const FC1_OUT: usize = 512;
pub const FC2_OUT: usize = 128;
pub const FC3_OUT: usize = 10;

// -- Weight storage ------------------------------------------------------------

struct Weights {
    fc1_w: Vec<f32>,
    fc1_b: Vec<f32>,
    fc2_w: Vec<f32>,
    fc2_b: Vec<f32>,
    fc3_w: Vec<f32>,
    fc3_b: Vec<f32>,
}

static WEIGHTS: OnceLock<Weights> = OnceLock::new();
/// Ensures exactly one init attempt from the env-var path.
static INIT_ATTEMPTED: OnceLock<()> = OnceLock::new();

/// Load MNIST MLP weights from the flat binary at `path`.
///
/// Format (little-endian f32 blocks, row-major):
///   fc1_w [512x784], fc1_b [512], fc2_w [128x512], fc2_b [128],
///   fc3_w [10x128],  fc3_b [10]
pub fn init(path: &str) -> Result<(), String> {
    let data = std::fs::read(path)
        .map_err(|e| format!("krain-aivm: failed to read {path}: {e}"))?;

    let expected = (FC1_OUT * FC1_IN + FC1_OUT
        + FC2_OUT * FC1_OUT + FC2_OUT
        + FC3_OUT * FC2_OUT + FC3_OUT) * 4;

    if data.len() != expected {
        return Err(format!(
            "krain-aivm: weights {path} is {} bytes, expected {expected}",
            data.len()
        ));
    }

    let floats: Vec<f32> = data
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let mut offset = 0;
    macro_rules! take {
        ($n:expr) => {{
            let v = floats[offset..offset + $n].to_vec();
            offset += $n;
            v
        }};
    }
    let w = Weights {
        fc1_w: take!(FC1_OUT * FC1_IN),
        fc1_b: take!(FC1_OUT),
        fc2_w: take!(FC2_OUT * FC1_OUT),
        fc2_b: take!(FC2_OUT),
        fc3_w: take!(FC3_OUT * FC2_OUT),
        fc3_b: take!(FC3_OUT),
    };
    WEIGHTS.set(w).map_err(|_| "krain-aivm: already initialised".to_string())
}

/// Returns the loaded weights, attempting a lazy load from `KRAIN_AIVM_WEIGHTS`
/// env var on the first call.
fn weights() -> Option<&'static Weights> {
    INIT_ATTEMPTED.get_or_init(|| {
        let path = std::env::var("KRAIN_AIVM_WEIGHTS")
            .unwrap_or_else(|_| "/config/aivm-weights/mnist_mlp_v1.bin".to_string());
        if let Err(e) = init(&path) {
            eprintln!("krain-aivm: auto-init failed: {e}");
            eprintln!("krain-aivm: set KRAIN_AIVM_WEIGHTS to the weights file path");
        }
    });
    WEIGHTS.get()
}

// -- Tensor operations ---------------------------------------------------------

fn linear(input: &[f32], weight: &[f32], bias: &[f32], out: &mut Vec<f32>) {
    let in_feat = input.len();
    let out_feat = bias.len();
    out.clear();
    out.reserve(out_feat);
    for j in 0..out_feat {
        let mut acc: f32 = bias[j];
        for i in 0..in_feat {
            acc += weight[j * in_feat + i] * input[i];
        }
        out.push(acc);
    }
}

fn relu(input: &[f32], out: &mut Vec<f32>) {
    out.clear();
    out.reserve(input.len());
    for &x in input {
        out.push(if x > 0.0 { x } else { 0.0 });
    }
}

fn argmax(logits: &[f32]) -> usize {
    let mut best = 0;
    let mut best_val = logits[0];
    for (i, &v) in logits.iter().enumerate().skip(1) {
        if v > best_val { best_val = v; best = i; }
    }
    best
}

// -- Encoding helpers ----------------------------------------------------------

fn decode_f32_le(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn encode_f32_le(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 4);
    for &v in vals { out.extend_from_slice(&v.to_le_bytes()); }
    out
}

// -- Public API ----------------------------------------------------------------

/// Step 1: FC1 linear -- input [784xf32] -> output [512xf32].
pub fn fc1_linear(_model_id: u64, pre: &[u8]) -> Result<Vec<u8>, &'static str> {
    let w = weights().ok_or("krain-aivm: not initialised")?;
    if pre.len() != FC1_IN * 4 { return Err("krain-aivm: fc1_linear: wrong input length"); }
    let input = decode_f32_le(pre);
    let mut output = Vec::new();
    linear(&input, &w.fc1_w, &w.fc1_b, &mut output);
    Ok(encode_f32_le(&output))
}

/// Step 2: ReLU activation -- [512xf32] -> [512xf32].
pub fn relu_512(_model_id: u64, pre: &[u8]) -> Result<Vec<u8>, &'static str> {
    if pre.len() != FC1_OUT * 4 { return Err("krain-aivm: relu_512: wrong input length"); }
    let input = decode_f32_le(pre);
    let mut output = Vec::new();
    relu(&input, &mut output);
    Ok(encode_f32_le(&output))
}

/// Step 3: Dropout no-op (eval mode) -- [512xf32] -> [512xf32], identity.
pub fn dropout_noop_512(_model_id: u64, pre: &[u8]) -> Result<Vec<u8>, &'static str> {
    if pre.len() != FC1_OUT * 4 { return Err("krain-aivm: dropout_noop_512: wrong input length"); }
    Ok(pre.to_vec())
}

/// Step 4: FC2 linear -- [512xf32] -> [128xf32].
pub fn fc2_linear(_model_id: u64, pre: &[u8]) -> Result<Vec<u8>, &'static str> {
    let w = weights().ok_or("krain-aivm: not initialised")?;
    if pre.len() != FC1_OUT * 4 { return Err("krain-aivm: fc2_linear: wrong input length"); }
    let input = decode_f32_le(pre);
    let mut output = Vec::new();
    linear(&input, &w.fc2_w, &w.fc2_b, &mut output);
    Ok(encode_f32_le(&output))
}

/// Step 5: ReLU activation -- [128xf32] -> [128xf32].
pub fn relu_128(_model_id: u64, pre: &[u8]) -> Result<Vec<u8>, &'static str> {
    if pre.len() != FC2_OUT * 4 { return Err("krain-aivm: relu_128: wrong input length"); }
    let input = decode_f32_le(pre);
    let mut output = Vec::new();
    relu(&input, &mut output);
    Ok(encode_f32_le(&output))
}

/// Step 6: Dropout no-op (eval mode) -- [128xf32] -> [128xf32], identity.
pub fn dropout_noop_128(_model_id: u64, pre: &[u8]) -> Result<Vec<u8>, &'static str> {
    if pre.len() != FC2_OUT * 4 { return Err("krain-aivm: dropout_noop_128: wrong input length"); }
    Ok(pre.to_vec())
}

/// Step 7: FC3 linear -- [128xf32] -> [10xf32] logits.
pub fn fc3_linear(_model_id: u64, pre: &[u8]) -> Result<Vec<u8>, &'static str> {
    let w = weights().ok_or("krain-aivm: not initialised")?;
    if pre.len() != FC2_OUT * 4 { return Err("krain-aivm: fc3_linear: wrong input length"); }
    let input = decode_f32_le(pre);
    let mut output = Vec::new();
    linear(&input, &w.fc3_w, &w.fc3_b, &mut output);
    Ok(encode_f32_le(&output))
}

/// Step 8: Argmax + encode -- [10xf32] -> [digit_u8 | 10xf32 LE] = 41 bytes.
pub fn argmax_result(_model_id: u64, pre: &[u8]) -> Result<Vec<u8>, &'static str> {
    if pre.len() != FC3_OUT * 4 { return Err("krain-aivm: argmax_result: wrong input length"); }
    let logits = decode_f32_le(pre);
    let digit = argmax(&logits);
    let mut out = Vec::with_capacity(1 + FC3_OUT * 4);
    out.push(digit as u8);
    for &l in &logits { out.extend_from_slice(&l.to_le_bytes()); }
    Ok(out)
}
