// KRAIN AIVM Wasmer host functions.
//
// Registered under the "krain_aivm" WASM import module in
// crates/stylus/src/native.rs.  Called by Stylus WASM contracts
// that declare `#[link(wasm_import_module = "krain_aivm")]` imports
// (e.g. contracts/src/mnist-agent/src/aivm.rs in axon-protocol).
//
// Each function:
//   1. Reads `pre_len` bytes from WASM linear memory at `pre_ptr`.
//   2. Delegates to the `krain_aivm` crate for the tensor computation.
//   3. Writes the output bytes to WASM linear memory at `out_ptr`.
//   4. Returns bytes written (>= 0) or -1 on error.
//
// Weights are loaded once at node startup from the path in
// KRAIN_AIVM_WEIGHTS (default: /config/aivm-weights/mnist_mlp_v1.bin).

use arbutil::evm::api::{DataReader, EvmApi};
use caller_env::GuestPtr;

use crate::env::WasmEnvMut;

/// Initialise AIVM weights.  Called once from node startup.
pub fn init_weights() {
    let path = std::env::var("KRAIN_AIVM_WEIGHTS")
        .unwrap_or_else(|_| "/config/aivm-weights/mnist_mlp_v1.bin".to_string());
    if let Err(e) = krain_aivm::init(&path) {
        eprintln!("krain-aivm: WARNING — {e}");
        eprintln!("krain-aivm: AIVM host functions will return -1 until weights are loaded.");
    }
}

// ── Memory helpers ────────────────────────────────────────────────────────────────────────────────

fn read_mem<D: DataReader, E: EvmApi<D>>(
    env: &mut WasmEnvMut<D, E>,
    ptr: GuestPtr,
    len: usize,
) -> Option<Vec<u8>> {
    let (data, store) = env.data_and_store_mut();
    let memory = data.memory.as_ref()?.clone();
    let view = memory.view(store);
    let mut buf = vec![0u8; len];
    view.read(ptr as u64, &mut buf).ok()?;
    Some(buf)
}

fn write_mem<D: DataReader, E: EvmApi<D>>(
    env: &mut WasmEnvMut<D, E>,
    ptr: GuestPtr,
    data_bytes: &[u8],
) -> bool {
    let (data, store) = env.data_and_store_mut();
    let memory = match data.memory.as_ref() {
        Some(m) => m.clone(),
        None => return false,
    };
    let view = memory.view(store);
    view.write(ptr as u64, data_bytes).is_ok()
}

// ── Dispatch helper ─────────────────────────────────────────────────────────────────────────────────

fn dispatch<D, E, F>(
    env: &mut WasmEnvMut<D, E>,
    model_id: u64,
    pre_ptr: GuestPtr,
    pre_len: u32,
    out_ptr: GuestPtr,
    out_max: u32,
    op: F,
) -> i32
where
    D: DataReader,
    E: EvmApi<D>,
    F: FnOnce(u64, &[u8]) -> Result<Vec<u8>, &'static str>,
{
    let pre = match read_mem(env, pre_ptr, pre_len as usize) {
        Some(b) => b,
        None => return -1,
    };
    let output = match op(model_id, &pre) {
        Ok(out) => out,
        Err(e) => {
            eprintln!("krain-aivm: {e}");
            return -1;
        }
    };
    if output.len() > out_max as usize {
        return -1;
    }
    if !write_mem(env, out_ptr, &output) {
        return -1;
    }
    output.len() as i32
}

// ── Exported host functions ────────────────────────────────────────────────────────────────────────────

pub fn aivm_fc1_linear<D: DataReader, E: EvmApi<D>>(
    mut env: WasmEnvMut<D, E>,
    model_id: u64,
    pre_ptr: GuestPtr,
    pre_len: u32,
    out_ptr: GuestPtr,
    out_max: u32,
) -> i32 {
    dispatch(&mut env, model_id, pre_ptr, pre_len, out_ptr, out_max, krain_aivm::fc1_linear)
}

pub fn aivm_relu_512<D: DataReader, E: EvmApi<D>>(
    mut env: WasmEnvMut<D, E>,
    model_id: u64,
    pre_ptr: GuestPtr,
    pre_len: u32,
    out_ptr: GuestPtr,
    out_max: u32,
) -> i32 {
    dispatch(&mut env, model_id, pre_ptr, pre_len, out_ptr, out_max, krain_aivm::relu_512)
}

pub fn aivm_dropout_noop_512<D: DataReader, E: EvmApi<D>>(
    mut env: WasmEnvMut<D, E>,
    model_id: u64,
    pre_ptr: GuestPtr,
    pre_len: u32,
    out_ptr: GuestPtr,
    out_max: u32,
) -> i32 {
    dispatch(&mut env, model_id, pre_ptr, pre_len, out_ptr, out_max, krain_aivm::dropout_noop_512)
}

pub fn aivm_fc2_linear<D: DataReader, E: EvmApi<D>>(
    mut env: WasmEnvMut<D, E>,
    model_id: u64,
    pre_ptr: GuestPtr,
    pre_len: u32,
    out_ptr: GuestPtr,
    out_max: u32,
) -> i32 {
    dispatch(&mut env, model_id, pre_ptr, pre_len, out_ptr, out_max, krain_aivm::fc2_linear)
}

pub fn aivm_relu_128<D: DataReader, E: EvmApi<D>>(
    mut env: WasmEnvMut<D, E>,
    model_id: u64,
    pre_ptr: GuestPtr,
    pre_len: u32,
    out_ptr: GuestPtr,
    out_max: u32,
) -> i32 {
    dispatch(&mut env, model_id, pre_ptr, pre_len, out_ptr, out_max, krain_aivm::relu_128)
}

pub fn aivm_dropout_noop_128<D: DataReader, E: EvmApi<D>>(
    mut env: WasmEnvMut<D, E>,
    model_id: u64,
    pre_ptr: GuestPtr,
    pre_len: u32,
    out_ptr: GuestPtr,
    out_max: u32,
) -> i32 {
    dispatch(&mut env, model_id, pre_ptr, pre_len, out_ptr, out_max, krain_aivm::dropout_noop_128)
}

pub fn aivm_fc3_linear<D: DataReader, E: EvmApi<D>>(
    mut env: WasmEnvMut<D, E>,
    model_id: u64,
    pre_ptr: GuestPtr,
    pre_len: u32,
    out_ptr: GuestPtr,
    out_max: u32,
) -> i32 {
    dispatch(&mut env, model_id, pre_ptr, pre_len, out_ptr, out_max, krain_aivm::fc3_linear)
}

pub fn aivm_argmax_result<D: DataReader, E: EvmApi<D>>(
    mut env: WasmEnvMut<D, E>,
    model_id: u64,
    pre_ptr: GuestPtr,
    pre_len: u32,
    out_ptr: GuestPtr,
    out_max: u32,
) -> i32 {
    dispatch(&mut env, model_id, pre_ptr, pre_len, out_ptr, out_max, krain_aivm::argmax_result)
}
