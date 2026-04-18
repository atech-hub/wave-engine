//! Junction monitors — verify contracts at component boundaries.
//!
//! Each monitor watches one specific handoff between subsystems.
//! Component monitors (parent folder) observe internals;
//! junction monitors observe the contracts between them.

pub mod grad_check;        // J1: analytical vs numerical gradient agreement
pub mod param_completeness; // J2: every weight is trainable or explicitly frozen
#[cfg(test)]
mod grad_check_test;       // J1 self-tests
pub mod roundtrip_integrity; // J4: flatten/unflatten/flatten = identity
pub mod vector_length;     // J5: params.len() == count_trainable == grads.len()
pub mod live_gradient;     // J6: every trainable param sees nonzero gradient
pub mod pathway_completeness; // J3: forward fan-out == backward fan-in
pub mod tensor_shape;       // J9: tensor dimensions match at every boundary
pub mod value_range;        // J7: named tensors must stay within declared bounds
pub mod train_infer_alignment; // J8: train loss vs inference metric correlation tracker
pub mod tier_parity;        // J10: CPU/GPU forward outputs must agree to tolerance
