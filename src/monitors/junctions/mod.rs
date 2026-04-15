//! Junction monitors — verify contracts at component boundaries.
//!
//! Each monitor watches one specific handoff between subsystems.
//! Component monitors (parent folder) observe internals;
//! junction monitors observe the contracts between them.

pub mod grad_check;        // J1: analytical vs numerical gradient agreement
#[cfg(test)]
mod grad_check_test;       // J1 self-tests
pub mod vector_length;     // J5: params.len() == count_trainable == grads.len()
