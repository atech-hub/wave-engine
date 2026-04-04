//! OutProjWeights — Dense or block-diagonal output projection.
//!
//! Extracted from model.rs. The hub abstraction: all consumers use methods,
//! never pattern-match on the variant directly.

use super::model::{LinearWeights, linear_fn};

/// Block-diagonal linear — groups of bands processed independently.
#[derive(Clone)]
pub struct BlockDiagonalWeights {
    pub groups: Vec<LinearWeights>,
    pub n_groups: usize,
    pub group_size: usize,
}

/// Abstract out_proj — dense or block-diagonal.
#[derive(Clone)]
pub enum OutProjWeights {
    Dense(LinearWeights),
    BlockDiagonal(BlockDiagonalWeights),
}

impl OutProjWeights {
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        match self {
            Self::Dense(lw) => linear_fn(&lw.w, &lw.b, x),
            Self::BlockDiagonal(bd) => {
                let mut out = vec![0.0f32; x.len()];
                for (g, group) in bd.groups.iter().enumerate() {
                    let start = g * bd.group_size;
                    for i in 0..bd.group_size {
                        let mut sum = group.b[i];
                        for j in 0..bd.group_size { sum += group.w[i][j] * x[start + j]; }
                        out[start + i] = sum;
                    }
                }
                out
            }
        }
    }

    pub fn forward_batch(&self, xs: &[Vec<f32>]) -> Vec<Vec<f32>> {
        xs.iter().map(|x| self.forward(x)).collect()
    }

    pub fn param_count(&self) -> usize {
        match self {
            Self::Dense(lw) => lw.w.len() * lw.w[0].len() + lw.b.len(),
            Self::BlockDiagonal(bd) => bd.n_groups * (bd.group_size * bd.group_size + bd.group_size),
        }
    }

    pub fn flatten_into(&self, out: &mut Vec<f32>) {
        match self {
            Self::Dense(lw) => {
                for row in &lw.w { out.extend_from_slice(row); }
                out.extend_from_slice(&lw.b);
            }
            Self::BlockDiagonal(bd) => {
                for group in &bd.groups {
                    for row in &group.w { out.extend_from_slice(row); }
                    out.extend_from_slice(&group.b);
                }
            }
        }
    }

    pub fn unflatten_from(&mut self, params: &[f32], offset: &mut usize) {
        match self {
            Self::Dense(lw) => {
                let dim = lw.w[0].len();
                let blen = lw.b.len();
                for row in &mut lw.w { row.copy_from_slice(&params[*offset..*offset+dim]); *offset += dim; }
                lw.b.copy_from_slice(&params[*offset..*offset+blen]); *offset += blen;
            }
            Self::BlockDiagonal(bd) => {
                let gs = bd.group_size;
                for group in &mut bd.groups {
                    for row in &mut group.w { row.copy_from_slice(&params[*offset..*offset+gs]); *offset += gs; }
                    group.b.copy_from_slice(&params[*offset..*offset+gs]); *offset += gs;
                }
            }
        }
    }

    pub fn dim(&self) -> usize {
        match self {
            Self::Dense(lw) => lw.w.len(),
            Self::BlockDiagonal(bd) => bd.n_groups * bd.group_size,
        }
    }

    pub fn n_groups(&self) -> usize {
        match self {
            Self::Dense(_) => 1,
            Self::BlockDiagonal(bd) => bd.n_groups,
        }
    }

    pub fn group_size(&self) -> usize {
        match self {
            Self::Dense(lw) => lw.w.len(),
            Self::BlockDiagonal(bd) => bd.group_size,
        }
    }

    /// Flat weight buffer for GPU upload
    pub fn weights_flat(&self) -> Vec<f32> {
        let mut out = Vec::new();
        match self {
            Self::Dense(lw) => { for row in &lw.w { out.extend_from_slice(row); } }
            Self::BlockDiagonal(bd) => { for g in &bd.groups { for row in &g.w { out.extend_from_slice(row); } } }
        }
        out
    }

    /// Flat bias buffer for GPU upload
    pub fn bias_flat(&self) -> Vec<f32> {
        match self {
            Self::Dense(lw) => lw.b.clone(),
            Self::BlockDiagonal(bd) => bd.groups.iter().flat_map(|g| g.b.iter().copied()).collect(),
        }
    }

    /// Backward: d_x = W^T @ d_y per group
    pub fn backward_dx(&self, d_y: &[f32]) -> Vec<f32> {
        match self {
            Self::Dense(lw) => {
                let n = lw.w.len();
                let m = lw.w[0].len();
                let mut dx = vec![0.0f32; m];
                for j in 0..m { for i in 0..n { dx[j] += lw.w[i][j] * d_y[i]; } }
                dx
            }
            Self::BlockDiagonal(bd) => {
                let mut dx = vec![0.0f32; bd.n_groups * bd.group_size];
                for (g, group) in bd.groups.iter().enumerate() {
                    let start = g * bd.group_size;
                    let gs = bd.group_size;
                    for j in 0..gs { for i in 0..gs { dx[start+j] += group.w[i][j] * d_y[start+i]; } }
                }
                dx
            }
        }
    }

    /// Create Dense variant
    pub fn dense(w: Vec<Vec<f32>>, b: Vec<f32>) -> Self {
        Self::Dense(LinearWeights { w, b })
    }

    /// Temporary accessor for legacy code that expects LinearWeights.
    /// Returns the Dense inner or panics for BlockDiagonal.
    /// TODO: Remove when legacy files are deleted in Phase 4.
    pub fn as_linear(&self) -> &LinearWeights {
        match self {
            Self::Dense(lw) => lw,
            Self::BlockDiagonal(_) => panic!("as_linear() called on BlockDiagonal — legacy code path with non-dense out_proj"),
        }
    }

    pub fn as_linear_mut(&mut self) -> &mut LinearWeights {
        match self {
            Self::Dense(lw) => lw,
            Self::BlockDiagonal(_) => panic!("as_linear_mut() called on BlockDiagonal"),
        }
    }

    /// Backward: d_W and d_b (accumulated over positions)
    pub fn backward_dw_db(&self, d_y: &[Vec<f32>], x: &[Vec<f32>]) -> (Vec<Vec<f32>>, Vec<f32>) {
        let dim = self.dim();
        let mut d_w = vec![vec![0.0f32; dim]; dim];
        let mut d_b = vec![0.0f32; dim];
        for pos in 0..d_y.len() {
            for i in 0..dim {
                for j in 0..dim { d_w[i][j] += d_y[pos][i] * x[pos][j]; }
                d_b[i] += d_y[pos][i];
            }
        }
        (d_w, d_b)
    }
}
