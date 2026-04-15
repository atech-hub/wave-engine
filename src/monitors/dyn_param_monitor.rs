//! Dynamic Parameter Evolution Monitor (#7).
//!
//! Tracks trajectories of learnable parameters (layer_scale, rk4_weights,
//! wd_scale, lr_scale) with velocity (change since last snapshot) and
//! spring tension (distance from equilibrium).

use crate::WavePacketModel;

/// Snapshot of all dynamic parameters at one point in training.
pub struct DynParamSnapshot {
    pub layer_scale: Vec<f32>,
    pub rk4_weights: Vec<[f32; 4]>,
    pub wd_scale: Vec<f32>,
    pub lr_scale: Vec<f32>,
    pub layer_scale_velocity: Vec<f32>,
    pub rk4_velocity: Vec<[f32; 4]>,
    pub wd_velocity: Vec<f32>,
    pub layer_scale_tension: Vec<f32>,
    pub rk4_tension: Vec<[f32; 4]>,
    pub wd_tension: Vec<f32>,
}

/// Equilibrium values — where springs pull toward.
const LAYER_SCALE_EQ: f32 = 1.0;
const RK4_EQ: [f32; 4] = [1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0];
const WD_EQ: f32 = 1.0;

/// Take a snapshot of all dynamic parameters from the model.
/// If `prev` is provided, velocity is computed as (current - previous).
/// Otherwise velocity is zero.
pub fn snapshot(model: &WavePacketModel, prev: Option<&DynParamSnapshot>) -> DynParamSnapshot {
    let n_layers = model.blocks.len();

    // Current values
    let layer_scale: Vec<f32> = model.layer_scale.clone();
    let rk4_weights: Vec<[f32; 4]> = model.blocks.iter()
        .map(|b| b.ffn.kerr.rk4_weights)
        .collect();
    let wd_scale: Vec<f32> = model.wd_scale.clone();
    let lr_scale: Vec<f32> = model.lr_scale.clone();

    // Velocity: change since last snapshot
    let layer_scale_velocity = match prev {
        Some(p) if p.layer_scale.len() == layer_scale.len() => {
            layer_scale.iter().zip(p.layer_scale.iter())
                .map(|(&c, &p_val)| c - p_val).collect()
        }
        _ => vec![0.0; layer_scale.len()],
    };
    let rk4_velocity = match prev {
        Some(p) if p.rk4_weights.len() == rk4_weights.len() => {
            rk4_weights.iter().zip(p.rk4_weights.iter())
                .map(|(c, p_val)| {
                    let mut v = [0.0f32; 4];
                    for i in 0..4 { v[i] = c[i] - p_val[i]; }
                    v
                }).collect()
        }
        _ => vec![[0.0; 4]; n_layers],
    };
    let wd_velocity = match prev {
        Some(p) if p.wd_scale.len() == wd_scale.len() => {
            wd_scale.iter().zip(p.wd_scale.iter())
                .map(|(&c, &p_val)| c - p_val).collect()
        }
        _ => vec![0.0; wd_scale.len()],
    };

    // Spring tension: |current - equilibrium|
    let layer_scale_tension: Vec<f32> = layer_scale.iter()
        .map(|&s| (s - LAYER_SCALE_EQ).abs()).collect();
    let rk4_tension: Vec<[f32; 4]> = rk4_weights.iter()
        .map(|w| {
            let mut t = [0.0f32; 4];
            for i in 0..4 { t[i] = (w[i] - RK4_EQ[i]).abs(); }
            t
        }).collect();
    let wd_tension: Vec<f32> = wd_scale.iter()
        .map(|&s| (s - WD_EQ).abs()).collect();

    DynParamSnapshot {
        layer_scale,
        rk4_weights,
        wd_scale,
        lr_scale,
        layer_scale_velocity,
        rk4_velocity,
        wd_velocity,
        layer_scale_tension,
        rk4_tension,
        wd_tension,
    }
}

/// Serialize snapshot to a JSONL fragment (no outer braces — caller wraps).
/// Format: "dyn_params":{...}
pub fn to_json(snap: &DynParamSnapshot) -> String {
    let mut parts = Vec::new();

    // layer_scale
    if !snap.layer_scale.is_empty() {
        let vals: Vec<String> = snap.layer_scale.iter().map(|s| format!("{:.4}", s)).collect();
        parts.push(format!(r#""layer_scale":[{}]"#, vals.join(",")));
        let vels: Vec<String> = snap.layer_scale_velocity.iter().map(|v| format!("{:.6}", v)).collect();
        parts.push(format!(r#""layer_scale_vel":[{}]"#, vels.join(",")));
        let tens: Vec<String> = snap.layer_scale_tension.iter().map(|t| format!("{:.4}", t)).collect();
        parts.push(format!(r#""layer_scale_tension":[{}]"#, tens.join(",")));
    }

    // rk4_weights
    if !snap.rk4_weights.is_empty() {
        let vals: Vec<String> = snap.rk4_weights.iter().map(|w| {
            format!("[{:.4},{:.4},{:.4},{:.4}]", w[0], w[1], w[2], w[3])
        }).collect();
        parts.push(format!(r#""rk4_weights":[{}]"#, vals.join(",")));
        let vels: Vec<String> = snap.rk4_velocity.iter().map(|v| {
            format!("[{:.6},{:.6},{:.6},{:.6}]", v[0], v[1], v[2], v[3])
        }).collect();
        parts.push(format!(r#""rk4_vel":[{}]"#, vels.join(",")));
        let tens: Vec<String> = snap.rk4_tension.iter().map(|t| {
            format!("[{:.4},{:.4},{:.4},{:.4}]", t[0], t[1], t[2], t[3])
        }).collect();
        parts.push(format!(r#""rk4_tension":[{}]"#, tens.join(",")));
    }

    // wd_scale
    if !snap.wd_scale.is_empty() {
        let vals: Vec<String> = snap.wd_scale.iter().map(|s| format!("{:.4}", s)).collect();
        parts.push(format!(r#""wd_scale":[{}]"#, vals.join(",")));
        let vels: Vec<String> = snap.wd_velocity.iter().map(|v| format!("{:.6}", v)).collect();
        parts.push(format!(r#""wd_vel":[{}]"#, vels.join(",")));
        let tens: Vec<String> = snap.wd_tension.iter().map(|t| format!("{:.4}", t)).collect();
        parts.push(format!(r#""wd_tension":[{}]"#, tens.join(",")));
    }

    // lr_scale
    if !snap.lr_scale.is_empty() {
        let vals: Vec<String> = snap.lr_scale.iter().map(|s| format!("{:.4}", s)).collect();
        parts.push(format!(r#""lr_scale":[{}]"#, vals.join(",")));
    }

    if parts.is_empty() {
        return String::new();
    }

    format!(r#""dyn_params":{{{}}}"#, parts.join(","))
}
