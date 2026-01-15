use parasail_rs::Aligner;
use ndarray::Array2;
use rayon::prelude::*;
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressStyle, ProgressDrawTarget, ProgressState};
use std::cmp::max;

/// Align two sequences and return identity score
pub fn align(aligner: &Aligner, seq1: &[u8], seq2: &[u8]) -> f32 {
    let alignment = aligner.align(Some(seq1), seq2).unwrap();
    let alignment_len = alignment.get_length().unwrap() as f32;
    let min_len = max(seq1.len(), seq2.len()) as f32;
    let total = if alignment_len > min_len {alignment_len} else {min_len};

    alignment.get_matches().unwrap() as f32 / total
}

/// Compute upper triangular pairwise identities and return as NxN matrix
/// Only computes upper triangle, sets diagonal to 1.0, leaves lower triangle as 0.0
/// The lower triangle will be filled by the caller if needed
pub fn compute_upper_triangle_matrix(
    sequences: &[String],
    aligner: &Aligner,
    show_progress: bool,
) -> Array2<f32> {
    let total_size = sequences.len();

    let pb = if show_progress {
        let bar = ProgressBar::new(total_size.pow(2) as u64);
        bar.set_draw_target(ProgressDrawTarget::stderr());
        bar.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] |{bar:40.cyan/blue}| {pos}M/{len}M ({per_sec}, {eta})")
                .unwrap()
                .with_key("pos", |state: &ProgressState, w: &mut dyn std::fmt::Write|
                    write!(w, "{}", state.pos() / 1_000_000).unwrap())
                .with_key("len", |state: &ProgressState, w: &mut dyn std::fmt::Write|
                    write!(w, "{}", state.len().unwrap_or(0) / 1_000_000).unwrap())
                .progress_chars("█▉▊▋▌▍▎▏ ")
        );
        bar
    } else {
        ProgressBar::hidden()
    };

    // Compute pairwise identities in parallel
    // Only compute upper triangle and diagonal
    let out_data: Vec<f32> = (0..total_size.pow(2))
        .into_par_iter()
        .progress_with(pb)
        .map(|idx| {
            let row = idx / total_size;
            let col = idx % total_size;

            if row == col {
                // Diagonal: self-identity is always 1.0
                1.0
            } else if row < col {
                // Upper triangle: compute alignment
                align(
                    aligner,
                    sequences[row].as_bytes(),
                    sequences[col].as_bytes(),
                )
            } else {
                // Lower triangle: placeholder
                0.0
            }
        })
        .collect();

    Array2::from_shape_vec((total_size, total_size), out_data).unwrap()
}
