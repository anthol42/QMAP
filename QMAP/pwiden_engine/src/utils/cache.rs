use sha2::{Sha256, Digest};
use std::fs;
use std::path::PathBuf;

/// Get the cache directory for pwiden_engine
pub fn get_cache_directory() -> Result<PathBuf, String> {
    let cache_dir = dirs::cache_dir()
        .ok_or("Failed to get system cache directory")?;
    let pwiden_cache = cache_dir.join("pwiden_engine");

    // Create directory if it doesn't exist
    if !pwiden_cache.exists() {
        fs::create_dir_all(&pwiden_cache)
            .map_err(|e| format!("Failed to create cache directory: {}", e))?;
    }

    Ok(pwiden_cache)
}

/// Compute SHA-256 hash of sequence vector and alignment parameters
pub fn hash_sequences_and_params(
    sequences: &[String],
    matrix: &str,
    gap_open: i32,
    gap_extension: i32,
) -> String {
    let mut hasher = Sha256::new();

    // Hash sequences
    for seq in sequences {
        hasher.update(seq.as_bytes());
        hasher.update(b"\0"); // Delimiter to avoid collisions
    }

    // Hash alignment parameters
    hasher.update(matrix.as_bytes());
    hasher.update(b"\0");
    hasher.update(gap_open.to_le_bytes());
    hasher.update(gap_extension.to_le_bytes());

    format!("{:x}", hasher.finalize())
}

/// Get cache file path for edgelist
pub fn get_edgelist_cache_path(
    sequences: &[String],
    threshold: f32,
    matrix: &str,
    gap_open: i32,
    gap_extension: i32,
) -> Result<PathBuf, String> {
    let cache_dir = get_cache_directory()?;
    let params_hash = hash_sequences_and_params(sequences, matrix, gap_open, gap_extension);
    let filename = format!("edgelist_{}_thresh_{:.4}.bin", params_hash, threshold);
    Ok(cache_dir.join(filename))
}

/// Compute SHA-256 hash of train/test sequences and alignment parameters
pub fn hash_train_test_and_params(
    train_seqs: &[String],
    test_seqs: &[String],
    matrix: &str,
    gap_open: i32,
    gap_extension: i32,
) -> String {
    let mut hasher = Sha256::new();

    // Hash train sequences with marker and length
    hasher.update(b"TRAIN:");
    hasher.update((train_seqs.len() as u32).to_le_bytes());
    for seq in train_seqs {
        hasher.update(seq.as_bytes());
        hasher.update(b"\0");
    }

    // Hash test sequences with marker and length
    hasher.update(b"TEST:");
    hasher.update((test_seqs.len() as u32).to_le_bytes());
    for seq in test_seqs {
        hasher.update(seq.as_bytes());
        hasher.update(b"\0");
    }

    // Hash alignment parameters
    hasher.update(matrix.as_bytes());
    hasher.update(b"\0");
    hasher.update(gap_open.to_le_bytes());
    hasher.update(gap_extension.to_le_bytes());

    format!("{:x}", hasher.finalize())
}

/// Get cache file path for binary mask
pub fn get_binary_mask_cache_path(
    train_seqs: &[String],
    test_seqs: &[String],
    threshold: f32,
    matrix: &str,
    gap_open: i32,
    gap_extension: i32,
) -> Result<PathBuf, String> {
    let cache_dir = get_cache_directory()?;
    let params_hash = hash_train_test_and_params(train_seqs, test_seqs, matrix, gap_open, gap_extension);
    let filename = format!("binary_mask_{}_thresh_{:.4}.npy", params_hash, threshold);
    Ok(cache_dir.join(filename))
}
