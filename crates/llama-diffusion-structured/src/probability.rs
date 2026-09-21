//! Probability normalization over a restricted set of candidate logits.

use crate::{Error, Result};

/// Partial entropy of full-vocabulary probabilities, without candidate renormalization.
pub(crate) fn partial_entropy(row: &[f32], candidates: &[i32]) -> Result<f64> {
    if row.is_empty()
        || row.iter().any(|x| !x.is_finite())
        || candidates
            .iter()
            .any(|&id| id < 0 || id as usize >= row.len())
    {
        return Err(Error::InvalidLogits);
    }
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let log_shifted_z = row
        .iter()
        .map(|&x| (f64::from(x) - max).exp())
        .sum::<f64>()
        .ln();
    let mut indices: Vec<_> = (0..row.len()).collect();
    let count = 20.min(indices.len());
    if count < indices.len() {
        indices.select_nth_unstable_by(count, |&a, &b| row[b].total_cmp(&row[a]));
    }
    indices.truncate(count);
    indices.extend(candidates.iter().map(|&id| id as usize));
    indices.sort_unstable();
    indices.dedup();
    Ok(indices
        .into_iter()
        .map(|i| {
            let log_p = (f64::from(row[i]) - max) - log_shifted_z;
            -log_p.exp() * log_p
        })
        .sum())
}

/// Softmax over the allowed candidates, with no full-vocabulary normalization.
pub fn restricted_softmax(logits: &[f64]) -> Result<Vec<f64>> {
    if logits.is_empty() || logits.iter().any(|x| !x.is_finite()) {
        return Err(Error::InvalidLogits);
    }
    let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<_> = logits.iter().map(|x| (x - max).exp()).collect();
    let sum: f64 = weights.iter().sum();
    Ok(weights.into_iter().map(|x| x / sum).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reread_entropy_uses_full_vocabulary_and_unique_requested_tokens() {
        let row = vec![0.0; 100];
        let expected = 20.0 / 100.0 * 100_f64.ln();
        assert!((partial_entropy(&row, &[]).unwrap() - expected).abs() < 1e-12);
        // A common logit offset must not erase the normalization term.
        let shifted = vec![f32::MAX; 100];
        assert!((partial_entropy(&shifted, &[]).unwrap() - expected).abs() < 1e-12);
        let peaked = [0.0, -100.0, -100.0];
        assert!(partial_entropy(&peaked, &[1, 2, 2]).unwrap() < 1e-10);
        assert!(partial_entropy(&[f32::NAN], &[]).is_err());
        assert!(partial_entropy(&[0.0], &[1]).is_err());
    }

    #[test]
    fn softmax_handles_extremes_and_rejects_nonfinite_values() {
        assert_eq!(restricted_softmax(&[10000.0, 10000.0]).unwrap(), [0.5, 0.5]);
        assert_eq!(
            restricted_softmax(&[10000.0, -10000.0]).unwrap(),
            [1.0, 0.0]
        );
        assert_eq!(restricted_softmax(&[42.0]).unwrap(), [1.0]);
        for values in [&[][..], &[f64::NAN], &[f64::INFINITY]] {
            assert!(restricted_softmax(values).is_err());
        }
    }
}
