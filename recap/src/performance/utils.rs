/// Calculates data into sections of ten with percent
pub fn calculate_histogram(data: &[f64]) -> [f64; 10] {
    let histogram = data.iter().fold([0.0; 10], |mut acc, value| {
        let ms = (*value * 1000.0) as usize;
        let bucket = (ms / 10).min(9);
        acc[bucket] += 1.0;
        acc
    });

    // Convert counts to percentages
    let total: f64 = histogram.iter().sum();
    if total > 0.0 {
        histogram.map(|count| (count / total) * 100.0)
    } else {
        [0.0; 10]
    }
}
