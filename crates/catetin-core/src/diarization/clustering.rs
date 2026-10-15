use std::collections::HashMap;

use crate::events::SpeakerId;

/// Online Speaker Diarization Clustering Engine.
/// Uses Cosine Similarity against running speaker centroid embeddings
/// to assign speech segments to consistent speakers in real time.
#[derive(Debug, Clone)]
pub struct OnlineSpeakerClusterer {
    /// Similarity threshold above which an embedding is matched to an existing speaker.
    /// Recommended: 0.70 to 0.78 for CAM++ or x-vectors.
    similarity_threshold: f32,
    /// Centroid update rate (momentum). Higher value = slower centroid drift.
    momentum: f32,
    /// Map from remote speaker index to its running centroid embedding vector.
    centroids: HashMap<u32, Vec<f32>>,
    next_speaker_id: u32,
}

impl Default for OnlineSpeakerClusterer {
    fn default() -> Self {
        Self::new(0.72, 0.85)
    }
}

impl OnlineSpeakerClusterer {
    pub fn new(similarity_threshold: f32, momentum: f32) -> Self {
        Self {
            similarity_threshold,
            momentum,
            centroids: HashMap::new(),
            next_speaker_id: 1,
        }
    }

    /// Reset all tracked speaker clusters (e.g. on new meeting session)
    pub fn reset(&mut self) {
        self.centroids.clear();
        self.next_speaker_id = 1;
    }

    /// Number of identified unique remote speakers so far
    pub fn speaker_count(&self) -> usize {
        self.centroids.len()
    }

    /// Classify an embedding vector and return the matched or newly created SpeakerId.
    pub fn identify_or_register(&mut self, embedding: &[f32]) -> SpeakerId {
        if embedding.is_empty() {
            return SpeakerId::Unknown;
        }

        let normalized = normalize_vector(embedding);

        // Find existing cluster with highest cosine similarity
        let mut best_speaker_id = None;
        let mut best_similarity = -1.0f32;

        for (&spk_id, centroid) in &self.centroids {
            let sim = cosine_similarity(&normalized, centroid);
            if sim > best_similarity {
                best_similarity = sim;
                best_speaker_id = Some(spk_id);
            }
        }

        if let Some(id) = best_speaker_id {
            if best_similarity >= self.similarity_threshold {
                // Matched existing speaker: update running centroid with momentum
                self.update_centroid(id, &normalized);
                return SpeakerId::Remote(id);
            }
        }

        // Did not match any existing speaker above threshold: register new speaker!
        let new_id = self.next_speaker_id;
        self.next_speaker_id += 1;
        self.centroids.insert(new_id, normalized);
        SpeakerId::Remote(new_id)
    }

    fn update_centroid(&mut self, speaker_id: u32, new_embedding: &[f32]) {
        if let Some(centroid) = self.centroids.get_mut(&speaker_id) {
            for (c, &e) in centroid.iter_mut().zip(new_embedding.iter()) {
                *c = self.momentum * (*c) + (1.0 - self.momentum) * e;
            }
            *centroid = normalize_vector(centroid);
        }
    }
}

/// Compute cosine similarity between normalized unit vectors: A . B
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum::<f32>().clamp(-1.0, 1.0)
}

/// L2 normalize a vector to unit length
pub fn normalize_vector(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|&x| x * x).sum::<f32>().sqrt();
    if norm < 1e-8 {
        v.to_vec()
    } else {
        v.iter().map(|&x| x / norm).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity() {
        let v1 = vec![1.0, 0.0, 0.0];
        let v2 = vec![1.0, 0.0, 0.0];
        let v3 = vec![0.0, 1.0, 0.0];

        assert!((cosine_similarity(&v1, &v2) - 1.0).abs() < 1e-5);
        assert!((cosine_similarity(&v1, &v3) - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_speaker_clustering() {
        let mut clusterer = OnlineSpeakerClusterer::new(0.70, 0.80);

        // Speaker A voiceprint vector
        let voice_a = vec![0.8, 0.2, 0.1, 0.0];
        // Slightly perturbed Speaker A vector (natural vocal variation)
        let voice_a_variation = vec![0.82, 0.18, 0.11, 0.02];
        // Distinct Speaker B voiceprint vector
        let voice_b = vec![0.05, 0.1, 0.9, 0.4];

        let spk1 = clusterer.identify_or_register(&voice_a);
        assert_eq!(spk1, SpeakerId::Remote(1));

        let spk1_again = clusterer.identify_or_register(&voice_a_variation);
        assert_eq!(spk1_again, SpeakerId::Remote(1), "Should cluster with Speaker 1");

        let spk2 = clusterer.identify_or_register(&voice_b);
        assert_eq!(spk2, SpeakerId::Remote(2), "Should create new Speaker 2");
        assert_eq!(clusterer.speaker_count(), 2);
    }
}
