//! ROI extraction (spec §5, §8 `roi.rs`): DBSCAN clustering of detection
//! centers accumulated during LEARNING → outlier rejection → convex hull →
//! RDP smoothing → geometric validation → safety margin.
//!
//! DBSCAN is implemented here (spec allows "linfa-clustering o
//! implementazione custom"); it is deterministic and unit-tested.

use serde::{Deserialize, Serialize};

/// A 2D point in image pixel coordinates.
pub type Pt = (f64, f64);

/// DBSCAN clustering. Returns `Some(cluster_id)` per point; `None` = noise.
///
/// Classic algorithm: core points have ≥ `min_samples` neighbors (including
/// self) within `eps`; border points join a cluster without expanding it.
pub fn dbscan(points: &[Pt], eps: f64, min_samples: u32) -> Vec<Option<u32>> {
    let n = points.len();
    let mut labels = vec![None; n];
    if n == 0 || eps <= 0.0 {
        return labels;
    }
    let eps2 = eps * eps;

    // Precomputed squared-distance neighborhoods.
    let neighbors: Vec<Vec<usize>> = (0..n)
        .map(|i| {
            (0..n)
                .filter(|j| dist2(points[i], points[*j]) <= eps2)
                .collect()
        })
        .collect();

    let mut cluster = 0u32;
    for i in 0..n {
        if labels[i].is_some() || neighbors[i].len() < min_samples as usize {
            continue; // already labeled or noise (may become border later)
        }
        // New cluster: BFS from i.
        labels[i] = Some(cluster);
        let mut queue = std::collections::VecDeque::from(neighbors[i].clone());
        while let Some(j) = queue.pop_front() {
            if labels[j].is_none() {
                labels[j] = Some(cluster);
                if neighbors[j].len() >= min_samples as usize {
                    // Core point: expand.
                    for &k in &neighbors[j] {
                        if labels[k].is_none() {
                            queue.push_back(k);
                        }
                    }
                }
            }
        }
        cluster += 1;
    }
    labels
}

fn dist2(a: Pt, b: Pt) -> f64 {
    let dx = a.0 - b.0;
    let dy = a.1 - b.1;
    dx * dx + dy * dy
}

/// Result of extracting the ROI polygon for one camera.
#[derive(Debug, Clone)]
pub enum RoiOutcome {
    /// A validated polygon was produced.
    Polygon(RoiPolygon),
    /// Validation failed → camera stays in LEARNING (spec §5.6).
    Invalid(String),
    /// Not enough usable points (no clusters at all).
    InsufficientData,
}

/// A validated, persisted ROI polygon (image-space pixel coordinates).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoiPolygon {
    /// Ring vertices, clockwise or counter-clockwise, closed implicitly.
    pub polygon: Vec<[f64; 2]>,
    pub image_width: u32,
    pub image_height: u32,
    /// Fraction of image area covered (informational).
    pub area_ratio: f64,
}

impl RoiPolygon {
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_default()
    }
    pub fn from_json(s: &str) -> Option<Self> {
        serde_json::from_str(s).ok()
    }

    /// Point-in-polygon (ray casting) used by the ACTIVE gate.
    pub fn contains(&self, x: f64, y: f64) -> bool {
        let pts = &self.polygon;
        let n = pts.len();
        if n < 3 {
            return false;
        }
        let mut inside = false;
        let mut j = n - 1;
        for i in 0..n {
            let [xi, yi] = pts[i];
            let [xj, yj] = pts[j];
            let intersects = ((yi > y) != (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi);
            if intersects {
                inside = !inside;
            }
            j = i;
        }
        inside
    }
}

/// Full extraction pipeline (spec §5, ordered steps 2–7).
#[allow(clippy::too_many_arguments)] // all parameters are spec §5.2 tunables
pub fn extract_roi(
    points: &[Pt],
    image_width: u32,
    image_height: u32,
    eps: f32,
    min_samples: u32,
    rdp_epsilon: f32,
    area_min: f64,
    area_max: f64,
    margin_pct: f64,
) -> RoiOutcome {
    let labels = dbscan(points, eps as f64, min_samples);

    // 3. Outlier rejection: keep only clustered points.
    let clustered: Vec<Pt> = points
        .iter()
        .zip(labels.iter())
        .filter_map(|(p, l)| l.map(|_| *p))
        .collect();
    if clustered.len() < 3 {
        return RoiOutcome::InsufficientData;
    }

    // 4. Convex hull (Andrew's monotone chain via `geo::ConvexHull`).
    let hull = convex_hull(&clustered);

    // 5. RDP smoothing.
    let simplified = rdp(&hull, rdp_epsilon as f64);
    if simplified.len() < 3 {
        return RoiOutcome::InsufficientData;
    }

    // 7. Safety margin: expand polygon by margin_pct about the centroid.
    let expanded = scale_about_centroid(&simplified, 1.0 + margin_pct);

    // 6. Geometric validation on the final polygon.
    let image_area = image_width as f64 * image_height as f64;
    if image_area <= 0.0 {
        return RoiOutcome::Invalid("invalid image dimensions".into());
    }
    let area = polygon_area(&expanded);
    let ratio = area / image_area;
    if !(area_min..=area_max).contains(&ratio) {
        return RoiOutcome::Invalid(format!(
            "ROI area ratio {ratio:.3} outside [{area_min:.2}, {area_max:.2}]"
        ));
    }

    RoiOutcome::Polygon(RoiPolygon {
        polygon: expanded.into_iter().map(|p| [p.0, p.1]).collect(),
        image_width,
        image_height,
        area_ratio: ratio,
    })
}

/// Andrew's monotone chain convex hull.
pub fn convex_hull(points: &[Pt]) -> Vec<Pt> {
    let mut pts = points.to_vec();
    pts.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap()
            .then(a.1.partial_cmp(&b.1).unwrap())
    });
    pts.dedup();
    if pts.len() < 3 {
        return pts;
    }
    let mut lower: Vec<Pt> = Vec::new();
    for p in &pts {
        while lower.len() >= 2 && cross(lower[lower.len() - 2], lower[lower.len() - 1], *p) <= 0.0 {
            lower.pop();
        }
        lower.push(*p);
    }
    let mut upper: Vec<Pt> = Vec::new();
    for p in pts.iter().rev() {
        while upper.len() >= 2 && cross(upper[upper.len() - 2], upper[upper.len() - 1], *p) <= 0.0 {
            upper.pop();
        }
        upper.push(*p);
    }
    upper.pop();
    lower.pop();
    lower.extend(upper);
    lower
}

fn cross(o: Pt, a: Pt, b: Pt) -> f64 {
    (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
}

/// Ramer–Douglas–Peucker simplification of a closed ring.
pub fn rdp(ring: &[Pt], epsilon: f64) -> Vec<Pt> {
    if ring.len() <= 3 || epsilon <= 0.0 {
        return ring.to_vec();
    }
    // Split the ring at the two farthest-apart vertices to preserve closure,
    // simplify each open segment, then concatenate.
    let (i0, i1) = farthest_pair(ring);
    let seg_a = simplify_open(&ring[i0..=i1], epsilon);
    let mut tail: Vec<Pt> = ring[i1..].to_vec();
    tail.extend_from_slice(&ring[..=i0]);
    let seg_b = simplify_open(&tail, epsilon);
    let mut out = seg_a;
    out.extend_from_slice(&seg_b[1..seg_b.len().saturating_sub(1)]);
    out
}

fn farthest_pair(ring: &[Pt]) -> (usize, usize) {
    let mut best = (0usize, ring.len() / 2);
    let mut best_d = -1.0;
    for i in 0..ring.len() {
        for j in i + 1..ring.len() {
            let d = dist2(ring[i], ring[j]);
            if d > best_d {
                best_d = d;
                best = (i, j);
            }
        }
    }
    best
}

fn simplify_open(pts: &[Pt], epsilon: f64) -> Vec<Pt> {
    if pts.len() <= 2 {
        return pts.to_vec();
    }
    let (a, b) = (pts[0], pts[pts.len() - 1]);
    let ab = (b.0 - a.0, b.1 - a.1);
    let len2 = ab.0 * ab.0 + ab.1 * ab.1;
    let mut max_idx = 0usize;
    let mut max_d = -1.0;
    for (i, p) in pts.iter().enumerate().skip(1).take(pts.len() - 2) {
        let d = if len2 == 0.0 {
            dist2(*p, a)
        } else {
            let ap = (p.0 - a.0, p.1 - a.1);
            let t = (ap.0 * ab.0 + ap.1 * ab.1) / len2;
            let proj = (a.0 + t * ab.0, a.1 + t * ab.1);
            dist2(*p, proj)
        };
        if d > max_d {
            max_d = d;
            max_idx = i;
        }
    }
    if max_d > epsilon * epsilon {
        let mut left = simplify_open(&pts[..=max_idx], epsilon);
        let right = simplify_open(&pts[max_idx..], epsilon);
        left.extend_from_slice(&right[1..]);
        left
    } else {
        vec![a, b]
    }
}

fn polygon_area(ring: &[Pt]) -> f64 {
    let n = ring.len();
    if n < 3 {
        return 0.0;
    }
    let mut area = 0.0;
    for i in 0..n {
        let j = (i + 1) % n;
        area += ring[i].0 * ring[j].1 - ring[j].0 * ring[i].1;
    }
    (area / 2.0).abs()
}

fn centroid(ring: &[Pt]) -> Pt {
    let n = ring.len();
    if n == 0 {
        return (0.0, 0.0);
    }
    let sum: Pt = ring
        .iter()
        .fold((0.0, 0.0), |acc, p| (acc.0 + p.0, acc.1 + p.1));
    (sum.0 / n as f64, sum.1 / n as f64)
}

fn scale_about_centroid(ring: &[Pt], factor: f64) -> Vec<Pt> {
    let c = centroid(ring);
    ring.iter()
        .map(|p| (c.0 + (p.0 - c.0) * factor, c.1 + (p.1 - c.1) * factor))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dbscan_finds_two_clusters_and_noise() {
        let mut pts = vec![(100.0, 100.0), (110.0, 105.0), (105.0, 95.0)];
        pts.extend(vec![(500.0, 500.0), (510.0, 505.0), (495.0, 490.0)]);
        pts.push((900.0, 900.0)); // noise
        let labels = dbscan(&pts, 50.0, 3);
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[1], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_ne!(labels[0], labels[3]);
        assert_eq!(labels[6], None);
    }

    #[test]
    fn dbscan_border_point_joins_cluster() {
        let pts = vec![(0.0, 0.0), (10.0, 0.0), (20.0, 0.0), (30.0, 0.0)];
        let labels = dbscan(&pts, 10.0, 3);
        // Chain: all reachable from core points (0,0),(10,0),(20,0)…
        assert!(labels.iter().all(|l| l.is_some()));
    }

    #[test]
    fn hull_is_convex_and_contains_points() {
        let pts = vec![
            (0.0, 0.0),
            (10.0, 0.0),
            (5.0, 5.0),
            (0.0, 10.0),
            (10.0, 10.0),
        ];
        let hull = convex_hull(&pts);
        assert_eq!(hull.len(), 4);
        // Centroid of hull must lie inside it.
        let c = centroid(&hull);
        let area = polygon_area(&hull);
        assert!(area > 0.0);
        let roi = RoiPolygon {
            polygon: hull.iter().map(|p| [p.0, p.1]).collect(),
            image_width: 20,
            image_height: 20,
            area_ratio: area / 400.0,
        };
        assert!(roi.contains(c.0, c.1));
    }

    #[test]
    fn rdp_reduces_collinear_ring() {
        let mut ring = vec![
            (0.0, 0.0),
            (5.0, 0.0),
            (10.0, 0.0),
            (15.0, 0.0),
            (10.0, 10.0),
            (0.0, 10.0),
        ];
        ring.push((0.0, 0.0)); // explicit close; ring handles it anyway
        let simplified = rdp(&ring, 2.0);
        assert!(simplified.len() < ring.len());
        assert!(simplified.len() >= 3);
    }

    #[test]
    fn extract_roi_validates_area_bounds() {
        // Cluster in a small area of a 1920×1080 frame → ratio < 10% → invalid.
        let mut pts: Vec<Pt> = (0..20)
            .map(|i| (100.0 + (i % 5) as f64 * 10.0, 100.0 + (i / 5) as f64 * 10.0))
            .collect();
        pts.extend((0..15).map(|i| (110.0 + (i % 4) as f64 * 8.0, 105.0 + i as f64 * 2.0)));
        match extract_roi(&pts, 1920, 1080, 50.0, 15, 5.0, 0.10, 0.90, 0.05) {
            RoiOutcome::Invalid(_) => {} // small cluster → below 10%
            other => panic!("expected Invalid, got {other:?}"),
        }

        // Wide cluster covering ~15% of the frame → valid polygon.
        // (grid pitch 80/90 px with eps 130 ⇒ every point has ~8 neighbours,
        // so min_samples=8 keeps the whole cluster reachable)
        let pts: Vec<Pt> = (0..60)
            .map(|i| {
                (
                    200.0 + (i % 12) as f64 * 80.0,
                    200.0 + (i / 12) as f64 * 90.0,
                )
            })
            .collect();
        match extract_roi(&pts, 1920, 1080, 130.0, 8, 5.0, 0.10, 0.90, 0.05) {
            RoiOutcome::Polygon(roi) => {
                assert!(roi.area_ratio > 0.10 && roi.area_ratio < 0.90);
                assert!(roi.polygon.len() >= 3);
                let json = roi.to_json();
                let back = RoiPolygon::from_json(&json).unwrap();
                assert_eq!(back.polygon.len(), roi.polygon.len());
            }
            other => panic!("expected Polygon, got {other:?}"),
        }
    }

    #[test]
    fn margin_expands_polygon() {
        let ring = vec![(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)];
        let expanded = scale_about_centroid(&ring, 1.05);
        let a0 = polygon_area(&ring);
        let a1 = polygon_area(&expanded);
        assert!((a1 / a0 - 1.05_f64.powi(2)).abs() < 1e-6);
    }

    #[test]
    fn point_in_polygon() {
        let roi = RoiPolygon {
            polygon: vec![[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]],
            image_width: 200,
            image_height: 200,
            area_ratio: 0.25,
        };
        assert!(roi.contains(50.0, 50.0));
        assert!(!roi.contains(150.0, 50.0));
        assert!(roi.contains(0.0, 50.0)); // edge counts as inside (ray casting)
    }
}
