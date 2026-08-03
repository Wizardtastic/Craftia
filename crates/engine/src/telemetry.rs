//! Telemetry collector: central ring-buffer store of metric samples,
//! collected once per frame. Separate from ProfilerState (which focuses
//! on GPU/CPU timings) — this handles all metrics the dashboard displays.

use std::collections::VecDeque;
use std::time::Instant;

/// One data point at a point in time.
#[derive(Clone, Debug)]
pub struct TelemetrySnapshot {
    /// Wall-clock timestamp (relative seconds since first sample).
    #[allow(dead_code)] // Used by dashboard graph time axis
    pub time: f64,

    // -- Frame timing --
    pub cpu_frame_ms: f32,
    pub gpu_frame_ms: f32,

    // -- GPU pass breakdown (for stacked graph) --
    pub gpu_shadow_ms: f32,
    pub gpu_sky_ms: f32,
    pub gpu_opaque_ms: f32,
    pub gpu_transparent_ms: f32,
    pub gpu_ui_ms: f32,
    pub gpu_post_ms: f32,

    // -- Memory --
    pub gpu_allocated_mb: f32,
    pub gpu_reserved_mb: f32,
    pub process_rss_mb: f32,

    // -- Chunks --
    pub chunks_loaded: u32,
    pub chunks_meshed: u32,
    pub chunks_gpu: u32,
    pub chunk_vertices: u32,
    pub chunk_indices: u32,

    // -- Streamer --
    pub streamer_gen_queue: u32,
    pub streamer_mesh_queue: u32,
    pub streamer_pending_remesh: u32,

    // -- Water --
    pub water_pending_flow: u32,

    // -- ECS --
    pub entity_count: u32,
    pub archetype_count: u32,
}

impl Default for TelemetrySnapshot {
    fn default() -> Self {
        Self {
            time: 0.0,
            cpu_frame_ms: 0.0,
            gpu_frame_ms: 0.0,
            gpu_shadow_ms: 0.0,
            gpu_sky_ms: 0.0,
            gpu_opaque_ms: 0.0,
            gpu_transparent_ms: 0.0,
            gpu_ui_ms: 0.0,
            gpu_post_ms: 0.0,
            gpu_allocated_mb: 0.0,
            gpu_reserved_mb: 0.0,
            process_rss_mb: 0.0,
            chunks_loaded: 0,
            chunks_meshed: 0,
            chunks_gpu: 0,
            chunk_vertices: 0,
            chunk_indices: 0,
            streamer_gen_queue: 0,
            streamer_mesh_queue: 0,
            streamer_pending_remesh: 0,
            water_pending_flow: 0,
            entity_count: 0,
            archetype_count: 0,
        }
    }
}

/// Which metric groups are enabled.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MetricGroup {
    FrameTiming,
    Memory,
    Chunks,
    Streamer,
    Water,
    Ecs,
}

/// Selector for extracting a single metric series from the ring buffer.
#[allow(dead_code)] // Some variants used only by dashboard graph rendering
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MetricSelector {
    CpuFrameMs,
    GpuFrameMs,
    GpuShadowMs,
    GpuSkyMs,
    GpuOpaqueMs,
    GpuTransparentMs,
    GpuUiMs,
    GpuPostMs,
    GpuAllocatedMb,
    GpuReservedMb,
    ProcessRssMb,
    ChunksLoaded,
    ChunksMeshed,
    ChunksGpu,
    ChunkVertices,
    ChunkIndices,
    StreamerGenQueue,
    StreamerMeshQueue,
    StreamerPendingRemesh,
    WaterPendingFlow,
    EntityCount,
    ArchetypeCount,
}

/// Central ring-buffer store of metric samples.
pub struct TelemetryCollector {
    /// Ring buffer of samples, newest at the end.
    samples: VecDeque<TelemetrySnapshot>,
    /// Maximum number of samples to retain (default 3600 = 60s at 60 FPS).
    max_samples: usize,
    /// Time of first sample (for relative time).
    start_time: Instant,
    /// Whether telemetry collection is enabled.
    enabled: bool,
    /// Which metric groups are enabled.
    #[allow(dead_code)] // Future: per-group filtering
    enabled_groups: std::collections::HashSet<MetricGroup>,
}

impl TelemetryCollector {
    pub fn new(max_samples: usize) -> Self {
        Self {
            samples: VecDeque::with_capacity(max_samples),
            max_samples,
            start_time: Instant::now(),
            enabled: false,
            enabled_groups: [
                MetricGroup::FrameTiming,
                MetricGroup::Memory,
                MetricGroup::Chunks,
                MetricGroup::Streamer,
                MetricGroup::Water,
                MetricGroup::Ecs,
            ]
            .into_iter()
            .collect(),
        }
    }

    pub fn push(&mut self, snap: TelemetrySnapshot) {
        // Always collect data so the dashboard has history when toggled on.
        self.samples.push_back(snap);
        while self.samples.len() > self.max_samples {
            self.samples.pop_front();
        }
    }

    #[allow(dead_code)]
    pub fn clear(&mut self) {
        self.samples.clear();
    }

    #[allow(dead_code)]
    pub fn set_enabled(&mut self, group: MetricGroup, enabled: bool) {
        if enabled {
            self.enabled_groups.insert(group);
        } else {
            self.enabled_groups.remove(&group);
        }
    }

    pub fn toggle(&mut self) {
        self.enabled = !self.enabled;
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    #[allow(dead_code)]
    pub fn samples(&self) -> &VecDeque<TelemetrySnapshot> {
        &self.samples
    }

    pub fn last(&self) -> Option<&TelemetrySnapshot> {
        self.samples.back()
    }

    pub fn elapsed_secs(&self) -> f64 {
        self.start_time.elapsed().as_secs_f64()
    }

    /// Extract a single metric as a `Vec<f32>` over the last `count` samples.
    pub fn extract_f32(&self, selector: MetricSelector, count: usize) -> Vec<f32> {
        let len = self.samples.len().min(count);
        let start = self.samples.len().saturating_sub(len);
        self.samples
            .range(start..)
            .map(|s| match selector {
                MetricSelector::CpuFrameMs => s.cpu_frame_ms,
                MetricSelector::GpuFrameMs => s.gpu_frame_ms,
                MetricSelector::GpuShadowMs => s.gpu_shadow_ms,
                MetricSelector::GpuSkyMs => s.gpu_sky_ms,
                MetricSelector::GpuOpaqueMs => s.gpu_opaque_ms,
                MetricSelector::GpuTransparentMs => s.gpu_transparent_ms,
                MetricSelector::GpuUiMs => s.gpu_ui_ms,
                MetricSelector::GpuPostMs => s.gpu_post_ms,
                MetricSelector::GpuAllocatedMb => s.gpu_allocated_mb,
                MetricSelector::GpuReservedMb => s.gpu_reserved_mb,
                MetricSelector::ProcessRssMb => s.process_rss_mb,
                MetricSelector::ChunksLoaded => s.chunks_loaded as f32,
                MetricSelector::ChunksMeshed => s.chunks_meshed as f32,
                MetricSelector::ChunksGpu => s.chunks_gpu as f32,
                MetricSelector::ChunkVertices => s.chunk_vertices as f32,
                MetricSelector::ChunkIndices => s.chunk_indices as f32,
                MetricSelector::StreamerGenQueue => s.streamer_gen_queue as f32,
                MetricSelector::StreamerMeshQueue => s.streamer_mesh_queue as f32,
                MetricSelector::StreamerPendingRemesh => s.streamer_pending_remesh as f32,
                MetricSelector::WaterPendingFlow => s.water_pending_flow as f32,
                MetricSelector::EntityCount => s.entity_count as f32,
                MetricSelector::ArchetypeCount => s.archetype_count as f32,
            })
            .collect()
    }
}

impl Default for TelemetryCollector {
    fn default() -> Self {
        Self::new(3600)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_and_extract() {
        let mut collector = TelemetryCollector::new(100);
        let mut snap = TelemetrySnapshot {
            cpu_frame_ms: 16.0,
            ..Default::default()
        };
        collector.push(snap.clone());
        snap.cpu_frame_ms = 32.0;
        collector.push(snap);

        let values = collector.extract_f32(MetricSelector::CpuFrameMs, 10);
        assert_eq!(values.len(), 2);
        assert!((values[0] - 16.0).abs() < f32::EPSILON);
        assert!((values[1] - 32.0).abs() < f32::EPSILON);
    }

    #[test]
    fn ring_buffer_eviction() {
        let mut collector = TelemetryCollector::new(3);
        for i in 0..5 {
            let snap = TelemetrySnapshot {
                cpu_frame_ms: i as f32,
                ..Default::default()
            };
            collector.push(snap);
        }
        assert_eq!(collector.samples().len(), 3);
        assert!((collector.samples()[0].cpu_frame_ms - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn toggle() {
        let mut collector = TelemetryCollector::new(100);
        assert!(!collector.enabled());
        collector.toggle();
        assert!(collector.enabled());
        // Data always collects regardless of enabled state.
        collector.push(TelemetrySnapshot::default());
        assert_eq!(collector.samples().len(), 1);
    }

    #[test]
    fn extract_empty() {
        let collector = TelemetryCollector::new(100);
        let values = collector.extract_f32(MetricSelector::GpuFrameMs, 10);
        assert!(values.is_empty());
    }
}
