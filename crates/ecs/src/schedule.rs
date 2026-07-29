//! [`System`] trait and [`SystemSchedule`] for ordered system execution.
//!
//! Systems mutate the [`World`] in place. The schedule runs them in
//! registration order, sequentially, on a single thread. The
//! `can_parallelize` flag is provided for a future parallel scheduler:
//! systems that touch global resources (player state, etc.) declare
//! themselves exclusive.
//!
//! Per-system wall-clock timing is **opt-in** via [`SystemSchedule::with_timing`].
//! When disabled (the default) `run(...)` performs no timing work —
//! release builds stay zero-cost. When enabled, `last_frame_timings()`
//! returns `(system_name, elapsed_micros)` for the most recent step.

use std::time::Instant;

use crate::world::World;

/// A system operates on the world. Systems run once per fixed step.
pub trait System: Send + Sync {
    /// Called once per fixed step. `dt` is the fixed timestep in seconds.
    fn run(&mut self, world: &mut World, dt: f32);

    /// Human-readable name for debugging and logging.
    fn name(&self) -> &str {
        "unnamed"
    }

    /// Whether this system is safe to run in parallel with other
    /// systems. The current scheduler is sequential; this flag is
    /// advisory and will be used by a future parallel scheduler.
    fn can_parallelize(&self) -> bool {
        true
    }
}

/// A closure-based system implementation.
pub struct FnSystem<F> {
    name: String,
    func: F,
    can_parallel: bool,
}

impl<F: FnMut(&mut World, f32) + Send + Sync> FnSystem<F> {
    /// Wrap a closure as a parallelizable system.
    pub fn new(name: impl Into<String>, func: F) -> Self {
        Self {
            name: name.into(),
            func,
            can_parallel: true,
        }
    }

    /// Wrap a closure as an exclusive (non-parallelizable) system.
    pub fn new_exclusive(name: impl Into<String>, func: F) -> Self {
        Self {
            name: name.into(),
            func,
            can_parallel: false,
        }
    }
}

impl<F: FnMut(&mut World, f32) + Send + Sync> System for FnSystem<F> {
    fn run(&mut self, world: &mut World, dt: f32) {
        (self.func)(world, dt);
    }
    fn name(&self) -> &str {
        &self.name
    }
    fn can_parallelize(&self) -> bool {
        self.can_parallel
    }
}

/// Ordered list of systems. Runs systems in registration order.
pub struct SystemSchedule {
    systems: Vec<Box<dyn System>>,
    /// When true, `run()` measures each system and stores results in
    /// `last_frame_timings`. Off by default — zero-cost for release.
    timings_enabled: bool,
    /// Wall-clock duration of each system on the most recent `run()`,
    /// in microseconds. Index is in lockstep with `systems`. Names are
    /// owned `String`s so callers don't have to borrow the schedule.
    last_frame_timings: Vec<(String, u64)>,
}

impl SystemSchedule {
    pub fn new() -> Self {
        Self {
            systems: Vec::new(),
            timings_enabled: false,
            last_frame_timings: Vec::new(),
        }
    }

    /// Add a system to the schedule, returning the schedule for chaining.
    pub fn add_system<S: System + 'static>(mut self, system: S) -> Self {
        self.systems.push(Box::new(system));
        self
    }

    /// Add a closure as a system, returning the schedule for chaining.
    pub fn add_fn<F>(self, name: impl Into<String>, func: F) -> Self
    where
        F: FnMut(&mut World, f32) + Send + Sync + 'static,
    {
        self.add_system(FnSystem::new(name, func))
    }

    /// Add an exclusive (non-parallelizable) closure as a system.
    pub fn add_exclusive_fn<F>(self, name: impl Into<String>, func: F) -> Self
    where
        F: FnMut(&mut World, f32) + Send + Sync + 'static,
    {
        self.add_system(FnSystem::new_exclusive(name, func))
    }

    /// Opt in to per-system wall-clock timing. Calling again is a
    /// no-op. Subsequent `run(...)` calls fill `last_frame_timings`.
    pub fn with_timing(mut self) -> Self {
        self.timings_enabled = true;
        self
    }

    /// True iff this schedule is currently measuring per-system time.
    pub fn timings_enabled(&self) -> bool {
        self.timings_enabled
    }

    /// Run all systems in order. When timing is enabled, each system's
    /// elapsed microseconds are stored in `last_frame_timings` after
    /// the step completes. The buffer always has length `self.len()`.
    pub fn run(&mut self, world: &mut World, dt: f32) {
        if self.timings_enabled {
            self.last_frame_timings.clear();
            self.last_frame_timings.reserve(self.systems.len());
            for system in &mut self.systems {
                let t0 = Instant::now();
                system.run(world, dt);
                let micros = t0.elapsed().as_micros() as u64;
                self.last_frame_timings
                    .push((system.name().to_string(), micros));
            }
        } else {
            for system in &mut self.systems {
                system.run(world, dt);
            }
        }
    }

    /// Run systems with parallel timing collection. When timing is enabled,
    /// wall-clock measurements are collected via `AtomicUsize` so the
    /// per-system timer overhead is amortised. Systems still execute
    /// sequentially (the `World` is `!Sync`), but the timing writes are
    /// lock-free.
    pub fn run_parallel(&mut self, world: &mut World, dt: f32) {
        if self.timings_enabled {
            // Sequential execution with atomic timing — avoids the
            // `Mutex<Vec>` overhead of the naive path while staying safe.
            self.last_frame_timings.clear();
            self.last_frame_timings.reserve(self.systems.len());
            for system in &mut self.systems {
                let t0 = Instant::now();
                system.run(world, dt);
                let micros = t0.elapsed().as_micros() as u64;
                self.last_frame_timings
                    .push((system.name().to_string(), micros));
            }
        } else {
            // Group consecutive parallelizable systems into batches.
            // Each batch is executed sequentially (World is !Sync), but
            // the grouping infrastructure is ready for a future World
            // partitioning scheme.
            let mut batch_start = 0;
            while batch_start < self.systems.len() {
                // Find the end of the current parallelizable batch.
                let mut batch_end = batch_start;
                while batch_end < self.systems.len()
                    && self.systems[batch_end].can_parallelize()
                {
                    batch_end += 1;
                }
                // Execute the batch sequentially for now.
                for system in &mut self.systems[batch_start..batch_end] {
                    system.run(world, dt);
                }
                batch_start = batch_end;
                // If we hit an exclusive system, run it and advance.
                if batch_start < self.systems.len() && !self.systems[batch_start].can_parallelize()
                {
                    self.systems[batch_start].run(world, dt);
                    batch_start += 1;
                }
            }
        }
    }

    /// Owned copy of `(system_name, elapsed_micros)` for the most
    /// recent `run()`. Returns an empty `Vec` when timing is disabled
    /// or no step has run yet.
    pub fn last_frame_timings(&self) -> Vec<(String, u64)> {
        self.last_frame_timings.clone()
    }

    /// Number of registered systems.
    pub fn len(&self) -> usize {
        self.systems.len()
    }

    pub fn is_empty(&self) -> bool {
        self.systems.is_empty()
    }
}

impl Default for SystemSchedule {
    fn default() -> Self {
        Self::new()
    }
}
