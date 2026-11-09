//! Glue between the GPU backend's existing clocks and the stage-profile document.
//!
//! Counters here are not added to the CPU wall sum. Submit/wait is CPU time
//! blocked on work the step already waited for. Timestamp spans are optional
//! `TIMESTAMP_QUERY` deltas.

use fl_core::profile::clock;
use fl_core::GpuCounter;

/// Sums of backend clocks over the ticks where a GPU device was used.
#[derive(Clone, Copy, Debug, Default)]
pub struct GpuRollup {
    pub host_prep_ms: f64,
    pub submit_wait_ms: f64,
    pub timestamp_span_ms: f64,
    pub timestamp_samples: u64,
    pub readback_copy_ms: f64,
    pub dispatches: u64,
    pub bookkeeping_ms: f64,
    pub pyramid_span_ms: f64,
    pub pyramid_samples: u64,
    pub field_host_ms: f64,
    pub field_submit_wait_ms: f64,
    pub field_timestamp_span_ms: f64,
    pub field_timestamp_samples: u64,
    pub begin_submit_ms: f64,
    pub uptake_wall_ms: f64,
    pub uptake_pack_ms: f64,
    pub uptake_upload_encode_ms: f64,
    pub uptake_wait_readback_ms: f64,
    pub uptake_timestamp_span_ms: f64,
    pub uptake_timestamp_samples: u64,
    pub deposit_submit_ms: f64,
    pub deposit_timestamp_span_ms: f64,
    pub deposit_timestamp_samples: u64,
    pub finish_wall_ms: f64,
    pub finish_readback_copy_ms: f64,
    pub finish_timestamp_span_ms: f64,
    pub finish_timestamp_samples: u64,
    pub commit_submit_ms: f64,
    /// Ticks that contributed resident begin/uptake/deposit/finish/commit walls.
    pub resident_ticks: u64,
    /// Ticks that contributed a field-step wall.
    pub field_ticks: u64,
    /// Ticks that contributed an evaluate/sense wall.
    pub eval_ticks: u64,
}

impl GpuRollup {
    pub fn delta(self, earlier: Self) -> Self {
        Self {
            host_prep_ms: self.host_prep_ms - earlier.host_prep_ms,
            submit_wait_ms: self.submit_wait_ms - earlier.submit_wait_ms,
            timestamp_span_ms: self.timestamp_span_ms - earlier.timestamp_span_ms,
            timestamp_samples: self
                .timestamp_samples
                .saturating_sub(earlier.timestamp_samples),
            readback_copy_ms: self.readback_copy_ms - earlier.readback_copy_ms,
            dispatches: self.dispatches.saturating_sub(earlier.dispatches),
            bookkeeping_ms: self.bookkeeping_ms - earlier.bookkeeping_ms,
            pyramid_span_ms: self.pyramid_span_ms - earlier.pyramid_span_ms,
            pyramid_samples: self.pyramid_samples.saturating_sub(earlier.pyramid_samples),
            field_host_ms: self.field_host_ms - earlier.field_host_ms,
            field_submit_wait_ms: self.field_submit_wait_ms - earlier.field_submit_wait_ms,
            field_timestamp_span_ms: self.field_timestamp_span_ms - earlier.field_timestamp_span_ms,
            field_timestamp_samples: self
                .field_timestamp_samples
                .saturating_sub(earlier.field_timestamp_samples),
            begin_submit_ms: self.begin_submit_ms - earlier.begin_submit_ms,
            uptake_wall_ms: self.uptake_wall_ms - earlier.uptake_wall_ms,
            uptake_pack_ms: self.uptake_pack_ms - earlier.uptake_pack_ms,
            uptake_upload_encode_ms: self.uptake_upload_encode_ms - earlier.uptake_upload_encode_ms,
            uptake_wait_readback_ms: self.uptake_wait_readback_ms - earlier.uptake_wait_readback_ms,
            uptake_timestamp_span_ms: self.uptake_timestamp_span_ms
                - earlier.uptake_timestamp_span_ms,
            uptake_timestamp_samples: self
                .uptake_timestamp_samples
                .saturating_sub(earlier.uptake_timestamp_samples),
            deposit_submit_ms: self.deposit_submit_ms - earlier.deposit_submit_ms,
            deposit_timestamp_span_ms: self.deposit_timestamp_span_ms - earlier.deposit_timestamp_span_ms,
            deposit_timestamp_samples: self
                .deposit_timestamp_samples
                .saturating_sub(earlier.deposit_timestamp_samples),
            finish_wall_ms: self.finish_wall_ms - earlier.finish_wall_ms,
            finish_readback_copy_ms: self.finish_readback_copy_ms - earlier.finish_readback_copy_ms,
            finish_timestamp_span_ms: self.finish_timestamp_span_ms
                - earlier.finish_timestamp_span_ms,
            finish_timestamp_samples: self
                .finish_timestamp_samples
                .saturating_sub(earlier.finish_timestamp_samples),
            commit_submit_ms: self.commit_submit_ms - earlier.commit_submit_ms,
            resident_ticks: self.resident_ticks.saturating_sub(earlier.resident_ticks),
            field_ticks: self.field_ticks.saturating_sub(earlier.field_ticks),
            eval_ticks: self.eval_ticks.saturating_sub(earlier.eval_ticks),
        }
    }

    #[cfg(feature = "gpu")]
    pub fn add_backend(&mut self, gpu: &fl_gpu::GpuBackend) {
        let eval = gpu.stats.last;
        let field = gpu.stats.field_last;
        let res = gpu.stats.resident_last;
        if eval.total_ms > 0.0 || eval.dispatches > 0 {
            self.eval_ticks += 1;
            self.host_prep_ms += eval.host_prep_ms;
            self.submit_wait_ms += eval.gpu_roundtrip_ms;
            self.readback_copy_ms += eval.readback_copy_ms;
            self.dispatches += eval.dispatches as u64;
            self.bookkeeping_ms += eval.bookkeeping_ms;
            if let Some(ms) = eval.kernel_ms {
                self.timestamp_span_ms += ms;
                self.timestamp_samples += 1;
            }
            if let Some(ms) = eval.pyramid_ms {
                self.pyramid_span_ms += ms;
                self.pyramid_samples += 1;
            }
        }
        if field.host_ms > 0.0 || field.roundtrip_ms > 0.0 {
            self.field_ticks += 1;
            self.field_host_ms += field.host_ms;
            self.field_submit_wait_ms += field.roundtrip_ms;
            if let Some(ms) = field.kernel_ms {
                self.field_timestamp_span_ms += ms;
                self.field_timestamp_samples += 1;
            }
        }
        let resident =
            res.begin_ms + res.uptake_ms + res.deposit_ms + res.finish_ms + res.commit_ms;
        if resident > 0.0
            || res.uptake_timestamp_span_ms.is_some()
            || res.finish_timestamp_span_ms.is_some()
        {
            self.resident_ticks += 1;
            self.begin_submit_ms += res.begin_ms;
            self.uptake_wall_ms += res.uptake_ms;
            self.uptake_pack_ms += res.uptake_pack_ms;
            self.uptake_upload_encode_ms += res.uptake_upload_encode_ms;
            self.uptake_wait_readback_ms += res.uptake_wait_readback_ms;
            self.deposit_submit_ms += res.deposit_ms;
            self.finish_wall_ms += res.finish_ms;
            self.finish_readback_copy_ms += res.finish_readback_copy_ms;
            self.commit_submit_ms += res.commit_ms;
            if let Some(ms) = res.uptake_timestamp_span_ms {
                self.uptake_timestamp_span_ms += ms;
                self.uptake_timestamp_samples += 1;
            }
            if let Some(ms) = res.finish_timestamp_span_ms {
                self.finish_timestamp_span_ms += ms;
                self.finish_timestamp_samples += 1;
            }
            if let Some(ms) = res.deposit_timestamp_span_ms {
                self.deposit_timestamp_span_ms += ms;
                self.deposit_timestamp_samples += 1;
            }
        }
    }

    pub fn counters(&self) -> Vec<GpuCounter> {
        let mut out = Vec::new();
        let push = |out: &mut Vec<GpuCounter>, id, clock_name, total_ms, samples, note| {
            if samples == 0 && total_ms == 0.0 {
                return;
            }
            out.push(GpuCounter {
                id,
                clock: clock_name,
                total_ms,
                samples,
                note,
            });
        };
        push(
            &mut out,
            "sense_or_mlp_host_prep",
            clock::CPU_WALL,
            self.host_prep_ms,
            self.eval_ticks,
            "CPU packing and uploads before the sense/MLP submit. Not GPU kernel time.",
        );
        push(
            &mut out,
            "sense_or_mlp_submit_and_wait",
            clock::CPU_WALL_BLOCKED_ON_GPU,
            self.submit_wait_ms,
            self.eval_ticks,
            "CPU time from submit through the logit/sense readback the step already waits for. Includes GPU execution only as seen from the CPU. Not GPU kernel time.",
        );
        push(
            &mut out,
            "sense_or_mlp_timestamp_span",
            clock::GPU_TIMESTAMP_QUERY,
            self.timestamp_span_ms,
            self.timestamp_samples,
            "TIMESTAMP_QUERY span of the sense or MLP compute pass. Not CPU time. Do not add to cpu_wall.",
        );
        push(
            &mut out,
            "sense_or_mlp_readback_copy",
            clock::CPU_WALL,
            self.readback_copy_ms,
            self.eval_ticks,
            "CPU copy of the mapped sense/MLP output after the wait returned. Already included in sense_or_mlp_submit_and_wait. Not GPU kernel time.",
        );
        if self.dispatches > 0 {
            out.push(GpuCounter {
                id: "sense_or_mlp_dispatches",
                clock: clock::COUNT,
                total_ms: 0.0,
                samples: self.dispatches,
                note: "Sense or MLP compute dispatches per profiled run. One dispatch per topology page. samples is a count, not milliseconds.",
            });
        }
        push(
            &mut out,
            "sense_or_mlp_bookkeeping",
            clock::CPU_WALL,
            self.bookkeeping_ms,
            self.eval_ticks,
            "CPU scatter of results and slot eviction after the readback.",
        );
        push(
            &mut out,
            "pyramid_timestamp_span",
            clock::GPU_TIMESTAMP_QUERY,
            self.pyramid_span_ms,
            self.pyramid_samples,
            "TIMESTAMP_QUERY span of the sensor-pyramid passes. Not CPU time. Do not add to cpu_wall.",
        );
        push(
            &mut out,
            "field_host_record",
            clock::CPU_WALL,
            self.field_host_ms,
            self.field_ticks,
            "CPU upload and command recording for the field step. Not GPU kernel time.",
        );
        push(
            &mut out,
            "field_submit_and_wait",
            clock::CPU_WALL_BLOCKED_ON_GPU,
            self.field_submit_wait_ms,
            self.field_ticks,
            "CPU time from the field submit through the readback the step already waits for. Not GPU kernel time.",
        );
        push(
            &mut out,
            "field_timestamp_span",
            clock::GPU_TIMESTAMP_QUERY,
            self.field_timestamp_span_ms,
            self.field_timestamp_samples,
            "TIMESTAMP_QUERY span from the first field pass to the last. Not CPU time. Do not add to cpu_wall.",
        );
        push(
            &mut out,
            "resident_begin_submit",
            clock::CPU_WALL_SUBMIT_NO_WAIT,
            self.begin_submit_ms,
            self.resident_ticks,
            "CPU time through queue.submit of the resident begin copy. Does not wait. Not GPU kernel time.",
        );
        push(
            &mut out,
            "resident_uptake_submit_and_wait",
            clock::CPU_WALL_BLOCKED_ON_GPU,
            self.uptake_wall_ms,
            self.resident_ticks,
            "CPU time for resident uptake, including the readback wait that call already performs. Not GPU kernel time.",
        );
        push(
            &mut out,
            "resident_uptake_timestamp_span",
            clock::GPU_TIMESTAMP_QUERY,
            self.uptake_timestamp_span_ms,
            self.uptake_timestamp_samples,
            "TIMESTAMP_QUERY span across resident uptake passes, read in the wait uptake already performs. Not CPU time. Do not add to cpu_wall.",
        );
        push(
            &mut out,
            "resident_uptake_pack",
            clock::CPU_WALL,
            self.uptake_pack_ms,
            self.resident_ticks,
            "CPU build of the compact uptake headers (cell, bilinear weights, demand, affinity slot). Affinity floats stay resident and are expanded on the device. Already included in resident_uptake_submit_and_wait. Not GPU kernel time.",
        );
        push(
            &mut out,
            "resident_uptake_upload_encode",
            clock::CPU_WALL,
            self.uptake_upload_encode_ms,
            self.resident_ticks,
            "CPU write_buffer and command recording before the uptake submit. Already included in resident_uptake_submit_and_wait. Not GPU kernel time.",
        );
        push(
            &mut out,
            "resident_uptake_wait_readback",
            clock::CPU_WALL_BLOCKED_ON_GPU,
            self.uptake_wait_readback_ms,
            self.resident_ticks,
            "CPU time from the uptake submit through the mapped readback. Includes the uptake timestamp span. Already included in resident_uptake_submit_and_wait. Not GPU kernel time.",
        );
        push(
            &mut out,
            "resident_deposit_submit",
            clock::CPU_WALL_SUBMIT_NO_WAIT,
            self.deposit_submit_ms,
            self.resident_ticks,
            "CPU time through queue.submit of resident deposits. Does not wait. Not GPU kernel time.",
        );
        push(
            &mut out,
            "resident_deposit_timestamp_span",
            clock::GPU_TIMESTAMP_QUERY,
            self.deposit_timestamp_span_ms,
            self.deposit_timestamp_samples,
            "TIMESTAMP_QUERY span of the deposit passes. Deposit does not wait; this span is read in the finish poll that already waits. Not CPU time. Do not add it to the finish wait.",
        );
        push(
            &mut out,
            "resident_finish_submit_and_wait",
            clock::CPU_WALL_BLOCKED_ON_GPU,
            self.finish_wall_ms,
            self.resident_ticks,
            "CPU time for resident finish, including the readback wait that call already performs. Not GPU kernel time.",
        );
        push(
            &mut out,
            "resident_finish_timestamp_span",
            clock::GPU_TIMESTAMP_QUERY,
            self.finish_timestamp_span_ms,
            self.finish_timestamp_samples,
            "TIMESTAMP_QUERY span of the resident finish pass, read in the wait finish already performs. Not CPU time. Do not add to cpu_wall.",
        );
        push(
            &mut out,
            "resident_finish_readback_copy",
            clock::CPU_WALL,
            self.finish_readback_copy_ms,
            self.resident_ticks,
            "CPU copy of the mapped per-cell finish rows after the wait returned. Already included in resident_finish_submit_and_wait. Not GPU kernel time.",
        );
        push(
            &mut out,
            "resident_commit_submit",
            clock::CPU_WALL_SUBMIT_NO_WAIT,
            self.commit_submit_ms,
            self.resident_ticks,
            "CPU time through queue.submit of the resident commit copy. Does not wait. Not GPU kernel time.",
        );
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counters_label_submit_wait_separately_from_timestamp_spans() {
        let rollup = GpuRollup {
            submit_wait_ms: 12.0,
            eval_ticks: 4,
            timestamp_span_ms: 9.0,
            timestamp_samples: 4,
            begin_submit_ms: 1.0,
            resident_ticks: 4,
            ..GpuRollup::default()
        };
        let counters = rollup.counters();
        let wait = counters
            .iter()
            .find(|c| c.id == "sense_or_mlp_submit_and_wait")
            .unwrap();
        let span = counters
            .iter()
            .find(|c| c.id == "sense_or_mlp_timestamp_span")
            .unwrap();
        let begin = counters
            .iter()
            .find(|c| c.id == "resident_begin_submit")
            .unwrap();
        assert_eq!(wait.clock, clock::CPU_WALL_BLOCKED_ON_GPU);
        assert_eq!(span.clock, clock::GPU_TIMESTAMP_QUERY);
        assert_eq!(begin.clock, clock::CPU_WALL_SUBMIT_NO_WAIT);
        assert!(wait.note.contains("Not GPU kernel time"));
        assert!(span.note.contains("Do not add"));
        assert!(begin.note.contains("Does not wait"));
        let split = GpuRollup {
            uptake_pack_ms: 1.5,
            uptake_upload_encode_ms: 0.4,
            uptake_wait_readback_ms: 5.0,
            uptake_wall_ms: 6.9,
            deposit_timestamp_span_ms: 1.2,
            deposit_timestamp_samples: 4,
            dispatches: 8,
            resident_ticks: 4,
            eval_ticks: 4,
            ..GpuRollup::default()
        };
        let counters = split.counters();
        let pack = counters.iter().find(|c| c.id == "resident_uptake_pack").unwrap();
        let wait = counters.iter().find(|c| c.id == "resident_uptake_wait_readback").unwrap();
        let dep = counters.iter().find(|c| c.id == "resident_deposit_timestamp_span").unwrap();
        let dispatches = counters.iter().find(|c| c.id == "sense_or_mlp_dispatches").unwrap();
        assert!(pack.note.contains("Already included"));
        assert_eq!(wait.clock, clock::CPU_WALL_BLOCKED_ON_GPU);
        assert!(dep.note.contains("does not wait") || dep.note.contains("Does not wait") || dep.note.contains("Deposit does not wait"));
        assert_eq!(dispatches.clock, clock::COUNT);
        assert_eq!(dispatches.samples, 8);
        assert!(
            counters.iter().all(|c| c.id != "field_submit_and_wait"),
            "zero samples stay out of the document"
        );
    }
}
