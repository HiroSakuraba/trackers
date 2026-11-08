//! FrankenNumPy integration (plan §3.1, §9.3, §16.6).
//!
//! * **Layout descriptors.** The authoritative field is described by
//!   FrankenNumPy `NdLayout`s — energy `[gy, gx]` and moments `[gy, gx, 32]`,
//!   both C-ordered F32 — and fl-core's buffers are *checked* against them
//!   (element counts and byte strides), so no consumer reinterprets a buffer
//!   by guessing its layout from its length.
//! * **Typed storage.** Field planes convert to `ArrayStorage::F32`, genuinely
//!   typed F32 storage rather than an F64 store with an F32 tag.
//! * **Interchange.** A committed epoch exports to `.npz` through `fnp-io`
//!   (field planes plus organism columns) for offline analysis.
//!
//! These adapters only read committed state.

#![forbid(unsafe_code)]

use fl_core::field::Field;
use fl_core::{World, CHANNELS};
use fnp_dtype::ArrayStorage;
use fnp_io::{write_npz_bytes, IOSupportedDType, NpyHeader};
use fnp_ndarray::{element_count, MemoryOrder, NdLayout};

/// Semantic layout of the authoritative field.
#[derive(Debug, Clone)]
pub struct FieldLayout {
    pub energy: NdLayout,
    pub moments: NdLayout,
}

pub fn field_layout(field: &Field) -> Result<FieldLayout, String> {
    let energy = NdLayout::contiguous(vec![field.gy, field.gx], 4, MemoryOrder::C).map_err(|e| format!("{e:?}"))?;
    let moments = NdLayout::contiguous(vec![field.gy, field.gx, CHANNELS], 4, MemoryOrder::C).map_err(|e| format!("{e:?}"))?;
    Ok(FieldLayout { energy, moments })
}

/// Check fl-core's field buffers against the FrankenNumPy descriptors.
pub fn check_field(field: &Field, layout: &FieldLayout) -> Result<(), String> {
    let ne = element_count(&layout.energy.shape).map_err(|e| format!("{e:?}"))?;
    let nm = element_count(&layout.moments.shape).map_err(|e| format!("{e:?}"))?;
    if field.e.len() != ne || field.m.len() != nm {
        return Err(format!("field buffers ({}, {}) do not match layouts ({ne}, {nm})", field.e.len(), field.m.len()));
    }
    // fl-core indexes energy as e[y*gx + x] and moments as m[(y*gx + x)*32 + k].
    let want_e = [field.gx as isize * 4, 4];
    let want_m = [(field.gx * CHANNELS) as isize * 4, CHANNELS as isize * 4, 4];
    if layout.energy.strides != want_e || layout.moments.strides != want_m {
        return Err(format!("stride mismatch: energy {:?} vs {want_e:?}, moments {:?} vs {want_m:?}", layout.energy.strides, layout.moments.strides));
    }
    if !layout.energy.is_contiguous() || !layout.moments.is_contiguous() {
        return Err("field layouts must be C-contiguous".into());
    }
    Ok(())
}

/// Field planes as genuinely typed F32 FrankenNumPy storage.
pub fn typed_planes(field: &Field) -> (ArrayStorage, ArrayStorage) {
    (ArrayStorage::F32(field.e.clone()), ArrayStorage::F32(field.m.clone()))
}

fn le_f32(v: impl IntoIterator<Item = f32>) -> Vec<u8> {
    v.into_iter().flat_map(f32::to_le_bytes).collect()
}

/// Export a committed epoch as `.npz` bytes (FrankenNumPy `fnp-io`).
///
/// Members: `energy [gy,gx] f32`, `moments [gy,gx,32] f32`,
/// `organism_xy [n,2] f64`, `organism_energy [n] f64`, `organism_id [n] u64`,
/// `organism_lineage [n] u64`, `organism_generation [n] u32`, `tick [] u64`.
pub fn export_npz(world: &World) -> Result<Vec<u8>, String> {
    let f = &world.field;
    let layout = field_layout(f)?;
    check_field(f, &layout)?;
    let o = &world.orgs;
    let n = o.len();
    let (te, tm) = typed_planes(f);
    let (ArrayStorage::F32(e), ArrayStorage::F32(m)) = (te, tm) else { unreachable!("typed_planes returns F32") };
    let h = |shape: Vec<usize>, d: IOSupportedDType| NpyHeader { shape, fortran_order: false, descr: d };
    let he = h(layout.energy.shape.clone(), IOSupportedDType::F32);
    let hm = h(layout.moments.shape.clone(), IOSupportedDType::F32);
    let hxy = h(vec![n, 2], IOSupportedDType::F64);
    let h1 = h(vec![n], IOSupportedDType::F64);
    let hid = h(vec![n], IOSupportedDType::U64);
    let hg = h(vec![n], IOSupportedDType::U32);
    let ht = h(vec![], IOSupportedDType::U64);
    let pe = le_f32(e);
    let pm = le_f32(m);
    let pxy: Vec<u8> = (0..n).flat_map(|i| [o.x[i], o.y[i]]).flat_map(f64::to_le_bytes).collect();
    let pen: Vec<u8> = o.energy.iter().flat_map(|v| v.to_le_bytes()).collect();
    let pid: Vec<u8> = o.id.iter().flat_map(|v| v.to_le_bytes()).collect();
    let plin: Vec<u8> = o.lineage.iter().flat_map(|v| v.to_le_bytes()).collect();
    let pgen: Vec<u8> = o.generation.iter().flat_map(|v| v.to_le_bytes()).collect();
    let pt = world.tick.to_le_bytes().to_vec();
    write_npz_bytes(&[
        ("energy", &he, &pe),
        ("moments", &hm, &pm),
        ("organism_xy", &hxy, &pxy),
        ("organism_energy", &h1, &pen),
        ("organism_id", &hid, &pid),
        ("organism_lineage", &hid, &plin),
        ("organism_generation", &hg, &pgen),
        ("tick", &ht, &pt),
    ])
    .map_err(|e| format!("npz: {e:?}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use fl_core::Config;

    #[test]
    fn field_layout_matches_fl_core_indexing_and_exports() {
        let mut c = Config::reference();
        c.population.initial_count = 64;
        let mut w = World::new(c).unwrap();
        w.step().unwrap();
        let l = field_layout(&w.field).unwrap();
        check_field(&w.field, &l).unwrap();
        let bytes = export_npz(&w).unwrap();
        assert_eq!(&bytes[..2], b"PK", "npz is a zip archive");
        let back = fnp_io::read_npz_bytes(&bytes, false).unwrap();
        assert_eq!(back.len(), 8);
    }
}
