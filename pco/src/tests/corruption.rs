//! Tests that a corrupt or hostile file is rejected rather than crashing.
//!
//! `stability.rs` covers *truncation*; this covers *mutation* and headers whose
//! declared sizes are not backed by any data. Everything the decoder reads is
//! attacker-controlled, so the contract is: return a `PcoError`, or decode to
//! something -- but never panic, and never allocate on a number the input has
//! not earned.

use crate::chunk_config::{ChunkConfig, DeltaSpec};
use crate::data_types::Number;
use crate::errors::PcoResult;
use crate::standalone::{
  simple_compress, simple_decompress, DecompressorItem, FileCompressor, FileDecompressor,
};
use crate::ModeSpec;

/// The mutations tried at each byte offset: every single-bit flip, plus the
/// two saturating values. Cheaper than all 256 values and hits the same
/// narrow bit fields, since the interesting header fields are 1-5 bits wide.
fn corrupt_values(orig: u8) -> impl Iterator<Item = u8> {
  (0..8)
    .map(move |bit| orig ^ (1 << bit))
    .chain([0x00, 0xff])
    .filter(move |&v| v != orig)
}

#[test]
fn test_single_byte_corruption_never_panics() -> PcoResult<()> {
  let nums: Vec<i64> = (0..400).map(|i| (i * 3) % 97).collect();

  for config in [
    ChunkConfig::default().with_mode_spec(ModeSpec::Classic),
    ChunkConfig::default().with_delta_spec(DeltaSpec::TryConsecutive(1)),
    ChunkConfig::default().with_delta_spec(DeltaSpec::TryLookback),
    ChunkConfig::default().with_mode_spec(ModeSpec::TryIntMult(7)),
  ] {
    let valid = simple_compress(&nums, &config)?;
    assert_eq!(
      simple_decompress::<i64>(&valid)?.len(),
      nums.len(),
      "{:?} did not round trip before corruption",
      config
    );

    for idx in 0..valid.len() {
      for value in corrupt_values(valid[idx]) {
        let mut corrupt = valid.clone();
        corrupt[idx] = value;
        // Corrupt data may legitimately decode to garbage, so reaching this
        // line at all is the assertion.
        let _ = simple_decompress::<i64>(&corrupt);
      }
    }
  }

  Ok(())
}

#[test]
fn test_single_byte_corruption_never_panics_floats() -> PcoResult<()> {
  let nums: Vec<f64> = (0..400).map(|i| (i as f64) * 0.25).collect();

  for mode_spec in [
    ModeSpec::Classic,
    ModeSpec::TryFloatQuant(20),
    ModeSpec::TryFloatMult(0.25),
  ] {
    let config = ChunkConfig::default().with_mode_spec(mode_spec);
    let valid = simple_compress(&nums, &config)?;

    for idx in 0..valid.len() {
      for value in corrupt_values(valid[idx]) {
        let mut corrupt = valid.clone();
        corrupt[idx] = value;
        let _ = simple_decompress::<f64>(&corrupt);
      }
    }
  }

  Ok(())
}

/// `n_hint` is a hint, not a promise: a tiny file may legally declare
/// `usize::MAX` (`standalone/guarantee.rs` writes exactly that), so it must not
/// reach `Vec::with_capacity` unclamped.
#[test]
fn test_absurd_n_hint_is_clamped() -> PcoResult<()> {
  let mut file = Vec::new();
  let fc = FileCompressor::default().with_n_hint(usize::MAX);
  fc.write_header(&mut file)?;
  let mut cc = fc.chunk_compressor(&[7_i64], &ChunkConfig::default())?;
  cc.write(&mut file)?;
  fc.write_footer(&mut file)?;

  let (fd, src) = FileDecompressor::new(file.as_slice())?;
  // 8 KiB is 1024 i64s.
  let nums = fd.with_max_prealloc(8192).simple_decompress::<i64>(src)?;

  assert_eq!(nums, vec![7_i64]);
  assert!(
    nums.capacity() <= 1024,
    "preallocated {} numbers",
    nums.capacity()
  );

  Ok(())
}

/// `FloatQuant`'s `k` is used as a shift width, so it must be rejected before
/// any latent splitting happens -- not after.
#[test]
fn test_float_quant_k_beyond_type_width_is_rejected() {
  fn check<T: Number>(k: u32, bits: u32) {
    let config = ChunkConfig::default().with_mode_spec(ModeSpec::TryFloatQuant(k));
    let nums = vec![T::default(); 8];
    assert!(
      simple_compress(&nums, &config).is_err(),
      "k={} should be rejected for a {}-bit type",
      k,
      bits
    );
  }

  for k in [64, 65, 100, 255, 256, u32::MAX] {
    check::<f64>(k, 64);
  }
  for k in [32, 33, 64, 255, 256, u32::MAX] {
    check::<f32>(k, 32);
  }
}

/// Decodes the first chunk of `file` into `dst`, as a reader that knows how
/// many numbers to expect does (so the output costs nothing, whatever count
/// the chunk claims).
fn decode_into(file: &[u8], dst: &mut [i64]) -> PcoResult<()> {
  let (fd, src) = FileDecompressor::new(file)?;
  if let DecompressorItem::Chunk(mut cd) = fd.chunk_decompressor::<i64, _>(src)? {
    while !cd.read(dst)?.finished {}
  }
  Ok(())
}

/// The most bytes decoding any single-byte corruption (see `corrupt_values`)
/// of the first 128 bytes of `file` into a buffer of `n` numbers allocates at
/// once.
fn worst_corruption_peak(file: &[u8], n: usize) -> usize {
  let mut dst = vec![0_i64; n];
  let mut worst = 0;
  for i in 0..file.len().min(128) {
    for value in corrupt_values(file[i]) {
      let mut corrupt = file.to_vec();
      corrupt[i] = value;
      let peak = crate::poisoning_allocator::peak_bytes_allocated(|| {
        let _ = decode_into(&corrupt, &mut dst);
      });
      worst = worst.max(peak);
    }
  }
  worst
}

/// What decoding the intact file allocates, plus room for scratch space; a
/// corrupt header must not cost more than this.
fn allocation_budget(file: &[u8], n: usize) -> usize {
  let mut dst = vec![0_i64; n];
  let intact = crate::poisoning_allocator::peak_bytes_allocated(|| {
    decode_into(file, &mut dst).unwrap();
  });
  4 * intact + (1 << 20)
}

/// A Dict mode dictionary length that the data cannot back (up to 2^25
/// entries) must not be allocated before the entries are read.
#[test]
fn test_dict_len_not_backed_by_data_costs_no_memory() -> PcoResult<()> {
  let nums: Vec<i64> = (0..5_000).map(|i| [5, 77, 1 << 40, -3][i % 4]).collect();
  let file = simple_compress(
    &nums,
    &ChunkConfig::default().with_mode_spec(ModeSpec::TryDict),
  )?;
  let (worst, budget) = (
    worst_corruption_peak(&file, nums.len()),
    allocation_budget(&file, nums.len()),
  );
  assert!(
    worst <= budget,
    "a corrupt {}-byte Dict file allocated {} bytes (budget {})",
    file.len(),
    worst,
    budget
  );
  Ok(())
}

/// A lookback window (up to 2^24 numbers) larger than the page can reach must
/// not be allocated: lookbacks before the page's state only ever read zeros.
#[test]
fn test_lookback_window_beyond_the_page_costs_no_memory() -> PcoResult<()> {
  let nums: Vec<i64> = (0..5_000).map(|i| (i % 97 * (i % 13)) as i64).collect();
  let file = simple_compress(
    &nums,
    &ChunkConfig::default().with_delta_spec(DeltaSpec::TryLookback),
  )?;
  let (worst, budget) = (
    worst_corruption_peak(&file, nums.len()),
    allocation_budget(&file, nums.len()),
  );
  assert!(
    worst <= budget,
    "a corrupt {}-byte lookback file allocated {} bytes (budget {})",
    file.len(),
    worst,
    budget
  );
  Ok(())
}
