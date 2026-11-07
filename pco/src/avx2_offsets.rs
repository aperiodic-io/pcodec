//! Explicit AVX2 gather kernels for reading offsets.
//!
//! LLVM only auto-vectorizes `read_offsets` with gathers when tuning for
//! specific CPUs (e.g. `-C target-cpu=native`); with generic tuning, even when
//! AVX2 is enabled, it judges gathers too expensive and emits scalar code.
//! These kernels use gathers explicitly, which made reading offsets of up to
//! 25 bits ~1.7x faster in microbenchmarks.
//!
//! Each kernel reads exactly the same bytes as the scalar
//! `bit_reader::read_uint_at::<_, READ_BYTES>`, so it has the same padding
//! requirements.

use std::arch::x86_64::*;

use crate::constants::Bitlen;
use crate::data_types::Latent;

/// Reads offsets for latents `0..k` using AVX2 gathers and adds them to
/// `latents`, returning `k`. The caller must handle latents `k..n` with the
/// scalar implementation. Returns 0 for unsupported (latent, read bytes)
/// combinations; reads of more than 25 bits (READ_BYTES > 4) would need 64-bit
/// gathers, which only have 4 lanes and benchmarked no faster than scalar.
///
/// # Safety
/// The CPU must support AVX2, `n` must be at most the length of each slice,
/// and `src` must be readable wherever the scalar implementation would read.
#[inline]
pub unsafe fn read_offsets<L: Latent, const READ_BYTES: usize>(
  src: &[u8],
  base_bit_idx: usize,
  offset_bits_csum: &[u32],
  offset_bits: &[u32],
  latents: &mut [L],
  n: usize,
) -> usize {
  assert!(n <= offset_bits_csum.len() && n <= offset_bits.len() && n <= latents.len());
  let src = src.as_ptr();
  let base_bit_idx = base_bit_idx as Bitlen;
  let csum = offset_bits_csum.as_ptr();
  let bits = offset_bits.as_ptr();
  let dst = latents.as_mut_ptr();
  // Latent is sealed to u8/u16/u32/u64, so BITS identifies the concrete type.
  match (L::BITS, READ_BYTES) {
    (8, 4) => read_u8_4(
      src,
      base_bit_idx,
      csum,
      bits,
      dst as *mut u8,
      n,
    ),
    (16, 4) => read_u16_4(
      src,
      base_bit_idx,
      csum,
      bits,
      dst as *mut u16,
      n,
    ),
    (32, 4) => read_u32_4(
      src,
      base_bit_idx,
      csum,
      bits,
      dst as *mut u32,
      n,
    ),
    _ => 0,
  }
}

/// Reads 8 offsets of up to 25 bits each, starting at latent `i`, as u32s.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn gather_8x32(
  src: *const u8,
  base_bit_idx: __m256i,
  csum: *const u32,
  bits: *const u32,
  i: usize,
) -> __m256i {
  let bit_idx = _mm256_add_epi32(
    base_bit_idx,
    _mm256_loadu_si256(csum.add(i) as *const __m256i),
  );
  let byte_idx = _mm256_srli_epi32::<3>(bit_idx);
  let words = _mm256_i32gather_epi32::<1>(src as *const i32, byte_idx);
  let bits_past_byte = _mm256_and_si256(bit_idx, _mm256_set1_epi32(7));
  let shifted = _mm256_srlv_epi32(words, bits_past_byte);
  let n_bits = _mm256_loadu_si256(bits.add(i) as *const __m256i);
  let one = _mm256_set1_epi32(1);
  let mask = _mm256_sub_epi32(_mm256_sllv_epi32(one, n_bits), one);
  _mm256_and_si256(shifted, mask)
}

#[target_feature(enable = "avx2")]
unsafe fn read_u8_4(
  src: *const u8,
  base_bit_idx: Bitlen,
  csum: *const u32,
  bits: *const u32,
  dst: *mut u8,
  n: usize,
) -> usize {
  let base = _mm256_set1_epi32(base_bit_idx as i32);
  // packus interleaves 128-bit lanes; this restores the original order
  let unshuffle = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
  let k = n / 32 * 32;
  for i in (0..k).step_by(32) {
    let v0 = gather_8x32(src, base, csum, bits, i);
    let v1 = gather_8x32(src, base, csum, bits, i + 8);
    let v2 = gather_8x32(src, base, csum, bits, i + 16);
    let v3 = gather_8x32(src, base, csum, bits, i + 24);
    // Offsets for u8 latents fit in 8 bits, so saturating packs are exact.
    let packed = _mm256_packus_epi16(
      _mm256_packus_epi32(v0, v1),
      _mm256_packus_epi32(v2, v3),
    );
    let offsets = _mm256_permutevar8x32_epi32(packed, unshuffle);
    let ptr = dst.add(i) as *mut __m256i;
    _mm256_storeu_si256(
      ptr,
      _mm256_add_epi8(_mm256_loadu_si256(ptr), offsets),
    );
  }
  k
}

#[target_feature(enable = "avx2")]
unsafe fn read_u16_4(
  src: *const u8,
  base_bit_idx: Bitlen,
  csum: *const u32,
  bits: *const u32,
  dst: *mut u16,
  n: usize,
) -> usize {
  let base = _mm256_set1_epi32(base_bit_idx as i32);
  let k = n / 16 * 16;
  for i in (0..k).step_by(16) {
    let v0 = gather_8x32(src, base, csum, bits, i);
    let v1 = gather_8x32(src, base, csum, bits, i + 8);
    // Offsets for u16 latents fit in 16 bits, so saturating packs are exact.
    let packed = _mm256_packus_epi32(v0, v1);
    let offsets = _mm256_permute4x64_epi64::<0b11_01_10_00>(packed);
    let ptr = dst.add(i) as *mut __m256i;
    _mm256_storeu_si256(
      ptr,
      _mm256_add_epi16(_mm256_loadu_si256(ptr), offsets),
    );
  }
  k
}

#[target_feature(enable = "avx2")]
unsafe fn read_u32_4(
  src: *const u8,
  base_bit_idx: Bitlen,
  csum: *const u32,
  bits: *const u32,
  dst: *mut u32,
  n: usize,
) -> usize {
  let base = _mm256_set1_epi32(base_bit_idx as i32);
  let k = n / 8 * 8;
  for i in (0..k).step_by(8) {
    let offsets = gather_8x32(src, base, csum, bits, i);
    let ptr = dst.add(i) as *mut __m256i;
    _mm256_storeu_si256(
      ptr,
      _mm256_add_epi32(_mm256_loadu_si256(ptr), offsets),
    );
  }
  k
}

#[cfg(test)]
mod tests {
  use rand_xoshiro::rand_core::{RngCore, SeedableRng};
  use rand_xoshiro::Xoroshiro128PlusPlus;

  use super::*;
  use crate::bit_reader;
  use crate::constants::{FULL_BATCH_N, OVERSHOOT_PADDING};

  fn check<L: Latent, const READ_BYTES: usize>(max_bits: Bitlen, expect_simd: bool) {
    let mut rng = Xoroshiro128PlusPlus::seed_from_u64(0);
    for trial in 0..200 {
      let n = if trial % 4 == 0 {
        FULL_BATCH_N
      } else {
        1 + (rng.next_u32() as usize % FULL_BATCH_N)
      };
      let base_bit_idx = (rng.next_u32() % 64) as usize;
      let offset_bits = (0..FULL_BATCH_N)
        .map(|_| match trial % 3 {
          0 => max_bits,
          1 => rng.next_u32() % (max_bits + 1),
          _ => (max_bits / 2).max(1),
        })
        .collect::<Vec<_>>();
      let mut offset_bits_csum = vec![0; FULL_BATCH_N];
      for i in 1..FULL_BATCH_N {
        offset_bits_csum[i] = offset_bits_csum[i - 1] + offset_bits[i - 1];
      }
      let total_bits = base_bit_idx + (offset_bits_csum[n - 1] + offset_bits[n - 1]) as usize;
      let src = (0..total_bits.div_ceil(8) + OVERSHOOT_PADDING)
        .map(|_| rng.next_u32() as u8)
        .collect::<Vec<_>>();
      let initial = (0..FULL_BATCH_N)
        .map(|_| L::from_u64(rng.next_u64()))
        .collect::<Vec<_>>();

      let mut expected = initial.clone();
      for i in 0..n {
        let bit_idx = base_bit_idx as Bitlen + offset_bits_csum[i];
        let offset = unsafe {
          bit_reader::read_uint_at::<L, READ_BYTES>(
            &src,
            (bit_idx / 8) as usize,
            bit_idx % 8,
            offset_bits[i],
          )
        };
        expected[i] = expected[i].wrapping_add(offset);
      }

      let mut actual = initial.clone();
      let k = unsafe {
        read_offsets::<L, READ_BYTES>(
          &src,
          base_bit_idx,
          &offset_bits_csum,
          &offset_bits,
          &mut actual,
          n,
        )
      };
      assert!(k <= n);
      if expect_simd && n >= 32 {
        assert!(k > 0);
      }
      assert_eq!(
        &actual[..k],
        &expected[..k],
        "trial {}",
        trial
      );
      assert_eq!(
        &actual[k..],
        &initial[k..],
        "trial {}",
        trial
      );
    }
  }

  #[test]
  fn test_matches_scalar() {
    if !is_x86_feature_detected!("avx2") {
      return;
    }
    check::<u8, 4>(8, true);
    check::<u16, 4>(16, true);
    check::<u32, 4>(25, true);
    check::<u32, 8>(32, false);
    check::<u64, 8>(57, false);
    check::<u64, 15>(64, false);
  }
}
