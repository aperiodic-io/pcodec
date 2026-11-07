//! SSE2 prefix sums for first-order consecutive delta decoding.
//!
//! The scalar decode loop has a serial dependency on the running moment, so
//! LLVM can't vectorize it. Instead we compute an in-register exclusive prefix
//! sum of each 128-bit vector (log2(lanes) shift-and-add steps) and carry the
//! running total between vectors. SSE2 is part of the x86_64 baseline, so no
//! runtime feature detection is needed.

use std::arch::x86_64::*;

use crate::data_types::Latent;

/// Decodes as many leading latents as fit in whole vectors, replacing each
/// delta with the running moment (exclusive prefix sum) and advancing the
/// moment. Returns how many latents were decoded; the caller must decode the
/// rest. Returns 0 for u64 latents, where 2-lane vectors benchmarked no
/// faster than the scalar loop.
#[inline]
pub fn decode_prefix<L: Latent>(moment: &mut L, latents: &mut [L]) -> usize {
  let n = latents.len();
  let moment = moment as *mut L;
  let dst = latents.as_mut_ptr() as *mut __m128i;
  // SAFETY: Latent is sealed to u8/u16/u32/u64, so BITS identifies the concrete
  // type, and each function only accesses whole vectors within `n` latents.
  unsafe {
    match L::BITS {
      8 => decode_u8(&mut *(moment as *mut u8), dst, n),
      16 => decode_u16(&mut *(moment as *mut u16), dst, n),
      32 => decode_u32(&mut *(moment as *mut u32), dst, n),
      _ => 0,
    }
  }
}

#[inline]
unsafe fn decode_u8(moment: &mut u8, dst: *mut __m128i, n: usize) -> usize {
  let n_vecs = n / 16;
  let mut carry = _mm_set1_epi8(*moment as i8);
  for j in 0..n_vecs {
    let v = _mm_loadu_si128(dst.add(j));
    let mut s = _mm_slli_si128::<1>(v);
    s = _mm_add_epi8(s, _mm_slli_si128::<1>(s));
    s = _mm_add_epi8(s, _mm_slli_si128::<2>(s));
    s = _mm_add_epi8(s, _mm_slli_si128::<4>(s));
    s = _mm_add_epi8(s, _mm_slli_si128::<8>(s));
    _mm_storeu_si128(dst.add(j), _mm_add_epi8(s, carry));
    // broadcast the vector's total (last byte of the inclusive sum)
    let total = _mm_add_epi8(s, v);
    let total = _mm_unpackhi_epi8(total, total);
    let total = _mm_shuffle_epi32::<0xff>(_mm_shufflehi_epi16::<0xff>(total));
    carry = _mm_add_epi8(carry, total);
  }
  *moment = _mm_cvtsi128_si32(carry) as u8;
  n_vecs * 16
}

#[inline]
unsafe fn decode_u16(moment: &mut u16, dst: *mut __m128i, n: usize) -> usize {
  let n_vecs = n / 8;
  let mut carry = _mm_set1_epi16(*moment as i16);
  for j in 0..n_vecs {
    let v = _mm_loadu_si128(dst.add(j));
    let mut s = _mm_slli_si128::<2>(v);
    s = _mm_add_epi16(s, _mm_slli_si128::<2>(s));
    s = _mm_add_epi16(s, _mm_slli_si128::<4>(s));
    s = _mm_add_epi16(s, _mm_slli_si128::<8>(s));
    _mm_storeu_si128(dst.add(j), _mm_add_epi16(s, carry));
    let total = _mm_add_epi16(s, v);
    let total = _mm_shuffle_epi32::<0xff>(_mm_shufflehi_epi16::<0xff>(total));
    carry = _mm_add_epi16(carry, total);
  }
  *moment = _mm_cvtsi128_si32(carry) as u16;
  n_vecs * 8
}

#[inline]
unsafe fn decode_u32(moment: &mut u32, dst: *mut __m128i, n: usize) -> usize {
  let n_vecs = n / 4;
  let mut carry = _mm_set1_epi32(*moment as i32);
  for j in 0..n_vecs {
    let v = _mm_loadu_si128(dst.add(j));
    let mut s = _mm_slli_si128::<4>(v);
    s = _mm_add_epi32(s, _mm_slli_si128::<4>(s));
    s = _mm_add_epi32(s, _mm_slli_si128::<8>(s));
    _mm_storeu_si128(dst.add(j), _mm_add_epi32(s, carry));
    carry = _mm_add_epi32(
      carry,
      _mm_shuffle_epi32::<0xff>(_mm_add_epi32(s, v)),
    );
  }
  *moment = _mm_cvtsi128_si32(carry) as u32;
  n_vecs * 4
}

#[cfg(test)]
mod tests {
  use rand_xoshiro::rand_core::{RngCore, SeedableRng};
  use rand_xoshiro::Xoroshiro128PlusPlus;

  use super::*;

  fn check<L: Latent>() {
    let mut rng = Xoroshiro128PlusPlus::seed_from_u64(0);
    for n in [0, 1, 3, 4, 7, 8, 15, 16, 17, 31, 33, 100, 255, 256] {
      let deltas = (0..n)
        .map(|_| L::from_u64(rng.next_u64()))
        .collect::<Vec<_>>();
      let initial_moment = L::from_u64(rng.next_u64());

      let mut expected = deltas.clone();
      let mut expected_moment = initial_moment;
      for x in expected.iter_mut() {
        let tmp = *x;
        *x = expected_moment;
        expected_moment = expected_moment.wrapping_add(tmp);
      }

      let mut actual = deltas.clone();
      let mut moment = initial_moment;
      let k = decode_prefix(&mut moment, &mut actual);
      assert!(k <= n);
      if L::BITS <= 32 {
        assert!(k + 16 > n, "only decoded {} of {}", k, n);
      }
      assert_eq!(&actual[..k], &expected[..k], "n={}", n);
      assert_eq!(&actual[k..], &deltas[k..], "n={}", n);
      let mut expected_moment_at_k = initial_moment;
      for &d in &deltas[..k] {
        expected_moment_at_k = expected_moment_at_k.wrapping_add(d);
      }
      assert_eq!(moment, expected_moment_at_k, "n={}", n);
      if k == n {
        assert_eq!(moment, expected_moment, "n={}", n);
      }
    }
  }

  #[test]
  fn test_matches_scalar() {
    check::<u8>();
    check::<u16>();
    check::<u32>();
    check::<u64>();
  }
}
