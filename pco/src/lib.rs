#![doc = include_str!("../README.md")]
//! # API Notes
//!
//! * In some places, Pco methods accept a destination (either `W: Write` or `&mut [T: Number]`).
//! If Pco returns an error, it is possible both the destination and the struct
//! have been modified.
//! * Pco will always try to process all numbers, and it will fail if insufficient bytes are
//! available. For instance, during decompression Pco will try to fill the entire `&mut [T]`
//! passed in, returning an insufficient data error if the `&[u8]` passed in is not long enough.

#![allow(clippy::manual_non_exhaustive)] // sometimes we want to ban explicit construction within the crate too
#![deny(clippy::unused_unit)]
#![deny(dead_code)]

#[doc = include_str!("../README.md")]
#[cfg(doctest)]
struct ReadmeDoctest;

pub use chunk_config::{ChunkConfig, DeltaSpec, ModeSpec, PagingSpec};
pub use constants::{DEFAULT_COMPRESSION_LEVEL, DEFAULT_MAX_PAGE_N, FULL_BATCH_N};
pub use progress::Progress;
pub use scratch::DecompressorScratch;

pub mod data_types;
/// for inspecting certain types of Pco metadata
pub mod describers;
pub mod errors;
/// structs representing stored information about how compression was done
pub mod metadata;
/// for compressing/decompressing .pco files
pub mod standalone;
/// for compressing/decompressing as part of an outer, wrapping format
pub mod wrapped;

mod ans;
mod bin_optimization;
mod bit_reader;
mod bit_writer;
mod bits;
mod chunk_config;
mod chunk_latent_compressor;
mod chunk_latent_decompressor;
mod compression_intermediates;
mod compression_table;
mod constants;
mod delta;
mod dyn_slices;
mod histograms;
mod macros;
mod mode;
mod page_latent_decompressor;
mod progress;
mod read_write_uint;
mod sampling;
mod scratch;
mod sort_utils;

#[cfg(test)]
mod tests;

// To ensure our tests surface uninitialized memory issues, since some default
// allocators fill with 0, we make the allocator write uncommon values instead.
// It also keeps per-thread byte counts, so that tests can bound how much a
// decoder allocates (see `peak_bytes_allocated`).
#[cfg(test)]
mod poisoning_allocator {
  use std::alloc::{GlobalAlloc, Layout, System};
  use std::cell::Cell;

  struct Poisoning;

  thread_local! {
    static CURRENT: Cell<usize> = const { Cell::new(0) };
    static PEAK: Cell<usize> = const { Cell::new(0) };
  }

  fn track(f: impl FnOnce(usize) -> usize) {
    // try_with: the counters may already be gone while a thread exits
    let _ = CURRENT.try_with(|current| {
      let now = f(current.get());
      current.set(now);
      let _ = PEAK.try_with(|peak| peak.set(peak.get().max(now)));
    });
  }

  /// The most bytes this thread held allocated at once while running `f`,
  /// beyond what it held before.
  pub(crate) fn peak_bytes_allocated(f: impl FnOnce()) -> usize {
    let before = CURRENT.with(Cell::get);
    PEAK.with(|peak| peak.set(before));
    f();
    PEAK.with(Cell::get) - before
  }

  unsafe impl GlobalAlloc for Poisoning {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
      let ptr = System.alloc(layout);
      if !ptr.is_null() {
        ptr.write_bytes(0xab, layout.size());
        track(|current| current + layout.size());
      }
      ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
      track(|current| current.saturating_sub(layout.size()));
      System.dealloc(ptr, layout)
    }
  }

  #[global_allocator]
  static ALLOCATOR: Poisoning = Poisoning;
}

#[cfg(feature = "bench")]
mod bench_utils;
/// Exists so that the `micro` bench target can force the linker to pull in
/// this crate's `#[divan::bench]` items, which register themselves at startup.
#[cfg(feature = "bench")]
#[doc(hidden)]
pub fn _bench_link() {}
