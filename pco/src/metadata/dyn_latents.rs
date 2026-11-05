use better_io::BetterBufRead;

use crate::bit_reader::{BitReader, BitReaderBuilder};
use crate::bit_writer::BitWriter;
use crate::constants::{MAX_SUPPORTED_PRECISION_BYTES, OVERSHOOT_PADDING};
use crate::data_types::Latent;
use crate::errors::PcoResult;
use crate::macros::{define_latent_enum, match_latent_enum};
use std::io::Write;

define_latent_enum!(
  #[derive(Clone, Debug, PartialEq, Eq)]
  pub DynLatents(Vec)
);

impl DynLatents {
  pub(crate) fn len(&self) -> usize {
    match_latent_enum!(
      self,
      DynLatents<T>(inner) => { inner.len() }
    )
  }

  pub(crate) fn exact_bit_size(&self) -> usize {
    match_latent_enum!(
      self,
      DynLatents<T>(inner) => { inner.len() * T::BITS as usize}
    )
  }

  /// Replaces the latents with `n` latents read from `reader_builder`.
  ///
  /// `n` comes from the data, so the latents are allocated as they are read:
  /// a count the data cannot back fails once the data runs out, having
  /// allocated no more than the data it read.
  pub(crate) fn read_long_uncompressed_in_place<R: BetterBufRead>(
    &mut self,
    reader_builder: &mut BitReaderBuilder<R>,
    n: usize,
  ) -> PcoResult<()> {
    const READ_BATCH_N: usize = 512;
    const READ_SIZE: usize = READ_BATCH_N * MAX_SUPPORTED_PRECISION_BYTES + OVERSHOOT_PADDING;
    match_latent_enum!(
      self,
      DynLatents<L>(inner) => {
        inner.clear();
        for start in (0..n).step_by(READ_BATCH_N) {
          let batch_n = READ_BATCH_N.min(n - start);
          reader_builder.with_reader(READ_SIZE, |reader| unsafe {
            inner.extend((0..batch_n).map(|_| reader.read_uint::<L>(L::BITS)));
            Ok(())
          })?;
        }
      }
    );

    Ok(())
  }

  pub(crate) unsafe fn read_short_uncompressed_from<L: Latent>(
    reader: &mut BitReader,
    len: usize,
  ) -> PcoResult<Self> {
    let mut latents = vec![L::ZERO; len];
    for x in &mut latents {
      *x = reader.read_uint::<L>(L::BITS)
    }
    Ok(Self::new(latents))
  }

  pub(crate) unsafe fn write_uncompressed_to<W: Write>(&self, writer: &mut BitWriter<W>) {
    match_latent_enum!(
      &self,
      DynLatents<L>(inner) => {
        for &latent in inner {
          writer.write_uint(latent, L::BITS);
        }
      }
    );
  }
}
