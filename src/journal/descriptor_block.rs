// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use crate::block_index::FsBlockIndex;
use crate::checksum::Checksum;
use crate::error::{CorruptKind, Ext4Error, IncompatibleKind};
use crate::journal::superblock::JournalSuperblock;
use crate::util::{read_u32be, u64_from_hilo};
use bitflags::bitflags;

/// Ensure a descriptor block's checksum is valid.
///
/// The checksum is stored in the last four bytes of the block.
pub(super) fn validate_descriptor_block_checksum(
    superblock: &JournalSuperblock,
    block: &[u8],
) -> Result<(), Ext4Error> {
    // OK to unwrap: minimum block length is 1024.
    let checksum_offset = block.len().checked_sub(4).unwrap();
    let expected_checksum = read_u32be(block, checksum_offset);
    let mut checksum = Checksum::new();
    checksum.update(superblock.uuid.as_bytes());
    checksum.update(&block[..checksum_offset]);
    checksum.update_u32_be(0);

    if checksum.finalize() == expected_checksum {
        Ok(())
    } else {
        Err(CorruptKind::JournalDescriptorBlockChecksum.into())
    }
}

/// Data block tag within a descriptor block.
///
/// Each descriptor block contains an array of tags, one for each data
/// block following the descriptor block. Each data block will replace a
/// block within the ext4 filesystem. The tag indicates where the data
/// block maps into the filesystem, and provides a checksum for the data
/// block.
#[derive(Debug, Eq, PartialEq)]
pub(super) struct DescriptorBlockTag {
    /// Absolute block index in the filesystem that should be replaced
    /// with the data block associated with this tag.
    pub(super) block_index: FsBlockIndex,

    /// Checksum of the block data, if the journal is checksummed
    /// (`None` for a legacy journal with no checksums).
    ///
    /// Note that this checksum is for the data block associated with
    /// this tag. The data in the tag itself is covered by the
    /// descriptor block checksum.
    pub(super) checksum: Option<u32>,

    flags: DescriptorBlockTagFlags,

    /// On-disk size of this tag: the base tag plus a 16-byte UUID unless
    /// the `SAME_UUID` (UUID_OMITTED) flag is set.
    encoded_size: usize,
}

impl DescriptorBlockTag {
    /// Read a tag from `bytes`, selecting the on-disk layout from the
    /// journal's feature flags:
    ///
    /// * `is_v3` (`CHECKSUM_V3`): `journal_block_tag3_t`, 16 bytes —
    ///   `t_blocknr` / `t_flags` / `t_blocknr_high` / `t_checksum`, each
    ///   a big-endian u32.
    /// * legacy: `journal_block_tag_t` — `t_blocknr` be32, `t_checksum`
    ///   be16 (unused: v2 checksums are not supported), `t_flags` be16;
    ///   `t_blocknr_high` be32 follows only when 64-bit. Base size is 8,
    ///   or 12 when 64-bit.
    ///
    /// Either layout is followed by a 16-byte UUID unless the
    /// `SAME_UUID` flag is set.
    ///
    /// Returns `None` if there are not enough bytes to read the tag.
    fn read_bytes(bytes: &[u8], is_v3: bool, is_64bit: bool) -> Option<Self> {
        let base = if is_v3 {
            16
        } else if is_64bit {
            12
        } else {
            8
        };
        if bytes.len() < base {
            return None;
        }

        let (blocknr_lo, blocknr_hi, flags_raw, checksum) = if is_v3 {
            (
                read_u32be(bytes, 0),
                read_u32be(bytes, 8),
                read_u32be(bytes, 4),
                Some(read_u32be(bytes, 12)),
            )
        } else {
            let hi = if is_64bit { read_u32be(bytes, 8) } else { 0 };
            let flags = u16::from_be_bytes([bytes[6], bytes[7]]);
            (read_u32be(bytes, 0), hi, u32::from(flags), None)
        };

        let flags = DescriptorBlockTagFlags::from_bits_retain(flags_raw);
        let block_index = if is_64bit {
            u64_from_hilo(blocknr_hi, blocknr_lo)
        } else {
            u64::from(blocknr_lo)
        };

        let has_uuid =
            !flags.contains(DescriptorBlockTagFlags::UUID_OMITTED);
        let encoded_size = base + if has_uuid { 16 } else { 0 };
        if has_uuid && bytes.len() < encoded_size {
            return None;
        }

        Some(Self {
            block_index,
            checksum,
            flags,
            encoded_size,
        })
    }
}

/// Iterator over tags in a descriptor block.
pub(super) struct DescriptorBlockTagIter<'a> {
    /// Remaining bytes in the block.
    bytes: &'a [u8],

    /// Set to true after the last element (or an error) is
    /// returned. All future calls to `next` will return `None`.
    is_done: bool,

    /// Journal uses the `CHECKSUM_V3` (`journal_block_tag3_t`) layout.
    is_v3: bool,

    /// Journal uses 64-bit block numbers.
    is_64bit: bool,
}

impl<'a> DescriptorBlockTagIter<'a> {
    /// Create a tag iterator from the raw bytes of a descriptor block
    /// (the bytes after the journal block header).
    pub(super) fn new(bytes: &'a [u8], is_v3: bool, is_64bit: bool) -> Self {
        Self {
            bytes,
            is_done: false,
            is_v3,
            is_64bit,
        }
    }
}

impl Iterator for DescriptorBlockTagIter<'_> {
    type Item = Result<DescriptorBlockTag, Ext4Error>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.is_done {
            return None;
        }

        let tag = if let Some(tag) = DescriptorBlockTag::read_bytes(
            self.bytes,
            self.is_v3,
            self.is_64bit,
        ) {
            tag
        } else {
            // If there were not enough bytes left to read the next tag,
            // then either there is no tag with the `LAST_TAG` flag set,
            // or the final tag does have that flag set but there are
            // not enough bytes to read the full tag.
            self.is_done = true;
            return Some(Err(
                CorruptKind::JournalDescriptorBlockTruncated.into()
            ));
        };

        // Escaped data blocks are not yet supported.
        if tag.flags.contains(DescriptorBlockTagFlags::ESCAPED) {
            self.is_done = true;
            return Some(Err(IncompatibleKind::JournalBlockEscaped.into()));
        }

        if tag.flags.contains(DescriptorBlockTagFlags::LAST_TAG) {
            // Last tag reached, nothing more to read.
            self.is_done = true;
            return Some(Ok(tag));
        }

        // Update the remaining bytes.
        self.bytes = &self.bytes[tag.encoded_size..];

        Some(Ok(tag))
    }
}

bitflags! {
    #[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
    struct DescriptorBlockTagFlags: u32 {
        const ESCAPED = 0x1;
        const UUID_OMITTED = 0x2;
        const DELETED = 0x4;
        const LAST_TAG = 0x8;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Uuid;

    /// Test success and failure cases of `validate_descriptor_block_checksum`.
    #[test]
    fn test_validate_descriptor_block_checksum() {
        let superblock = JournalSuperblock {
            block_size: 1024,
            sequence: 0,
            start_block: 0,
            uuid: Uuid([0; 16]),
        };
        let mut block = vec![0; 1024];
        assert_eq!(
            validate_descriptor_block_checksum(&superblock, &block)
                .unwrap_err(),
            CorruptKind::JournalDescriptorBlockChecksum
        );

        block[1020..].copy_from_slice(&[0x74, 0xef, 0x0e, 0xf6]);
        assert!(
            validate_descriptor_block_checksum(&superblock, &block).is_ok()
        );
    }

    fn push_u32be(bytes: &mut Vec<u8>, value: u32) {
        bytes.extend(&value.to_be_bytes());
    }

    /// Test `DescriptorBlockTagIter` on valid input. The first tag has
    /// no UUID, the second tag does have a UUID.
    #[test]
    fn test_descriptor_block_tag_iter() {
        let mut bytes = vec![];

        // Block number low.
        push_u32be(&mut bytes, 0x1000);
        // Flags.
        push_u32be(&mut bytes, DescriptorBlockTagFlags::UUID_OMITTED.bits());
        // Block number high.
        push_u32be(&mut bytes, 0xa000);
        // Checksum.
        push_u32be(&mut bytes, 0x123);

        // Block number low.
        push_u32be(&mut bytes, 0x2000);
        // Flags.
        push_u32be(&mut bytes, DescriptorBlockTagFlags::LAST_TAG.bits());
        // Block number high.
        push_u32be(&mut bytes, 0xb000);
        // Checksum.
        push_u32be(&mut bytes, 0x456);
        // UUID.
        bytes.extend([0; 16]);

        assert_eq!(
            DescriptorBlockTagIter::new(&bytes)
                .map(Result::unwrap)
                .collect::<Vec<_>>(),
            [
                DescriptorBlockTag {
                    block_index: 0xa000_0000_1000,
                    flags: DescriptorBlockTagFlags::UUID_OMITTED,
                    checksum: 0x123,
                },
                DescriptorBlockTag {
                    block_index: 0xb000_0000_2000,
                    flags: DescriptorBlockTagFlags::LAST_TAG,
                    checksum: 0x456,
                }
            ]
        );
    }

    /// Test `DescriptorBlockTagFlags` on empty input.
    #[test]
    fn test_descriptor_block_tag_iter_empty() {
        let bytes = vec![];
        assert_eq!(
            DescriptorBlockTagIter::new(&bytes)
                .next()
                .unwrap()
                .unwrap_err(),
            CorruptKind::JournalDescriptorBlockTruncated
        );
    }

    /// Test that `DescriptorBlockTagIter` correctly returns an error on
    /// truncated input.
    #[test]
    fn test_descriptor_block_tag_iter_missing_uuid() {
        let mut bytes = vec![];

        // Block number low.
        push_u32be(&mut bytes, 0x2000);
        // Flags.
        push_u32be(&mut bytes, DescriptorBlockTagFlags::LAST_TAG.bits());
        // Block number high.
        push_u32be(&mut bytes, 0xb000);
        // Checksum.
        push_u32be(&mut bytes, 0x456);

        // Intentionally leave out the UUID bytes to produce an error.

        assert_eq!(
            DescriptorBlockTagIter::new(&bytes)
                .next()
                .unwrap()
                .unwrap_err(),
            CorruptKind::JournalDescriptorBlockTruncated
        );
    }

    /// Test that `DescriptorBlockTagIter` correctly returns an error if
    /// an escaped block is present.
    #[test]
    fn test_descriptor_block_tag_iter_escaped_error() {
        let mut bytes = vec![];

        // Block number low.
        push_u32be(&mut bytes, 0x2000);
        // Flags.
        push_u32be(
            &mut bytes,
            (DescriptorBlockTagFlags::ESCAPED
                | DescriptorBlockTagFlags::UUID_OMITTED
                | DescriptorBlockTagFlags::LAST_TAG)
                .bits(),
        );
        // Block number high.
        push_u32be(&mut bytes, 0xb000);
        // Checksum.
        push_u32be(&mut bytes, 0x456);

        assert_eq!(
            DescriptorBlockTagIter::new(&bytes)
                .next()
                .unwrap()
                .unwrap_err(),
            IncompatibleKind::JournalBlockEscaped
        );
    }
}
