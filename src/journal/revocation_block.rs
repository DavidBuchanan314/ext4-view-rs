// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use crate::block_index::FsBlockIndex;
use crate::checksum::Checksum;
use crate::error::{CorruptKind, Ext4Error};
use crate::journal::JournalSuperblock;
use crate::journal::block_header::JournalBlockHeader;
use crate::util::read_u32be;
use crate::util::usize_from_u32;
use alloc::vec::Vec;

/// Ensure a revocation block's checksum is valid.
///
/// The checksum is stored in the last four bytes of the block.
pub(super) fn validate_revocation_block_checksum(
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
        Err(CorruptKind::JournalRevocationBlockChecksum.into())
    }
}

/// Read the revoked block indices from a revocation block.
///
/// The entries are appended to the end of `table`.
pub(super) fn read_revocation_block_table(
    block: &[u8],
    table: &mut Vec<FsBlockIndex>,
    is_64bit: bool,
    has_checksums: bool,
) -> Result<(), Ext4Error> {
    // Revocation records are 4 bytes (32-bit journal) or 8 bytes (64-bit).
    let record_len: usize = if is_64bit { 8 } else { 4 };
    // The 4-byte block tail checksum is only present on checksummed journals.
    let csum_size: usize = if has_checksums { 4 } else { 0 };

    // On-disk layout (jbd2 `jbd2_journal_revoke_header_t`): the 12-byte
    // journal block header, then `r_count` (be32) = the total number of
    // bytes used in the block (header + count + records). Records run from
    // offset 16 up to `r_count`.
    const HEADER_LEN: usize = JournalBlockHeader::SIZE; // 12
    const RECORDS_START: usize = HEADER_LEN + size_of::<u32>(); // 16

    let r_count = usize_from_u32(read_u32be(block, HEADER_LEN));
    if r_count < RECORDS_START
        || r_count > block.len().saturating_sub(csum_size)
    {
        return Err(CorruptKind::JournalRevocationBlockInvalidTableSize(
            r_count,
        )
        .into());
    }

    // Read each record from [RECORDS_START, r_count) and append to `table`.
    let mut offset = RECORDS_START;
    while offset + record_len <= r_count {
        let block_index = if record_len == 8 {
            // OK to unwrap: slice length matches u64.
            u64::from_be_bytes(block[offset..offset + 8].try_into().unwrap())
        } else {
            u64::from(read_u32be(block, offset))
        };
        table.push(block_index);
        offset += record_len;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::journal::superblock::JournalIncompatibleFeatures;
    use crate::uuid::Uuid;

    /// Test success and failure cases of `validate_revocation_block_checksum`.
    #[test]
    fn test_validate_revocation_block_checksum() {
        let superblock = JournalSuperblock {
            block_size: 1024,
            sequence: 0,
            start_block: 0,
            uuid: Uuid([0; 16]),
            incompat: JournalIncompatibleFeatures::CHECKSUM_V3,
        };
        let mut block = vec![0; 1024];
        assert_eq!(
            validate_revocation_block_checksum(&superblock, &block)
                .unwrap_err(),
            CorruptKind::JournalRevocationBlockChecksum
        );

        block[1020..].copy_from_slice(&[0x74, 0xef, 0x0e, 0xf6]);
        assert!(
            validate_revocation_block_checksum(&superblock, &block).is_ok()
        );
    }

    /// Build a revocation block with three 8-byte (64-bit) records. The
    /// count field holds jbd2's `r_count`: the total bytes used, i.e.
    /// header (12) + count field (4) + records.
    fn create_test_revocation_block() -> Vec<u8> {
        let mut block = Vec::new();
        // Journal block header (zeroed; only r_count matters here).
        block.extend([0; JournalBlockHeader::SIZE]);
        // r_count = 12 + 4 + 3 * 8 = 40.
        block.extend(40u32.to_be_bytes());
        // Three records.
        block.extend(100u64.to_be_bytes());
        block.extend(101u64.to_be_bytes());
        block.extend(102u64.to_be_bytes());
        // A fourth record past r_count, so it must be ignored.
        block.extend(103u64.to_be_bytes());
        // Pad out to a full block.
        block.resize(1024usize, 0u8);
        block
    }

    /// Test a successful 64-bit revocation read.
    #[test]
    fn test_read_revocation_block_table_success() {
        let block = create_test_revocation_block();
        let mut table = Vec::new();
        read_revocation_block_table(&block, &mut table, true, false).unwrap();
        assert_eq!(table, [100, 101, 102]);
    }

    /// Test a legacy 32-bit revocation block (4-byte records).
    #[test]
    fn test_read_revocation_block_table_32bit() {
        let mut block = Vec::new();
        block.extend([0; JournalBlockHeader::SIZE]);
        // r_count = 12 + 4 + 3 * 4 = 28.
        block.extend(28u32.to_be_bytes());
        block.extend(100u32.to_be_bytes());
        block.extend(101u32.to_be_bytes());
        block.extend(102u32.to_be_bytes());
        block.resize(1024usize, 0u8);
        let mut table = Vec::new();
        read_revocation_block_table(&block, &mut table, false, false).unwrap();
        assert_eq!(table, [100, 101, 102]);
    }

    /// Test that an `r_count` smaller than the header+count is rejected.
    #[test]
    fn test_read_revocation_block_table_too_small() {
        let mut block = create_test_revocation_block();
        block[JournalBlockHeader::SIZE
            ..JournalBlockHeader::SIZE + size_of::<u32>()]
            .copy_from_slice(&7u32.to_be_bytes());
        let mut table = Vec::new();
        assert_eq!(
            read_revocation_block_table(&block, &mut table, true, false)
                .unwrap_err(),
            CorruptKind::JournalRevocationBlockInvalidTableSize(7)
        );
    }

    /// Test that an `r_count` larger than the block is rejected.
    #[test]
    fn test_read_revocation_block_table_size_too_large() {
        let mut block = create_test_revocation_block();
        block[JournalBlockHeader::SIZE
            ..JournalBlockHeader::SIZE + size_of::<u32>()]
            .copy_from_slice(&2000u32.to_be_bytes());
        let mut table = Vec::new();
        assert_eq!(
            read_revocation_block_table(&block, &mut table, true, false)
                .unwrap_err(),
            CorruptKind::JournalRevocationBlockInvalidTableSize(2000)
        );
    }
}
