//! Numeric formats, including the sub-byte weight formats.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DType {
    F32,
    F16,
    BF16,
    /// FP8 with 4 exponent bits — the usual choice for weights and activations.
    F8E4M3,
    /// FP8 with 5 exponent bits — wider range, fewer mantissa bits, used where
    /// activation outliers would otherwise saturate E4M3.
    F8E5M2,
    I8,
    /// 4-bit integer weights. Packed two per byte, so a tensor with an odd
    /// element count still costs a whole trailing byte.
    I4,
}

impl DType {
    pub const fn bits(self) -> usize {
        match self {
            DType::F32 => 32,
            DType::F16 | DType::BF16 => 16,
            DType::F8E4M3 | DType::F8E5M2 | DType::I8 => 8,
            DType::I4 => 4,
        }
    }

    /// Bytes needed for `n` elements, rounding sub-byte formats up.
    pub const fn bytes_for(self, n: usize) -> usize {
        (n * self.bits()).div_ceil(8)
    }

    /// True for formats that need a scale (and possibly a zero point) per
    /// group of elements rather than one per tensor.
    pub const fn is_quantized(self) -> bool {
        matches!(self, DType::F8E4M3 | DType::F8E5M2 | DType::I8 | DType::I4)
    }

    /// Precision that accumulation and reference comparison run in.
    pub const fn accumulator(self) -> DType {
        DType::F32
    }

    /// Relative tolerance a kernel in this format is checked against.
    ///
    /// These are the gate thresholds, not measurements: a kernel whose output
    /// drifts further than this from the F32 reference is rejected regardless
    /// of how fast it is.
    pub const fn default_rtol(self) -> f64 {
        match self {
            DType::F32 => 1e-6,
            DType::F16 => 1e-3,
            DType::BF16 => 8e-3,
            DType::F8E4M3 | DType::F8E5M2 => 6e-2,
            DType::I8 => 6e-2,
            DType::I4 => 1.5e-1,
        }
    }
}

/// How weights are stored, including the overhead of quantization metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct WeightFormat {
    pub dtype: DType,
    /// Elements sharing one scale. `None` means per-tensor scaling.
    pub group_size: Option<usize>,
    /// Whether each group also carries a zero point (asymmetric quantization).
    pub zero_point: bool,
    /// Scales and zero points are kept in this format.
    pub scale_dtype: DType,
}

impl WeightFormat {
    pub const fn dense(dtype: DType) -> Self {
        Self {
            dtype,
            group_size: None,
            zero_point: false,
            scale_dtype: DType::F16,
        }
    }

    /// The common 4-bit setup: 128-element groups, symmetric, FP16 scales.
    pub const fn w4_g128() -> Self {
        Self {
            dtype: DType::I4,
            group_size: Some(128),
            zero_point: true,
            scale_dtype: DType::F16,
        }
    }

    pub const fn w8_per_tensor(dtype: DType) -> Self {
        Self {
            dtype,
            group_size: None,
            zero_point: false,
            scale_dtype: DType::F32,
        }
    }

    /// Storage cost of `n` weights, counting scales and zero points.
    ///
    /// Group metadata is not a rounding error at 4 bits: 128-element groups
    /// with FP16 scales and zero points add 4 bytes per 64 bytes of payload,
    /// which is over 6% on top of the nominal compression ratio.
    pub fn bytes_for(&self, n: usize) -> usize {
        let payload = self.dtype.bytes_for(n);
        let Some(group) = self.group_size else {
            return payload;
        };
        let groups = n.div_ceil(group);
        let per_group = self.scale_dtype.bytes_for(1) * if self.zero_point { 2 } else { 1 };
        payload + groups * per_group
    }

    /// Effective bits per weight once metadata is counted.
    pub fn effective_bits(&self, n: usize) -> f64 {
        if n == 0 {
            return 0.0;
        }
        self.bytes_for(n) as f64 * 8.0 / n as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sub_byte_packing_rounds_up() {
        assert_eq!(DType::I4.bytes_for(2), 1);
        assert_eq!(DType::I4.bytes_for(3), 2, "an odd count still costs a byte");
        assert_eq!(DType::I4.bytes_for(1024), 512);
        assert_eq!(DType::BF16.bytes_for(1024), 2048);
    }

    #[test]
    fn group_metadata_is_counted_against_the_compression_ratio() {
        let f = WeightFormat::w4_g128();
        // 128 weights: 64 bytes of payload, plus one FP16 scale and zero point.
        assert_eq!(f.bytes_for(128), 64 + 4);
        let bits = f.effective_bits(128);
        assert!(
            bits > 4.0 && bits < 4.5,
            "4-bit with g128 lands near 4.25 bits, got {bits}"
        );
    }

    #[test]
    fn per_tensor_formats_carry_no_per_group_overhead() {
        let f = WeightFormat::w8_per_tensor(DType::F8E4M3);
        assert_eq!(f.bytes_for(1000), 1000);
        assert!((f.effective_bits(1000) - 8.0).abs() < 1e-9);
    }
}
