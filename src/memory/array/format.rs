use cuda_sys::cuda::{CUarray_format, CUarray_format_enum};

/// Describes the format used for a CUDA Array.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArrayFormat {
    /// Unsigned 8-bit integer
    UnsignedInt8,
    /// Unsigned 16-bit integer
    UnsignedInt16,
    /// Unsigned 32-bit integer
    UnsignedInt32,
    /// Signed 8-bit integer
    SignedInt8,
    /// Signed 16-bit integer
    SignedInt16,
    /// Signed 32-bit integer
    SignedInt32,
    /// Half-precision floating point number
    Half,
    /// Single-precision floating point number
    Float,
}

impl ArrayFormat {
    /// Creates ArrayFormat from the CUDA Driver API enum
    pub fn from_raw(raw: CUarray_format) -> Self {
        match raw {
            CUarray_format_enum::CU_AD_FORMAT_UNSIGNED_INT8 => ArrayFormat::UnsignedInt8,
            CUarray_format_enum::CU_AD_FORMAT_UNSIGNED_INT16 => ArrayFormat::UnsignedInt16,
            CUarray_format_enum::CU_AD_FORMAT_UNSIGNED_INT32 => ArrayFormat::UnsignedInt32,
            CUarray_format_enum::CU_AD_FORMAT_SIGNED_INT8 => ArrayFormat::SignedInt8,
            CUarray_format_enum::CU_AD_FORMAT_SIGNED_INT16 => ArrayFormat::SignedInt16,
            CUarray_format_enum::CU_AD_FORMAT_SIGNED_INT32 => ArrayFormat::SignedInt32,
            CUarray_format_enum::CU_AD_FORMAT_HALF => ArrayFormat::Half,
            CUarray_format_enum::CU_AD_FORMAT_FLOAT => ArrayFormat::Float,
        }
    }

    /// Converts ArrayFormat to the CUDA Driver API enum
    pub fn to_raw(self) -> CUarray_format {
        match self {
            ArrayFormat::UnsignedInt8 => CUarray_format_enum::CU_AD_FORMAT_UNSIGNED_INT8,
            ArrayFormat::UnsignedInt16 => CUarray_format_enum::CU_AD_FORMAT_UNSIGNED_INT16,
            ArrayFormat::UnsignedInt32 => CUarray_format_enum::CU_AD_FORMAT_UNSIGNED_INT32,
            ArrayFormat::SignedInt8 => CUarray_format_enum::CU_AD_FORMAT_SIGNED_INT8,
            ArrayFormat::SignedInt16 => CUarray_format_enum::CU_AD_FORMAT_SIGNED_INT16,
            ArrayFormat::SignedInt32 => CUarray_format_enum::CU_AD_FORMAT_SIGNED_INT32,
            ArrayFormat::Half => CUarray_format_enum::CU_AD_FORMAT_HALF,
            ArrayFormat::Float => CUarray_format_enum::CU_AD_FORMAT_FLOAT,
        }
    }
}

pub trait ArrayFormattable {
    fn array_format() -> ArrayFormat;
}

impl ArrayFormattable for u8 {
    fn array_format() -> ArrayFormat {
        ArrayFormat::UnsignedInt8
    }
}

impl ArrayFormattable for u16 {
    fn array_format() -> ArrayFormat {
        ArrayFormat::UnsignedInt16
    }
}

impl ArrayFormattable for u32 {
    fn array_format() -> ArrayFormat {
        ArrayFormat::UnsignedInt32
    }
}

impl ArrayFormattable for i8 {
    fn array_format() -> ArrayFormat {
        ArrayFormat::SignedInt8
    }
}

impl ArrayFormattable for i16 {
    fn array_format() -> ArrayFormat {
        ArrayFormat::SignedInt16
    }
}

impl ArrayFormattable for i32 {
    fn array_format() -> ArrayFormat {
        ArrayFormat::SignedInt32
    }
}

impl ArrayFormattable for f32 {
    fn array_format() -> ArrayFormat {
        ArrayFormat::Float
    }
}
