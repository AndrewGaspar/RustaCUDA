use std::marker::PhantomData;
use std::os::raw::c_uint;

use super::{ArrayDescriptor, ArrayFormat, ArrayObject, ArrayObjectFlags};
use crate::error::*;

/// Maps from concrete types to supported CUDA Array formats.
pub unsafe trait ArrayFormattable {
    /// A CUDA ArrayFormat that the type maps to.
    const FORMAT: ArrayFormat;
}

unsafe impl ArrayFormattable for u8 {
    const FORMAT: ArrayFormat = ArrayFormat::UnsignedInt8;
}

unsafe impl ArrayFormattable for u16 {
    const FORMAT: ArrayFormat = ArrayFormat::UnsignedInt16;
}

unsafe impl ArrayFormattable for u32 {
    const FORMAT: ArrayFormat = ArrayFormat::UnsignedInt32;
}

unsafe impl ArrayFormattable for i8 {
    const FORMAT: ArrayFormat = ArrayFormat::SignedInt8;
}

unsafe impl ArrayFormattable for i16 {
    const FORMAT: ArrayFormat = ArrayFormat::SignedInt16;
}

unsafe impl ArrayFormattable for i32 {
    const FORMAT: ArrayFormat = ArrayFormat::SignedInt32;
}

unsafe impl ArrayFormattable for f32 {
    const FORMAT: ArrayFormat = ArrayFormat::Float;
}

pub type Ix = usize;

pub struct Dim<I: ?Sized>(I);

pub type Ix1 = Dim<[Ix; 1]>;
pub type Ix2 = Dim<[Ix; 2]>;
pub type Ix3 = Dim<[Ix; 3]>;

/// Describes array shape, up to 3-dimensions, which is the maximum that CUDA Arrays support.
pub trait Dimension {
    const NDIM: usize;
}

impl Dimension for Dim<Ix1> {
    const NDIM: usize = 1;
}

impl Dimension for Dim<Ix2> {
    const NDIM: usize = 2;
}

impl Dimension for Dim<Ix3> {
    const NDIM: usize = 3;
}

/// ArrayBuilder assists in building typed Array objects.
#[derive(Debug)]
pub struct ArrayBuilder<T: ArrayFormattable> {
    desc: ArrayDescriptor,
    // Used to keep ArrayBuilder from being named without a valid ArrayFormattable type
    _phantom: PhantomData<T>,
}

impl<T: ArrayFormattable> ArrayBuilder<T> {
    pub fn new_1d(width: usize) -> Self {
        Self::new_3d(width, 0, 0)
    }

    pub fn new_2d(width: usize, height: usize) -> Self {
        Self::new_3d(width, height, 0)
    }

    pub fn new_3d(width: usize, height: usize, depth: usize) -> Self {
        Self {
            desc: ArrayDescriptor::new([width, height, depth], T::FORMAT, 1, Default::default()),
            _phantom: PhantomData,
        }
    }

    pub fn new_layered_1d(width: usize, num_layers: usize) -> Self {
        Self::new_layered_2d(width, 0, num_layers)
    }

    pub fn new_layered_2d(width: usize, height: usize, num_layers: usize) -> Self {
        Self {
            desc: ArrayDescriptor::new(
                [width, height, num_layers],
                T::FORMAT,
                1,
                ArrayObjectFlags::LAYERED,
            ),
            _phantom: PhantomData,
        }
    }

    pub fn new_cubemap(side: usize) -> Self {
        Self {
            desc: ArrayDescriptor::new([side, side, 6], T::FORMAT, 1, ArrayObjectFlags::CUBEMAP),
            _phantom: PhantomData,
        }
    }

    pub fn new_layered_cubemap(side: usize, num_layers: usize) -> Self {
        Self {
            desc: ArrayDescriptor::new(
                [side, side, num_layers * 6],
                T::FORMAT,
                1,
                ArrayObjectFlags::LAYERED | ArrayObjectFlags::CUBEMAP,
            ),
            _phantom: PhantomData,
        }
    }

    pub fn num_channels(mut self, num_channels: c_uint) -> Self {
        if cfg!(debug_assertions) {
            assert!(
                1 == num_channels || 2 == num_channels || 4 == num_channels,
                "num_channels was set to {}. Only 1, 2, and 4 are valid values.",
                num_channels
            );
        }

        self.desc.set_num_channels(num_channels);
        self
    }

    pub fn enable_surface_load_store(mut self, enable: bool) -> Self {
        let mut existing = self.desc.flags();
        existing.set(ArrayObjectFlags::SURFACE_LDST, enable);

        self.desc.set_flags(existing);
        self
    }

    pub fn enable_texture_gather(mut self, enable: bool) -> Self {
        if cfg!(debug_assertions) {
            assert!(
                !self.desc.flags().contains(ArrayObjectFlags::LAYERED),
                "texture_gather is not compatible with LAYERED arrays. desc = {:?}",
                self.desc
            );

            assert!(
                !self.desc.flags().contains(ArrayObjectFlags::CUBEMAP),
                "texture_gather is not compatible with CUBEMAP arrays. desc = {:?}",
                self.desc
            );

            assert!(
                self.desc.width() > 0 && self.desc.height() > 0 && self.desc.depth() == 0,
                "texture_gather is only compatible with 2D arrays. desc = {:?}",
                self.desc
            );
        }

        let mut existing = self.desc.flags();
        existing.set(ArrayObjectFlags::TEXTURE_GATHER, enable);

        self.desc.set_flags(existing);
        self
    }

    pub fn build(self) -> CudaResult<Array<T>> {
        Ok(Array(
            ArrayObject::from_descriptor(&self.desc)?,
            PhantomData,
        ))
    }
}

#[derive(Debug)]
pub struct Array<T: ArrayFormattable>(ArrayObject, PhantomData<T>);

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    #[should_panic]
    fn no_texture_gather_layered_array() {
        let _context = crate::quick_init().unwrap();

        let _array: Array<f32> = ArrayBuilder::new_layered_2d(7, 8, 10)
            .enable_texture_gather(true)
            .build()
            .unwrap();
    }
}
