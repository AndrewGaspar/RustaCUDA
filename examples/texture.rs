#[macro_use]
extern crate rustacuda;

use ndarray::prelude::*;
use rustacuda::prelude::*;
use std::error::Error;
use std::ffi::CString;

fn main() -> Result<(), Box<dyn Error>> {
    // Set up the context, load the module, and create a stream to run kernels in.
    rustacuda::init(CudaFlags::empty())?;
    let device = Device::get_device(0)?;
    let _ctx = Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;

    let ptx = CString::new(include_str!("../resources/texture.ptx"))?;
    let module = Module::load_from_string(&ptx)?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    const M: usize = 1024;
    const N: usize = 4096;

    /*
        Build an array that looks like:
        img = [
            0, 1, ..., 1022, 1023,
            1024, 1025, ..., 2046, 2047,
            ...,
            1024 * 4095, 1024 * 4095 + 1, ..., 1024 * 4095 + 1022, 1024 * 4095 + 1023
        ]
    */
    let img = Array2::from_shape_fn([M, N], |(i, j)| (i * M + j) as i32);

    for (i, row) in img.genrows().into_iter().enumerate() {
        assert_eq!(
            ((i * M) as i32..(i * M + N) as i32).collect::<Vec<_>>(),
            row.as_slice().unwrap()
        );
    }

    // flipped will contain img reversed around the first axis.
    let mut flipped: Array2<i32> = Array2::zeros([M, N]);

    // Create buffers for data
    // let mut in_x = DeviceBuffer::from_slice(&[1.0f32; 10])?;
    // let mut in_y = DeviceBuffer::from_slice(&[2.0f32; 10])?;
    // let mut out_1 = DeviceBuffer::from_slice(&[0.0f32; 10])?;
    // let mut out_2 = DeviceBuffer::from_slice(&[0.0f32; 10])?;

    // // This kernel adds each element in `in_x` and `in_y` and writes the result into `out`.
    // unsafe {
    //     // Launch the kernel with one block of one thread, no dynamic shared memory on `stream`.
    //     launch!(module.transform_texture<<<1, 1, 0, stream>>>(
    //         in_x.as_device_ptr(),
    //         in_y.as_device_ptr(),
    //         out_1.as_device_ptr(),
    //         out_1.len()
    //     ))?;
    // }

    // Kernel launches are asynchronous, so we wait for the kernels to finish executing.
    stream.synchronize()?;

    // Copy the results back to host memory
    // TODO

    println!("Launched kernel successfully.");
    Ok(())
}
