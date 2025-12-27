use fast_image_resize::CpuExtensions;
use fast_image_resize::images::{Image, ImageRef};
use fast_image_resize::pixels::PixelType;
use fast_image_resize::{FilterType, ResizeAlg, ResizeOptions, Resizer};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

#[pyfunction]
pub fn resize_image<'py>(
    py: Python<'py>,
    image_data: &[u8],
    src_height: u32,
    src_width: u32,
    dst_height: u32,
    dst_width: u32,
) -> PyResult<Bound<'py, PyBytes>> {
    // Print warning if compiled in debug mode
    #[cfg(debug_assertions)]
    {
        eprintln!(
            "WARNING: elefant_rust module was compiled in debug mode. Performance may be significantly lower. Compile with --release for production."
        );
    }

    // Example check (optional but good practice):
    let expected_len = (src_width * src_height * 3) as usize; // Assuming U8x3
    if image_data.len() != expected_len {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Input data length {} does not match dimensions {}x{}x3",
            image_data.len(),
            src_width,
            src_height
        )));
    }

    // Create an image reference
    let src_image = ImageRef::new(src_width, src_height, image_data, PixelType::U8x3).unwrap();

    // Create a new image for the destination
    let mut dst_image = Image::new(dst_width, dst_height, PixelType::U8x3);

    // Create a resizer
    let mut resizer = Resizer::new();
    #[cfg(target_arch = "x86_64")]
    unsafe {
        resizer.set_cpu_extensions(CpuExtensions::Avx2);
    }

    let resize_options =
        ResizeOptions::new().resize_alg(ResizeAlg::Interpolation(FilterType::Hamming));
    resizer
        .resize(&src_image, &mut dst_image, Some(&resize_options))
        .unwrap();
    // Create an owned copy before creating PyBytes
    // Extract only the height*width*3 bytes from the buffer
    let buffer = dst_image.buffer();
    Ok(PyBytes::new(py, buffer))
}
