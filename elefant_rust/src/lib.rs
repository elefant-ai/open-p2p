pub mod ffmpeg_decoder;
pub mod resize;
pub mod shuffle_thread;
pub mod zmq_queue;
pub mod zmq_queue_py;

use pyo3::prelude::*;
use resize::resize_image;
use shuffle_thread::PyShuffleThread;
use zmq_queue_py::add_zmq_queue_module;

/// A Python module implemented in Rust.
#[pymodule]
fn elefant_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(resize_image, m)?)?;

    // Add the ZMQ queue submodule
    add_zmq_queue_module(m)?;

    // Create video_proto_dataset submodule
    let video_proto_dataset = PyModule::new(m.py(), "video_proto_dataset")?;
    video_proto_dataset.add_class::<PyShuffleThread>()?;
    m.add_submodule(&video_proto_dataset)?;

    Ok(())
}
