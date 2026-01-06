use crate::zmq_queue::{
    ZMQQueueClient as RustZMQQueueClient, ZMQQueueServer as RustZMQQueueServer,
};
use pyo3::exceptions::{PyRuntimeError, PyTimeoutError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::sync::{Arc, Mutex};

/// Python wrapper for ZMQQueueServer
/// Using Arc<Mutex<>> to make it thread-safe for GIL release
#[pyclass(name = "ZMQQueueServer")]
pub struct PyZMQQueueServer {
    inner: Arc<Mutex<Option<RustZMQQueueServer>>>,
}

#[pymethods]
impl PyZMQQueueServer {
    #[new]
    fn new(url: &str, per_client_max_size: usize, n_clients: usize) -> PyResult<Self> {
        match RustZMQQueueServer::new(url, per_client_max_size, n_clients) {
            Ok(server) => Ok(PyZMQQueueServer {
                inner: Arc::new(Mutex::new(Some(server))),
            }),
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Failed to create server: {}",
                e
            ))),
        }
    }

    /// Put an item into the queue
    ///
    /// Args:
    ///     item: Bytes to send
    ///     wait_seconds: Optional timeout in seconds
    ///     ignore_full: Whether to ignore full queues
    ///
    /// Returns:
    ///     bool: True if successful, False if timed out (only when wait_seconds is set)
    fn put(
        &self,
        py: Python,
        item: &Bound<'_, PyBytes>,
        wait_seconds: Option<u64>,
        ignore_full: Option<bool>,
    ) -> PyResult<bool> {
        let ignore_full = ignore_full.unwrap_or(false);

        // Get the raw bytes from the PyBytes object
        // We need to convert to Arc<[u8]> to share it safely across threads
        let data: Arc<[u8]> = Arc::from(item.as_bytes());

        // Clone the Arc for use in the closure
        let inner = self.inner.clone();

        // Release the GIL for the blocking operation
        let result = py.allow_threads(move || {
            let mut guard = inner.lock().unwrap();
            if let Some(server) = guard.as_mut() {
                server.put(&data, wait_seconds, ignore_full)
            } else {
                Err(anyhow::anyhow!("Server has been closed"))
            }
        });

        match result {
            Ok(success) => Ok(success),
            Err(e) => Err(PyRuntimeError::new_err(format!("Put failed: {}", e))),
        }
    }

    /// Send an item to all clients
    fn put_to_all(&self, py: Python, item: &Bound<'_, PyBytes>) -> PyResult<()> {
        // Get the raw bytes and convert to Arc<[u8]>
        let data: Arc<[u8]> = Arc::from(item.as_bytes());

        // Clone the Arc for use in the closure
        let inner = self.inner.clone();

        // Release the GIL for the blocking operation
        let result = py.allow_threads(move || {
            let mut guard = inner.lock().unwrap();
            if let Some(server) = guard.as_mut() {
                server.put_to_all(&data)
            } else {
                Err(anyhow::anyhow!("Server has been closed"))
            }
        });

        match result {
            Ok(()) => Ok(()),
            Err(e) => Err(PyRuntimeError::new_err(format!("Put to all failed: {}", e))),
        }
    }
}

/// Python wrapper for ZMQQueueClient
/// Using Arc<Mutex<>> to make it thread-safe for GIL release
#[pyclass(name = "ZMQQueueClient")]
pub struct PyZMQQueueClient {
    inner: Arc<Mutex<RustZMQQueueClient>>,
}

#[pymethods]
impl PyZMQQueueClient {
    #[new]
    fn new(url: &str, client_id: usize, get_timeout_seconds: Option<u64>) -> PyResult<Self> {
        match RustZMQQueueClient::new(url, client_id, get_timeout_seconds.map(|s| s * 1000)) {
            Ok(client) => Ok(PyZMQQueueClient {
                inner: Arc::new(Mutex::new(client)),
            }),
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Failed to create client: {}",
                e
            ))),
        }
    }

    /// Get an item from the queue
    ///
    /// Returns:
    ///     bytes or None: The received data or None if server is shutting down
    ///
    /// Raises:
    ///     TimeoutError: If no data received within timeout
    fn get(&self, py: Python) -> PyResult<Option<PyObject>> {
        // Clone the Arc for use in the closure
        let inner = self.inner.clone();

        // Release the GIL for the blocking operation
        let result = py.allow_threads(move || {
            let guard = inner.lock().unwrap();
            guard.get()
        });

        match result {
            Ok(data) => Ok(Some(PyBytes::new(py, &data).into())),
            Err(crate::zmq_queue::ZmqQueueError::Timeout) => {
                Err(PyTimeoutError::new_err("timed out waiting for item"))
            }
            Err(crate::zmq_queue::ZmqQueueError::Other(e)) => {
                Err(PyRuntimeError::new_err(format!("Get failed: {}", e)))
            }
        }
    }
}

/// Create Python module for the ZMQ queue bindings
pub fn add_zmq_queue_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let zmq_module = PyModule::new(m.py(), "zmq_queue")?;

    zmq_module.add_class::<PyZMQQueueServer>()?;
    zmq_module.add_class::<PyZMQQueueClient>()?;

    m.add_submodule(&zmq_module)?;
    Ok(())
}
