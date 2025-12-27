use crate::zmq_queue::{ZMQQueueClient, ZMQQueueServer, ZmqQueueError};
use anyhow::{Result, anyhow};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use rand::prelude::*;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::thread;
use std::time::Duration;

// Hacky way of dealing with Python.
// None is always pickled as this byte string.
// We can remove this once the whole dataset is running in Rust.
const PY_NONE_PICKLE_STR: &[u8] = b"\x80\x05N.";

/// Interface for shuffling strategies
trait ShufflerInterface {
    /// Add a single item to the shuffler
    fn add_item(&mut self, example: &[u8]) -> Result<()>;

    /// Finish the current epoch
    fn finish_epoch(&mut self) -> Result<()>;
}

/// NoShuffle implementation - directly passes items through without shuffling
struct NoShuffle {
    shuffle_queue: ZMQQueueServer,
    n_examples_in_epoch: i64,
    epoch: i64,
    dataset_unique_id: String,
}

impl NoShuffle {
    fn new(shuffle_queue: ZMQQueueServer, dataset_unique_id: String) -> Self {
        NoShuffle {
            shuffle_queue,
            n_examples_in_epoch: 0,
            epoch: 0,
            dataset_unique_id,
        }
    }
}

impl ShufflerInterface for NoShuffle {
    fn add_item(&mut self, example: &[u8]) -> Result<()> {
        self.n_examples_in_epoch += 1;
        self.shuffle_queue.put(example, None, false)?;
        Ok(())
    }

    fn finish_epoch(&mut self) -> Result<()> {
        println!(
            "All preprocess threads done epoch {}, n_examples_in_epoch={}. {}",
            self.epoch, self.n_examples_in_epoch, self.dataset_unique_id
        );
        self.epoch += 1;
        self.n_examples_in_epoch = 0;
        self.shuffle_queue.put_to_all(PY_NONE_PICKLE_STR)?;
        Ok(())
    }
}

/// ShuffleBuffer implementation - maintains a buffer and randomly shuffles items
struct ShuffleBuffer {
    shuffle_queue: ZMQQueueServer,
    buffer: Vec<Vec<u8>>,
    max_buffer_size: usize,
    rng: StdRng,
    n_examples_in_epoch: i64,
    epoch: i64,
    dataset_unique_id: String,
}

impl ShuffleBuffer {
    fn new(
        shuffle_queue: ZMQQueueServer,
        dataset_unique_id: String,
        max_buffer_size: usize,
        rng_seed: u64,
    ) -> Self {
        ShuffleBuffer {
            shuffle_queue,
            buffer: Vec::with_capacity(max_buffer_size),
            max_buffer_size,
            rng: StdRng::seed_from_u64(rng_seed),
            n_examples_in_epoch: 0,
            epoch: 0,
            dataset_unique_id,
        }
    }
}

impl ShufflerInterface for ShuffleBuffer {
    fn add_item(&mut self, example: &[u8]) -> Result<()> {
        self.n_examples_in_epoch += 1;

        if self.buffer.len() < self.max_buffer_size {
            // Buffer not full, just add the item
            self.buffer.push(example.to_vec());
            if self.buffer.len() % 500 == 0 {
                println!(
                    "Filling shuffle buffer {} Buffer size: {} / {}",
                    self.dataset_unique_id,
                    self.buffer.len(),
                    self.max_buffer_size
                );
            }
        } else {
            // Buffer is full, randomly sample and remove an item
            let idx = self.rng.gen_range(0..self.buffer.len());
            let sampled_item = self.buffer.swap_remove(idx);
            self.shuffle_queue.put(&sampled_item, None, false)?;

            // Add the new item to the buffer
            self.buffer.push(example.to_vec());
        }

        Ok(())
    }

    fn finish_epoch(&mut self) -> Result<()> {
        // Randomly sample and send all remaining items in the buffer
        while !self.buffer.is_empty() {
            let idx = self.rng.gen_range(0..self.buffer.len());
            let sampled_item = self.buffer.swap_remove(idx);
            self.shuffle_queue.put(&sampled_item, None, false)?;
        }

        println!(
            "All preprocess threads done epoch {}, n_examples_in_epoch={}. {}",
            self.epoch, self.n_examples_in_epoch, self.dataset_unique_id
        );
        self.epoch += 1;
        self.n_examples_in_epoch = 0;
        self.shuffle_queue.put_to_all(PY_NONE_PICKLE_STR)?;
        Ok(())
    }
}

/// Configuration for the shuffle thread
#[derive(Clone)]
pub struct ShuffleConfig {
    pub dataset_unique_id: String,
    pub shuffle: bool,
    pub shuffle_buffer_size: usize,
    pub shuffled_chunks_queue_size: usize,
    pub n_preprocess_workers: usize,
    pub n_dataset_workers: usize,
    pub warn_on_starvation: bool,
    pub shuffle_rng_seed: u64,
}

/// Python wrapper for ShuffleThread
#[pyclass(name = "ShuffleThread")]
pub struct PyShuffleThread {
    handle: Option<thread::JoinHandle<()>>,
    shutdown_flag: Arc<AtomicBool>,
}

#[pymethods]
impl PyShuffleThread {
    #[new]
    #[pyo3(signature = (dataset_unique_id, shuffle, shuffle_buffer_size, shuffled_chunks_queue_size, n_preprocess_workers, n_dataset_workers, warn_on_starvation=true, shuffle_rng_seed=43))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        dataset_unique_id: String,
        shuffle: bool,
        shuffle_buffer_size: usize,
        shuffled_chunks_queue_size: usize,
        n_preprocess_workers: usize,
        n_dataset_workers: usize,
        warn_on_starvation: bool,
        shuffle_rng_seed: u64,
    ) -> PyResult<Self> {
        let config = ShuffleConfig {
            dataset_unique_id,
            shuffle,
            shuffle_buffer_size,
            shuffled_chunks_queue_size,
            n_preprocess_workers,
            n_dataset_workers,
            warn_on_starvation,
            shuffle_rng_seed,
        };

        let shutdown_flag = Arc::new(AtomicBool::new(false));
        let shutdown_flag_clone = shutdown_flag.clone();

        // Start the shuffle thread
        let handle = thread::spawn(move || {
            if let Err(e) = run_shuffle_worker(config, shutdown_flag_clone) {
                eprintln!("Shuffle worker error: {}", e);
            }
        });

        Ok(PyShuffleThread {
            handle: Some(handle),
            shutdown_flag,
        })
    }

    /// Signal the shuffle thread to shutdown gracefully
    fn shutdown(&mut self) -> PyResult<()> {
        self.shutdown_flag.store(true, Ordering::Relaxed);
        Ok(())
    }

    /// Wait for the shuffle thread to finish with a timeout
    fn join(&mut self, py: Python, timeout_seconds: Option<f64>) -> PyResult<bool> {
        if let Some(handle) = self.handle.take() {
            let timeout = timeout_seconds.map(Duration::from_secs_f64);

            py.allow_threads(|| {
                if let Some(timeout) = timeout {
                    // Use a simple polling approach since JoinHandle doesn't have timeout
                    let start = std::time::Instant::now();
                    while start.elapsed() < timeout {
                        if handle.is_finished() {
                            handle.join().map_err(|_| anyhow!("Thread panicked"))?;
                            return Ok(true);
                        }
                        std::thread::sleep(Duration::from_millis(10));
                    }
                    Ok(false) // Timeout
                } else {
                    handle.join().map_err(|_| anyhow!("Thread panicked"))?;
                    Ok(true)
                }
            })
            .map_err(|e: anyhow::Error| {
                PyRuntimeError::new_err(format!("Failed to join thread: {}", e))
            })
        } else {
            Ok(true) // Already joined
        }
    }

    /// Check if the thread is still running
    fn is_alive(&self) -> bool {
        self.handle.as_ref().is_some_and(|h| !h.is_finished())
    }

    /// Shutdown and join with timeout (convenience method)
    fn stop(&mut self, py: Python, timeout_seconds: Option<f64>) -> PyResult<bool> {
        self.shutdown()?;
        self.join(py, timeout_seconds)
    }
}

// Implement Drop to ensure clean shutdown
impl Drop for PyShuffleThread {
    fn drop(&mut self) {
        // Signal shutdown
        self.shutdown_flag.store(true, Ordering::Relaxed);

        // Try to join with a short timeout
        if let Some(handle) = self.handle.take() {
            // Give it a brief moment to shutdown gracefully
            let start = std::time::Instant::now();
            while start.elapsed() < Duration::from_millis(100) {
                if handle.is_finished() {
                    let _ = handle.join();
                    return;
                }
                std::thread::sleep(Duration::from_millis(1));
            }
            // If it doesn't finish quickly, just detach (the process will clean it up)
        }
    }
}

/// Get the ZMQ queue address
fn get_zeromq_queue_addr(dataset_unique_id: &str, queue_name: &str) -> String {
    let zmq_tempdir = format!("/tmp/elefant_zmq/zmq_{}", dataset_unique_id);
    std::fs::create_dir_all(&zmq_tempdir).ok();
    format!("ipc://{}/{}", zmq_tempdir, queue_name)
}

/// Main shuffle worker logic with shutdown support
#[allow(clippy::never_loop)]
fn run_shuffle_worker(config: ShuffleConfig, shutdown_flag: Arc<AtomicBool>) -> Result<()> {
    let pid = std::process::id();
    println!(
        "Shuffle worker pid={} started for dataset {}, n_dataset_workers={}",
        pid, config.dataset_unique_id, config.n_dataset_workers
    );

    // Create the shuffle queue server
    let shuffle_queue = ZMQQueueServer::new(
        &get_zeromq_queue_addr(&config.dataset_unique_id, "shuffle"),
        config.shuffled_chunks_queue_size,
        config.n_dataset_workers,
    )?;

    // Create the shuffler based on config
    let mut shuffler: Box<dyn ShufflerInterface> = if config.shuffle {
        Box::new(ShuffleBuffer::new(
            shuffle_queue,
            config.dataset_unique_id.clone(),
            config.shuffle_buffer_size,
            config.shuffle_rng_seed,
        ))
    } else {
        Box::new(NoShuffle::new(
            shuffle_queue,
            config.dataset_unique_id.clone(),
        ))
    };

    println!(
        "Shuffle worker {} setting up preprocessed chunks queues.",
        config.dataset_unique_id
    );

    // If in shuffle mode have a short-timeout for the preprocess queues.
    // This is in milliseconds.
    let preprocess_queue_timeout = if config.shuffle {
        Some(10)
    } else {
        Some(60 * 1000)
    };

    // Create clients for all preprocess queues
    let mut preprocessed_chunks_queues = Vec::new();
    for i in 0..config.n_preprocess_workers {
        let client = ZMQQueueClient::new(
            &get_zeromq_queue_addr(&config.dataset_unique_id, &format!("preprocess_{}", i)),
            0, // client_id
            // We have a short timeout here to avoid starving the shuffle thread,
            // if a thread is not ready we just moved on to the next one.
            preprocess_queue_timeout,
        )?;
        preprocessed_chunks_queues.push(client);
    }

    println!(
        "Shuffle worker {} finished setting up preprocessed chunks queues.",
        config.dataset_unique_id
    );

    // Forever loop composed of nested epoch loops
    loop {
        let mut workers_still_running = vec![true; preprocessed_chunks_queues.len()];
        // Epoch loop.
        loop {
            assert!(workers_still_running.iter().any(|&x| x));

            // Interleave chunks from each worker
            for i in 0..preprocessed_chunks_queues.len() {
                // Check shutdown flag
                if shutdown_flag.load(Ordering::Relaxed) {
                    return Ok(());
                }

                if workers_still_running[i] {
                    let start_time = if config.warn_on_starvation {
                        Some(std::time::Instant::now())
                    } else {
                        None
                    };

                    match preprocessed_chunks_queues[i].get() {
                        Ok(example) if example == PY_NONE_PICKLE_STR => {
                            workers_still_running[i] = false;
                            let n_workers_left: usize =
                                workers_still_running.iter().filter(|&&x| x).count();
                            println!(
                                "Preprocess queue {} empty, {} workers left.",
                                i, n_workers_left
                            );
                            if n_workers_left == 0 {
                                shuffler.finish_epoch()?;
                                // Set all workers_still_running back to true for the new epoch.
                                workers_still_running =
                                    vec![true; preprocessed_chunks_queues.len()];
                                break;
                            }
                        }
                        Ok(example) => {
                            if let Some(start) = start_time {
                                let elapsed = start.elapsed().as_millis();
                                if elapsed > 50 {
                                    println!("chunk queue {} starved for {} ms", i, elapsed);
                                }
                            }

                            shuffler.add_item(&example)?;
                        }
                        Err(ZmqQueueError::Timeout) => {
                            // No data currently available from this queue; try the next one
                            // println!("Preprocess queue {} timed out, moving on.", i);
                            continue;
                        }
                        Err(ZmqQueueError::Other(err)) => {
                            println!(
                                "Preprocess queue {} error. {}. {}",
                                i, config.dataset_unique_id, err
                            );
                            if shutdown_flag.load(Ordering::Relaxed) {
                                return Ok(());
                            }
                            panic!(
                                "Preprocess queue {} error. {}. {}",
                                i, config.dataset_unique_id, err
                            );
                        }
                    }
                }
            }
        }
    }
}
