use anyhow::{Result, anyhow};
use std::collections::HashSet;
use std::error::Error as StdError;
use std::fmt;
use zmq::{Context, Socket, SocketType};

// Note: We intentionally hold a Context inside server/client structs to ensure
// the ZMQ context outlives the sockets. These fields may appear unused but are
// required for correctness.

#[derive(Debug)]
pub enum ZmqQueueError {
    Timeout,
    Other(anyhow::Error),
}

impl fmt::Display for ZmqQueueError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ZmqQueueError::Timeout => write!(f, "timed out waiting for item"),
            ZmqQueueError::Other(e) => write!(f, "{}", e),
        }
    }
}

impl StdError for ZmqQueueError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            ZmqQueueError::Timeout => None,
            ZmqQueueError::Other(e) => Some(e.root_cause()),
        }
    }
}

pub type ZmqResult<T> = std::result::Result<T, ZmqQueueError>;

pub struct ZMQQueueServer {
    url: String,
    per_client_max_size: usize,
    n_clients: usize,
    #[allow(dead_code)]
    context: Context,
    socket: Socket,
    client_queue_sizes: Vec<usize>,
    last_sent_idx: usize,
}

impl ZMQQueueServer {
    pub fn new(url: &str, per_client_max_size: usize, n_clients: usize) -> Result<Self> {
        let context = Context::new();
        let socket = context.socket(SocketType::ROUTER)?;
        socket.bind(url)?;

        // println!(
        //     "ZMQQueueServer bound to {}, max_per_client_size={}, n_clients={}",
        //     url, per_client_max_size, n_clients
        // );

        let server = ZMQQueueServer {
            url: url.to_string(),
            per_client_max_size,
            n_clients,
            context,
            socket,
            client_queue_sizes: vec![0; n_clients],
            last_sent_idx: 0,
        };

        // Wait for all clients to connect
        server.wait_for_clients(b"ready")?;

        Ok(server)
    }

    fn wait_for_clients(&self, expected_msg: &[u8]) -> Result<()> {
        // Set timeout to 4 minutes for initial connection
        self.socket.set_rcvtimeo(240_000)?;

        // println!(
        //     "ZMQQueueServer {} waiting for {} clients to send {:?}",
        //     self.url,
        //     self.n_clients,
        //     std::str::from_utf8(expected_msg).unwrap_or("binary")
        // );

        let mut connected_clients = HashSet::new();

        while connected_clients.len() < self.n_clients {
            let message = match self.socket.recv_multipart(0) {
                Ok(msg) => msg,
                Err(zmq::Error::EAGAIN) => {
                    return Err(anyhow!(
                        "{} ZMQQueueServer waiting for clients timed out, connected_clients={:?}",
                        self.url,
                        connected_clients
                    ));
                }
                Err(e) => return Err(e.into()),
            };

            if message.len() != 2 {
                continue;
            }

            let identity = &message[0];
            let content = &message[1];

            if content == expected_msg {
                let client_id = String::from_utf8_lossy(identity).parse::<usize>()?;
                if !connected_clients.insert(client_id) {
                    return Err(anyhow!("Client {} already connected", client_id));
                }
                // println!(
                //     "{} client {} sent {:?}",
                //     self.url,
                //     client_id,
                //     std::str::from_utf8(expected_msg).unwrap_or("binary")
                // );
            } else {
                println!(
                    "{} client {:?} sent {:?}, while server waiting for {:?}",
                    self.url,
                    identity,
                    std::str::from_utf8(content).unwrap_or("binary"),
                    std::str::from_utf8(expected_msg).unwrap_or("binary")
                );
            }
        }

        // Verify client IDs are 0..n_clients
        let mut sorted_clients: Vec<_> = connected_clients.into_iter().collect();
        sorted_clients.sort();
        if sorted_clients != (0..self.n_clients).collect::<Vec<_>>() {
            return Err(anyhow!("Client IDs must be 0..{}", self.n_clients));
        }

        // println!(
        //     "{} All clients sent {:?}",
        //     self.url,
        //     std::str::from_utf8(expected_msg).unwrap_or("binary")
        // );
        Ok(())
    }

    pub fn put(
        &mut self,
        item: &[u8],
        wait_seconds: Option<u64>,
        ignore_full: bool,
    ) -> Result<bool> {
        self.check_for_acks(false, None)?;

        loop {
            // Print the client queue sizes
            // println!("{} put ZMQQueueServer client_queue_sizes={:?}", self.url, self.client_queue_sizes);

            // Try to send to the next client in round-robin order
            if self.client_queue_sizes[self.last_sent_idx] < self.per_client_max_size || ignore_full
            {
                self.send_to_client(self.last_sent_idx, item)?;
                self.last_sent_idx = (self.last_sent_idx + 1) % self.n_clients;
                return Ok(true);
            }

            // Current client is full, try others
            let mut found = false;
            for i in 1..self.n_clients {
                let idx = (self.last_sent_idx + i) % self.n_clients;
                if self.client_queue_sizes[idx] < self.per_client_max_size {
                    self.send_to_client(idx, item)?;
                    found = true;
                    break;
                }
            }

            if found {
                return Ok(true);
            }

            // All clients are full, wait for acks
            let n_acks = self.check_for_acks(true, wait_seconds)?;
            if n_acks == 0 && wait_seconds.is_some() {
                return Ok(false);
            } else if n_acks == 0 {
                return Err(anyhow!(
                    "{} ZMQQueueServer is full, client_queue_sizes={:?}",
                    self.url,
                    self.client_queue_sizes
                ));
            }
        }
    }

    pub fn put_to_all(&mut self, item: &[u8]) -> Result<()> {
        for i in 0..self.n_clients {
            self.send_to_client(i, item)?;
        }
        Ok(())
    }

    fn send_to_client(&mut self, client_idx: usize, item: &[u8]) -> Result<()> {
        self.socket
            .send_multipart([client_idx.to_string().as_bytes(), item], 0)?;
        self.client_queue_sizes[client_idx] += 1;
        Ok(())
    }

    fn check_for_acks(&mut self, wait: bool, wait_seconds: Option<u64>) -> Result<usize> {
        let mut n_acks = 0;
        let wait_seconds = wait_seconds.unwrap_or(24 * 60 * 60);

        //println!("{} check_for_acks ZMQQueueServer waiting for acks, wait={}, wait_seconds={}", self.url, wait, wait_seconds);
        loop {
            let flags = if wait && n_acks == 0 {
                self.socket.set_rcvtimeo((wait_seconds * 1000) as i32)?;
                0
            } else {
                zmq::DONTWAIT
            };

            let message = match self.socket.recv_multipart(flags) {
                Ok(msg) => msg,
                Err(zmq::Error::EAGAIN) => return Ok(n_acks),
                Err(zmq::Error::EINTR) => {
                    // println!("{} check_for_acks ZMQQueueServer received EINTR", self.url);
                    continue;
                }
                Err(e) => return Err(e.into()),
            };

            let identity = &message[0];
            let content = &message[1];
            //println!("{} check_for_acks ZMQQueueServer received message {:?} from client {:?}", self.url, identity, content);
            let client_id = String::from_utf8_lossy(identity).parse::<usize>()?;

            if content == b"ack" {
                if self.client_queue_sizes[client_id] == 0 {
                    return Err(anyhow!(
                        "Received ack from client {} with queue size 0",
                        client_id
                    ));
                }
                self.client_queue_sizes[client_id] -= 1;
                n_acks += 1;
                //println!("{} check_for_acks ZMQQueueServer received ack from client {}", self.url, client_id);
            } else {
                //println!("{} check_for_acks ZMQQueueServer received message from client {}", self.url, client_id);
                return Err(anyhow!(
                    "ZMQQueueServer received message from client {}",
                    client_id
                ));
            }
        }
    }
}

pub struct ZMQQueueClient {
    #[allow(dead_code)]
    url: String,
    #[allow(dead_code)]
    client_id: String,
    #[allow(dead_code)]
    context: Context,
    socket: Socket,
    #[allow(dead_code)]
    get_timeout_seconds: u64,
}

impl ZMQQueueClient {
    pub fn new(url: &str, client_id: usize, get_timeout_milliseconds: Option<u64>) -> Result<Self> {
        let context = Context::new();
        let socket = context.socket(SocketType::DEALER)?;

        let client_id_str = client_id.to_string();
        socket.set_identity(client_id_str.as_bytes())?;

        let timeout = get_timeout_milliseconds.unwrap_or(10 * 60 * 1000);
        socket.set_rcvtimeo(timeout as i32)?;

        socket.connect(url)?;

        // println!(
        //     "ZMQQueueClient {}, id {} connected to server, sending ready.",
        //     url, client_id
        // );

        socket.send(b"ready" as &[u8], 0)?;

        Ok(ZMQQueueClient {
            url: url.to_string(),
            client_id: client_id_str,
            context,
            socket,
            get_timeout_seconds: timeout,
        })
    }

    pub fn get(&self) -> ZmqResult<Vec<u8>> {
        loop {
            match self.socket.recv_bytes(0) {
                Ok(bytes) => {
                    // Acknowledge receipt
                    //println!("ZMQQueueClient {}, id {} sending ack", self.url, self.client_id);
                    self.socket
                        .send(b"ack" as &[u8], 0)
                        .map_err(|e| ZmqQueueError::Other(e.into()))?;
                    // println!("ZMQQueueClient {}, id {} sent ack", self.url, self.client_id);
                    return Ok(bytes);
                }
                Err(zmq::Error::EAGAIN) => return Err(ZmqQueueError::Timeout),
                Err(zmq::Error::EINTR) => {
                    // println!(
                    //     "{} ZMQQueueClient client id {} received EINTR",
                    //     self.url, self.client_id
                    // );
                    continue;
                }
                Err(e) => return Err(ZmqQueueError::Other(e.into())),
            };
        }
    }
}
