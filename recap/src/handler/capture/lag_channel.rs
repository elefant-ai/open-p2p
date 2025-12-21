use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize},
};

use parking_lot::Mutex;
use tokio_stream::Stream;
use tracing::trace;

pub fn lag_channel<T>() -> (Sender<T>, Recv<T>) {
    let inner = Arc::new(ChannelInner {
        output: Mutex::new(None),
        closed: AtomicBool::new(false),
        waker: Mutex::new(None),
    });
    let sender = Sender {
        inner: inner.clone(),
        senders: Arc::new(AtomicUsize::new(1)),
    };
    let receiver = Recv { inner };
    (sender, receiver)
}

#[derive(Debug)]
pub struct Sender<T> {
    inner: Arc<ChannelInner<T>>,
    senders: Arc<AtomicUsize>,
}

impl<T> Sender<T> {
    pub fn send(&self, value: T) -> Result<(), T> {
        self.inner.set_item(value)
    }
}

impl<T> Clone for Sender<T> {
    fn clone(&self) -> Self {
        self.senders
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Sender {
            inner: self.inner.clone(),
            senders: self.senders.clone(),
        }
    }
}

impl<T> Drop for Sender<T> {
    fn drop(&mut self) {
        if self
            .senders
            .fetch_sub(1, std::sync::atomic::Ordering::SeqCst)
            <= 1
        {
            self.inner.close();
        }
    }
}

#[derive(Debug)]
pub struct Recv<T> {
    inner: Arc<ChannelInner<T>>,
}

impl<T> Recv<T> {
    /// Try to receive a value from the channel.
    #[allow(unused)]
    pub fn try_recv(&self) -> Option<T> {
        self.inner.take_output()
    }

    /// Receive a value from the channel.
    #[allow(unused)]
    pub fn recv(&self) -> RecvFuture<'_, T> {
        RecvFuture { recv: self }
    }
}

pub struct RecvFuture<'a, T> {
    recv: &'a Recv<T>,
}

impl<T> std::future::Future for RecvFuture<'_, T> {
    type Output = Result<T, anyhow::Error>;

    fn poll(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        if self.recv.inner.is_closed() {
            return std::task::Poll::Ready(Err(anyhow::anyhow!("channel closed")));
        }

        if let Some(output) = self.recv.inner.take_output() {
            std::task::Poll::Ready(Ok(output))
        } else {
            self.recv.inner.set_waker(cx.waker().clone());
            std::task::Poll::Pending
        }
    }
}

impl<T> Drop for Recv<T> {
    fn drop(&mut self) {
        self.inner.close();
    }
}

impl<T> Stream for Recv<T> {
    type Item = T;

    fn poll_next(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        if self.inner.is_closed() {
            return std::task::Poll::Ready(None);
        }

        if let Some(output) = self.inner.take_output() {
            std::task::Poll::Ready(Some(output))
        } else {
            self.inner.set_waker(cx.waker().clone());
            std::task::Poll::Pending
        }
    }
}

#[derive(Debug)]
struct ChannelInner<T> {
    output: Mutex<Option<T>>,
    closed: AtomicBool,
    waker: Mutex<Option<std::task::Waker>>,
}

impl<T> ChannelInner<T> {
    fn take_output(&self) -> Option<T> {
        self.output.lock().take()
    }

    fn set_item(&self, item: T) -> Result<(), T> {
        if self.is_closed() {
            return Err(item);
        }
        if self.output.lock().is_some() {
            trace!("Lag channel already has an item, overwriting it.");
        }
        *self.output.lock() = Some(item);
        self.wake();
        Ok(())
    }

    fn is_closed(&self) -> bool {
        self.closed.load(std::sync::atomic::Ordering::SeqCst)
    }

    fn close(&self) {
        self.closed.store(true, std::sync::atomic::Ordering::SeqCst);
        if let Some(waker) = self.waker.lock().take() {
            waker.wake();
        }
    }

    fn wake(&self) {
        if let Some(waker) = self.waker.lock().take() {
            waker.wake();
        }
    }

    fn set_waker(&self, waker: std::task::Waker) {
        *self.waker.lock() = Some(waker);
    }
}
