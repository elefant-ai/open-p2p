use iced::Subscription;
use iced::futures::{SinkExt as _, Stream};

static ERROR_CHANNEL: std::sync::OnceLock<tokio::sync::mpsc::Sender<Outside>> =
    std::sync::OnceLock::new();

pub fn error_stream() -> impl Stream<Item = crate::Message> {
    iced::stream::channel(
        100,
        |mut output: iced::futures::channel::mpsc::Sender<crate::Message>| async move {
            let (tx, mut rx) = tokio::sync::mpsc::channel::<Outside>(100);
            ERROR_CHANNEL.get_or_init(|| tx);

            while let Some(message) = rx.recv().await {
                match message {
                    Outside::Error((uuid, error)) => {
                        output
                            .send(crate::Message::SetError(uuid, error))
                            .await
                            .unwrap();
                    }
                    Outside::Message(message) => {
                        output.send(message).await.unwrap();
                    }
                }
            }
        },
    )
}

enum Outside {
    Error((uuid::Uuid, Option<String>)),
    Message(crate::Message),
}

pub fn subscription(_: &crate::App) -> Subscription<crate::Message> {
    Subscription::run(error_stream)
}

/// Send an error message to the error stream
pub fn send_error(uuid: uuid::Uuid, error: Option<String>) {
    let tx = ERROR_CHANNEL.get().unwrap();
    tx.try_send(Outside::Error((uuid, error))).unwrap();
}

/// Send an error message to the error stream
pub fn send_message(message: crate::Message) {
    let tx = ERROR_CHANNEL.get().unwrap();
    tx.try_send(Outside::Message(message)).unwrap();
}

/// Clear an error message from the error stream
#[expect(unused)]
pub fn clear_error(uuid: uuid::Uuid) {
    send_error(uuid, None);
}
