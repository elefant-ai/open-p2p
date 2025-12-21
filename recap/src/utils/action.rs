use iced::{Task, futures::future::Either};

mod sealed {
    #[allow(missing_debug_implementations)]
    pub struct UnsetT;
}
use sealed::*;

#[allow(missing_debug_implementations)]
pub enum ActionTask<Msg, T = UnsetT> {
    Task(Task<Msg>),
    AppMessage(crate::Message),
    Custom(T),
}

impl<Msg> ActionTask<Msg, UnsetT> {
    pub fn handle(
        self,
        app: &mut crate::App,
        mapper: impl FnMut(Msg) -> crate::Message + Send + 'static,
    ) -> Task<crate::Message>
    where
        Msg: Send + 'static,
    {
        match self {
            ActionTask::Task(task) => task.map(mapper),
            ActionTask::AppMessage(msg) => app.update(msg),
            ActionTask::Custom(_) => {
                unimplemented!("Custom action not implemented change custom type to not be unit")
            }
        }
    }
}

impl<Msg, T> ActionTask<Msg, T> {
    pub fn handle_with_custom(
        self,
        app: &mut crate::App,
        mapper: impl FnMut(Msg) -> crate::Message + Send + 'static,
    ) -> Either<T, Task<crate::Message>>
    where
        Msg: Send + 'static,
    {
        match self {
            ActionTask::Task(task) => Either::Right(task.map(mapper)),
            ActionTask::AppMessage(msg) => Either::Right(app.update(msg)),
            ActionTask::Custom(custom) => Either::Left(custom),
        }
    }
}

pub trait Action<Msg, T = UnsetT> {
    /// Converts the action into an `ActionTask`.
    fn tat(self) -> ActionTask<Msg, T>;
}

impl<Msg, T> Action<Msg, T> for Task<Msg> {
    fn tat(self) -> ActionTask<Msg, T> {
        ActionTask::Task(self)
    }
}

impl<Msg, T> Action<Msg, T> for crate::Message {
    fn tat(self) -> ActionTask<Msg, T> {
        ActionTask::AppMessage(self)
    }
}

impl<Msg, T> Action<Msg, T> for () {
    fn tat(self) -> ActionTask<Msg, T> {
        ActionTask::Task(Task::none())
    }
}
