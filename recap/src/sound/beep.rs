use std::{
    f32::consts::PI,
    time::{Duration, Instant},
};

use rodio::{Source, source::SeekError};

use super::append_source;

pub fn beep() {
    let source =
        rodio::source::SineWave::new(440.).take_duration(std::time::Duration::from_millis(100));
    append_source(source);
}

pub fn double_beep() {
    let hz = 440.;
    let (input, output) = rodio::queue::queue(false);
    let source =
        rodio::source::SineWave::new(hz).take_duration(std::time::Duration::from_millis(100));
    input.append(source);
    let source = rodio::source::SineWave::new(hz)
        .take_duration(std::time::Duration::from_millis(100))
        .delay(Duration::from_millis(100));
    input.append(source);
    append_source(output);
}

pub fn long_beep() {
    let source =
        rodio::source::SineWave::new(350.).take_duration(std::time::Duration::from_millis(1000));
    append_source(source);
}

#[derive(Clone, Debug)]
pub struct Beep {
    freq: f32,
    num_sample: usize,
    duration: Duration,
    now: Instant,
}

impl Beep {
    /// The frequency of the sine.
    #[inline]
    #[allow(unused)]
    pub fn new(freq: f32, duration: Duration) -> Beep {
        Beep {
            freq,
            num_sample: 0,
            duration,
            now: Instant::now(),
        }
    }
}

impl Iterator for Beep {
    type Item = f32;

    #[inline]
    fn next(&mut self) -> Option<f32> {
        if self.now.elapsed() >= self.duration {
            return None;
        }
        self.num_sample = self.num_sample.wrapping_add(1);

        let value = 2.0 * PI * self.freq * self.num_sample as f32 / 48000.0;
        Some(value.sin())
    }
}

impl Source for Beep {
    #[inline]
    fn current_span_len(&self) -> Option<usize> {
        None
    }

    #[inline]
    fn channels(&self) -> u16 {
        1
    }

    #[inline]
    fn sample_rate(&self) -> u32 {
        48000
    }

    #[inline]
    fn total_duration(&self) -> Option<Duration> {
        Some(self.duration)
    }

    #[inline]
    fn try_seek(&mut self, _: Duration) -> Result<(), SeekError> {
        // This is a constant sound, normal seeking would not have any effect.
        // While changing the phase of the sine wave could change how it sounds in
        // combination with another sound (beating) such precision is not the intend
        // of seeking
        Ok(())
    }
}
