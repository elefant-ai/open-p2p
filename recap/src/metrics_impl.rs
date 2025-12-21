use hashbrown::HashMap;
use indexmap::IndexMap;
use std::sync::{Arc, atomic::Ordering};

use metrics::{
    Counter, Gauge, Histogram, Key, KeyName, Label, Metadata, Recorder, SharedString, Unit,
};
use metrics_util::{
    MetricKindMask,
    registry::{GenerationalAtomicStorage, Recency, Registry},
};
use parking_lot::RwLock;

#[derive(Debug)]
pub struct MetricsRecorder {
    inner: Arc<Inner>,
}

#[derive(Debug)]
struct Inner {
    pub recency: Recency<Key>,
    pub registry: Registry<Key, GenerationalAtomicStorage>,
    pub descriptions: RwLock<HashMap<String, (SharedString, Option<Unit>)>>,
}

impl MetricsRecorder {
    fn add_description_if_missing(
        &self,
        key_name: &KeyName,
        description: SharedString,
        unit: Option<Unit>,
    ) {
        let mut descriptions = self.inner.descriptions.write();
        descriptions
            .entry(key_name.as_str().to_owned())
            .or_insert((description, unit));
    }
}

impl Recorder for MetricsRecorder {
    fn describe_counter(&self, key_name: KeyName, unit: Option<Unit>, description: SharedString) {
        self.add_description_if_missing(&key_name, description, unit);
    }

    fn describe_gauge(&self, key_name: KeyName, unit: Option<Unit>, description: SharedString) {
        self.add_description_if_missing(&key_name, description, unit);
    }

    fn describe_histogram(&self, key_name: KeyName, unit: Option<Unit>, description: SharedString) {
        self.add_description_if_missing(&key_name, description, unit);
    }

    fn register_counter(&self, key: &Key, _metadata: &Metadata<'_>) -> Counter {
        self.inner
            .registry
            .get_or_create_counter(key, |c| c.clone().into())
    }

    fn register_gauge(&self, key: &Key, _metadata: &Metadata<'_>) -> Gauge {
        self.inner
            .registry
            .get_or_create_gauge(key, |c| c.clone().into())
    }

    fn register_histogram(&self, key: &Key, _metadata: &Metadata<'_>) -> Histogram {
        self.inner
            .registry
            .get_or_create_histogram(key, |c| c.clone().into())
    }
}

pub fn init_metrics() -> ExternalHandle {
    let inner = Arc::new(Inner {
        registry: Registry::new(GenerationalAtomicStorage::atomic()),
        descriptions: RwLock::new(HashMap::new()),
        recency: Recency::new(quanta::Clock::new(), MetricKindMask::NONE, None),
    });
    metrics::set_global_recorder(MetricsRecorder {
        inner: inner.clone(),
    })
    .unwrap();

    ExternalHandle { inner }
}

#[derive(Debug)]
pub struct ExternalHandle {
    inner: Arc<Inner>,
}

impl ExternalHandle {
    pub fn snapshot(&self) -> Snapshot {
        let mut counters = HashMap::new();
        let counter_handles = self.inner.registry.get_counter_handles();
        for (key, counter) in counter_handles {
            let gene = counter.get_generation();
            if !self
                .inner
                .recency
                .should_store_counter(&key, gene, &self.inner.registry)
            {
                continue;
            }

            let (name, labels) = key.into_parts();
            let value = counter.get_inner().load(Ordering::Acquire);
            let entry = counters
                .entry(name)
                .or_insert_with(HashMap::new)
                .entry(labels)
                .or_insert(0);
            *entry = value;
        }

        let mut gauges = HashMap::new();
        let gauge_handles = self.inner.registry.get_gauge_handles();
        for (key, gauge) in gauge_handles {
            let gene = gauge.get_generation();
            if !self
                .inner
                .recency
                .should_store_gauge(&key, gene, &self.inner.registry)
            {
                continue;
            }

            let (name, labels) = key.into_parts();
            let value = f64::from_bits(gauge.get_inner().load(Ordering::Acquire));
            let entry = gauges
                .entry(name)
                .or_insert_with(HashMap::new)
                .entry(labels)
                .or_insert(0.0);
            *entry = value;
        }

        let mut histograms = HashMap::new();
        let histogram_handles = self.inner.registry.get_histogram_handles();
        for (key, histogram) in histogram_handles {
            let gene = histogram.get_generation();
            if !self
                .inner
                .recency
                .should_store_histogram(&key, gene, &self.inner.registry)
            {
                continue;
            }

            let (name, labels) = key.into_parts();
            let entry = histograms
                .entry(name)
                .or_insert_with(IndexMap::new)
                .entry(labels)
                .or_insert_with(Vec::new);

            histogram.get_inner().clear_with(|samples| {
                entry.extend_from_slice(samples);
            });
        }

        Snapshot {
            counters,
            gauges,
            histograms,
        }
    }
}

#[derive(Debug)]
pub struct Snapshot {
    pub counters: HashMap<KeyName, HashMap<Vec<Label>, u64>>,
    pub gauges: HashMap<KeyName, HashMap<Vec<Label>, f64>>,
    pub histograms: HashMap<KeyName, IndexMap<Vec<Label>, Vec<f64>>>,
}

impl Snapshot {
    pub fn merge(&mut self, other: Snapshot) {
        for (name, counters) in other.counters {
            let entry = self.counters.entry(name).or_insert_with(HashMap::new);
            counters.into_iter().for_each(|(k, v)| {
                entry
                    .entry(k)
                    .and_modify(|existing| *existing = v)
                    .or_insert(v);
            });
        }

        for (name, gauges) in other.gauges {
            let entry = self.gauges.entry(name).or_insert_with(HashMap::new);
            gauges.into_iter().for_each(|(k, v)| {
                entry
                    .entry(k)
                    .and_modify(|existing| *existing = v)
                    .or_insert(v);
            });
        }

        for (name, histograms) in other.histograms {
            let entry = self.histograms.entry(name).or_insert_with(IndexMap::new);
            histograms.into_iter().for_each(|(k, v)| {
                entry
                    .entry(k)
                    .and_modify(|existing| existing.extend(v.clone()))
                    .or_insert(v);
            });
        }
    }

    pub fn view_counter(&self, name: &str, labels: &[Label]) -> Option<u64> {
        self.counters
            .get(name)
            .and_then(|labels_map| labels_map.get(labels).cloned())
    }

    pub fn view_gauge(&self, name: &str, labels: &[Label]) -> Option<f64> {
        self.gauges
            .get(name)
            .and_then(|labels_map| labels_map.get(labels).cloned())
    }

    pub fn view_histogram(&self, name: &str, labels: &[Label]) -> Option<&[f64]> {
        self.histograms
            .get(name)
            .and_then(|labels_map| labels_map.get(labels).map(std::vec::Vec::as_slice))
    }
}
