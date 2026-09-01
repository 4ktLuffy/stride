use std::sync::atomic::{AtomicU64, Ordering};

macro_rules! counter_id {
    ($name:ident, $counter:ident, $doc:literal) => {
        #[doc = $doc]
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(pub u64);

        static $counter: AtomicU64 = AtomicU64::new(1);

        impl $name {
            /// Allocate the next id in this process.
            pub fn next() -> Self {
                Self($counter.fetch_add(1, Ordering::Relaxed))
            }

            pub fn raw(self) -> u64 {
                self.0
            }
        }

        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{}-{}", stringify!($name), self.0)
            }
        }
    };
}

counter_id!(RequestId, REQUEST_COUNTER, "Identifies one client request.");
counter_id!(
    SequenceId,
    SEQUENCE_COUNTER,
    "Identifies one generation stream. A request with `n > 1` owns several."
);
