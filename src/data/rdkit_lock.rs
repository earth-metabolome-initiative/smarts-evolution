use std::sync::Mutex;

/// Global mutex for all RDKit C++ FFI calls.
///
/// RDKit's C++ code uses global mutable state (ring perception caches,
/// SMILES/SMARTS parser internals, etc.) that segfaults when called
/// concurrently from rayon worker threads. All RDKit FFI calls must
/// be serialized through this lock.
pub static RDKIT_LOCK: Mutex<()> = Mutex::new(());

/// Serialize a closure across all RDKit FFI calls in this process.
#[inline]
pub fn with_rdkit_lock<T>(f: impl FnOnce() -> T) -> T {
    let _guard = RDKIT_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    f()
}
