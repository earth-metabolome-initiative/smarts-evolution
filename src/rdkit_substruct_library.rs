use cxx::UniquePtr;

use crate::bridge::substruct_library::ffi;
use crate::fitness::mcc::ConfusionCounts;

pub struct CompiledSmartsQuery {
    inner: UniquePtr<ffi::CompiledSmartsQuery>,
}

impl CompiledSmartsQuery {
    pub fn new(smarts: &str) -> Option<Self> {
        let inner = ffi::new_compiled_smarts_query(smarts);
        if inner.is_null() {
            None
        } else {
            Some(Self { inner })
        }
    }
}

pub struct SubstructLibraryIndex {
    inner: UniquePtr<ffi::BinaryClassificationLibrary>,
}

impl SubstructLibraryIndex {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let inner = ffi::new_binary_classification_library();
        if inner.is_null() {
            return Err("failed to allocate SubstructLibrary wrapper".into());
        }
        Ok(Self { inner })
    }

    pub fn add_smiles(
        &mut self,
        smiles: &str,
        positive: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let added = self.inner.pin_mut().add_smiles(smiles, positive);
        if added {
            Ok(())
        } else {
            Err(format!("failed to add SMILES to SubstructLibrary: {smiles}").into())
        }
    }

    pub fn count_matches(&self, smarts: &str, num_threads: i32) -> Option<ConfusionCounts> {
        let inner = self.inner.as_ref()?;
        let (mut tp, mut fp, mut tn, mut fn_) = (0u64, 0u64, 0u64, 0u64);
        if !inner.count_matches(smarts, num_threads, &mut tp, &mut fp, &mut tn, &mut fn_) {
            return None;
        }
        Some(ConfusionCounts { tp, fp, tn, fn_ })
    }

    pub fn count_matches_compiled(
        &self,
        query: &CompiledSmartsQuery,
        num_threads: i32,
    ) -> Option<ConfusionCounts> {
        let inner = self.inner.as_ref()?;
        let query = query.inner.as_ref()?;
        let (mut tp, mut fp, mut tn, mut fn_) = (0u64, 0u64, 0u64, 0u64);
        inner.count_matches_compiled(query, num_threads, &mut tp, &mut fp, &mut tn, &mut fn_);
        Some(ConfusionCounts { tp, fp, tn, fn_ })
    }

    pub fn positive_matches(&self, smarts: &str, num_threads: i32) -> Option<Vec<usize>> {
        let inner = self.inner.as_ref()?;
        Some(
            inner
                .positive_matches(smarts, num_threads)
                .into_iter()
                .map(|idx| idx as usize)
                .collect(),
        )
    }

    pub fn positive_size(&self) -> usize {
        self.inner
            .as_ref()
            .map(|inner| inner.positive_size())
            .unwrap_or(0)
    }

    pub fn negative_size(&self) -> usize {
        self.inner
            .as_ref()
            .map(|inner| inner.negative_size())
            .unwrap_or(0)
    }
}
