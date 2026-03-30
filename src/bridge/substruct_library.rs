#[cxx::bridge(namespace = "smarts_evolution")]
pub mod ffi {
    unsafe extern "C++" {
        include!("substruct_library.h");

        pub type BinaryClassificationLibrary;
        pub type CompiledSmartsQuery;

        pub fn new_binary_classification_library() -> UniquePtr<BinaryClassificationLibrary>;
        pub fn new_compiled_smarts_query(smarts: &str) -> UniquePtr<CompiledSmartsQuery>;
        pub fn add_smiles(
            self: Pin<&mut BinaryClassificationLibrary>,
            smiles: &str,
            positive: bool,
        ) -> bool;
        pub fn count_matches(
            self: &BinaryClassificationLibrary,
            smarts: &str,
            num_threads: i32,
            tp: &mut u64,
            fp: &mut u64,
            tn: &mut u64,
            fn_: &mut u64,
        ) -> bool;
        pub fn count_matches_compiled(
            self: &BinaryClassificationLibrary,
            query: &CompiledSmartsQuery,
            num_threads: i32,
            tp: &mut u64,
            fp: &mut u64,
            tn: &mut u64,
            fn_: &mut u64,
        );
        pub fn positive_matches(
            self: &BinaryClassificationLibrary,
            smarts: &str,
            num_threads: i32,
        ) -> Vec<u32>;
        pub fn positive_size(self: &BinaryClassificationLibrary) -> usize;
        pub fn negative_size(self: &BinaryClassificationLibrary) -> usize;
    }
}
