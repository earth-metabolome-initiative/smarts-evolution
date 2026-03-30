const CPP_VERSION_FLAG: &str = "-std=c++17";

fn include_paths() -> Vec<String> {
    if let Ok(prefix) = std::env::var("CONDA_PREFIX") {
        return vec![
            format!("{prefix}/include"),
            format!("{prefix}/include/rdkit"),
        ];
    }

    match (std::env::consts::OS, std::env::consts::ARCH) {
        ("macos", "x86_64") => {
            vec![
                "/usr/local/include".into(),
                "/usr/local/include/rdkit".into(),
            ]
        }
        ("macos", "aarch64") => {
            vec![
                "/opt/homebrew/include".into(),
                "/opt/homebrew/include/rdkit".into(),
            ]
        }
        ("linux", _) => vec![
            "/usr/local/include".into(),
            "/usr/local/include/rdkit".into(),
            "/usr/include".into(),
            "/usr/include/rdkit".into(),
        ],
        (os, arch) => panic!("unsupported platform for RDKit shim: {os}/{arch}"),
    }
}

fn lib_paths() -> Vec<String> {
    if let Ok(prefix) = std::env::var("CONDA_PREFIX") {
        return vec![format!("{prefix}/lib")];
    }

    match (std::env::consts::OS, std::env::consts::ARCH) {
        ("macos", "aarch64") => vec!["/opt/homebrew/lib".into()],
        _ => Vec::new(),
    }
}

fn main() {
    if std::env::var("DOCS_RS").is_ok() {
        return;
    }

    println!("cargo:rerun-if-changed=src/bridge/substruct_library.rs");
    println!("cargo:rerun-if-changed=wrapper/include/substruct_library.h");
    println!("cargo:rerun-if-changed=wrapper/src/substruct_library.cc");

    cxx_build::bridge("src/bridge/substruct_library.rs")
        .file("wrapper/src/substruct_library.cc")
        .includes(include_paths())
        .include("wrapper/include")
        .include(std::env::var("CARGO_MANIFEST_DIR").unwrap())
        .flag(CPP_VERSION_FLAG)
        .warnings(false)
        .compile("smarts-evolution-substruct-library");

    for path in lib_paths() {
        println!("cargo:rustc-link-search=native={path}");
    }

    println!("cargo:rustc-link-lib=dylib=RDKitSubstructLibrary");
}
