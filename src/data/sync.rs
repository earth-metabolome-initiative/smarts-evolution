use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use flate2::read::GzDecoder;
use log::{info, warn};
use reqwest::Url;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tokio::runtime::Builder;
use uuid::Uuid;
use zenodo_rs::{ArtifactSelector, Auth, Endpoint, ZenodoClient};

use super::io::open_jsonl_reader;

type DynError = Box<dyn std::error::Error>;

const NPC_CONCEPT_DOI: &str = "10.5281/zenodo.14040990";
const NPC_ARTIFACT_KEY: &str = "completed.jsonl.zst";
const NPC_PREPARED_FILENAME: &str = "npc.fully_labeled.jsonl.zst";
const NPC_RAW_RELATIVE_PATH: &str = ".tmp/datasets/raw/npc/completed.jsonl.zst";
const NPC_STATE_RELATIVE_PATH: &str = ".tmp/datasets/state/npc.json";

const CLASSYFIRE_CONCEPT_DOI: &str = "10.5281/zenodo.19235916";
const CLASSYFIRE_ARTIFACT_KEY: &str = "classyfire-labels.jsonl.zst";
const CLASSYFIRE_PREPARED_FILENAME: &str = "classyfire.jsonl.zst";
const CLASSYFIRE_RAW_RELATIVE_PATH: &str =
    ".tmp/datasets/raw/classyfire/classyfire-labels.jsonl.zst";
const CLASSYFIRE_STATE_RELATIVE_PATH: &str = ".tmp/datasets/state/classyfire.json";

const PUBCHEM_CID_SMILES_URL: &str =
    "https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-SMILES.gz";
const PUBCHEM_CID_SMILES_RELATIVE_PATH: &str = ".tmp/datasets/raw/pubchem/CID-SMILES.gz";

const DATA_ROOT_ENV_VAR: &str = "SMARTS_EVOLUTION_DATA_ROOT";
const ZENODO_ENDPOINT_ENV_VAR: &str = "SMARTS_EVOLUTION_ZENODO_ENDPOINT";
const NPC_CONCEPT_DOI_ENV_VAR: &str = "SMARTS_EVOLUTION_NPC_CONCEPT_DOI";
const CLASSYFIRE_CONCEPT_DOI_ENV_VAR: &str = "SMARTS_EVOLUTION_CLASSYFIRE_CONCEPT_DOI";
const PUBCHEM_CID_SMILES_URL_ENV_VAR: &str = "SMARTS_EVOLUTION_PUBCHEM_CID_SMILES_URL";

#[derive(Debug, Default, Deserialize, Serialize)]
struct SyncState {
    #[serde(default)]
    source_record_id: Option<u64>,
    #[serde(default)]
    updated_at_epoch_seconds: Option<u64>,
}

#[derive(Deserialize)]
struct NpcReleaseRecord {
    smiles: String,
    #[serde(default)]
    pathway_results: Vec<String>,
    #[serde(default)]
    superclass_results: Vec<String>,
    #[serde(default)]
    class_results: Vec<String>,
}

#[derive(Deserialize)]
struct ClassyFireReleaseRecord {
    cid: u64,
    classyfire: Map<String, Value>,
}

#[derive(Serialize)]
struct PreparedClassyFireRecord {
    cid: u64,
    classyfire: Map<String, Value>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct PubChemSmilesRecord {
    cid: u64,
    smiles: String,
}

pub fn ensure_latest_npc_dataset() -> Result<PathBuf, DynError> {
    let data_root = data_root();
    let concept_doi = npc_concept_doi();
    let prepared_path = data_root.join(NPC_PREPARED_FILENAME);
    let raw_path = data_root.join(NPC_RAW_RELATIVE_PATH);
    let state_path = data_root.join(NPC_STATE_RELATIVE_PATH);
    let state = load_state(&state_path);

    let latest_record_id = match resolve_latest_record_id(&concept_doi) {
        Ok(record_id) => record_id,
        Err(error) => {
            return use_cached_prepared_dataset(
                "NPC",
                &prepared_path,
                format!("failed to resolve the latest Zenodo release: {error}"),
            );
        }
    };

    if prepared_path.exists() && state.source_record_id == Some(latest_record_id) {
        info!(
            "NPC dataset already matches Zenodo record {} at {}",
            latest_record_id,
            prepared_path.display()
        );
        return Ok(prepared_path);
    }

    if !raw_path.exists() || state.source_record_id != Some(latest_record_id) {
        if let Err(error) = download_latest_artifact(&concept_doi, NPC_ARTIFACT_KEY, &raw_path) {
            return use_cached_prepared_dataset(
                "NPC",
                &prepared_path,
                format!("failed to download the latest Zenodo artifact: {error}"),
            );
        }
    }

    if let Err(error) = build_npc_fully_labeled_dataset(&raw_path, &prepared_path) {
        return use_cached_prepared_dataset(
            "NPC",
            &prepared_path,
            format!("failed to rebuild the prepared dataset: {error}"),
        );
    }

    save_state(
        &state_path,
        &SyncState {
            source_record_id: Some(latest_record_id),
            updated_at_epoch_seconds: Some(current_unix_timestamp()?),
        },
    );

    Ok(prepared_path)
}

pub fn ensure_latest_classyfire_dataset() -> Result<PathBuf, DynError> {
    let data_root = data_root();
    let concept_doi = classyfire_concept_doi();
    let pubchem_url = pubchem_cid_smiles_url();
    let prepared_path = data_root.join(CLASSYFIRE_PREPARED_FILENAME);
    let raw_path = data_root.join(CLASSYFIRE_RAW_RELATIVE_PATH);
    let state_path = data_root.join(CLASSYFIRE_STATE_RELATIVE_PATH);
    let pubchem_path = data_root.join(PUBCHEM_CID_SMILES_RELATIVE_PATH);
    let state = load_state(&state_path);

    let latest_record_id = match resolve_latest_record_id(&concept_doi) {
        Ok(record_id) => record_id,
        Err(error) => {
            return use_cached_prepared_dataset(
                "ClassyFire",
                &prepared_path,
                format!("failed to resolve the latest Zenodo release: {error}"),
            );
        }
    };

    if prepared_path.exists() && state.source_record_id == Some(latest_record_id) {
        info!(
            "ClassyFire dataset already matches Zenodo record {} at {}",
            latest_record_id,
            prepared_path.display()
        );
        return Ok(prepared_path);
    }

    if !raw_path.exists() || state.source_record_id != Some(latest_record_id) {
        if let Err(error) =
            download_latest_artifact(&concept_doi, CLASSYFIRE_ARTIFACT_KEY, &raw_path)
        {
            return use_cached_prepared_dataset(
                "ClassyFire",
                &prepared_path,
                format!("failed to download the latest Zenodo artifact: {error}"),
            );
        }
    }

    if let Err(error) = ensure_pubchem_cid_smiles(&pubchem_url, &pubchem_path) {
        return use_cached_prepared_dataset(
            "ClassyFire",
            &prepared_path,
            format!("failed to download PubChem CID-SMILES.gz: {error}"),
        );
    }

    if let Err(error) =
        build_classyfire_labeled_smiles_dataset(&raw_path, &pubchem_path, &prepared_path)
    {
        return use_cached_prepared_dataset(
            "ClassyFire",
            &prepared_path,
            format!("failed to rebuild the prepared dataset: {error}"),
        );
    }

    save_state(
        &state_path,
        &SyncState {
            source_record_id: Some(latest_record_id),
            updated_at_epoch_seconds: Some(current_unix_timestamp()?),
        },
    );

    Ok(prepared_path)
}

fn resolve_latest_record_id(concept_doi: &str) -> Result<u64, DynError> {
    let client = zenodo_client()?;
    let runtime = zenodo_runtime()?;
    let record = runtime.block_on(client.resolve_latest_by_doi_str(concept_doi))?;
    Ok(record.id.0)
}

fn download_latest_artifact(
    concept_doi: &str,
    artifact_key: &str,
    destination: &Path,
) -> Result<(), DynError> {
    ensure_parent_dir(destination)?;

    let client = zenodo_client()?;
    let runtime = zenodo_runtime()?;
    let selector = ArtifactSelector::latest_file_by_doi(concept_doi, artifact_key)?;
    let resolved = runtime.block_on(client.download_artifact(&selector, destination))?;

    info!(
        "Downloaded Zenodo artifact {} from record {} to {} ({} bytes)",
        artifact_key,
        resolved.resolved_record.0,
        destination.display(),
        resolved.bytes_written
    );
    Ok(())
}

fn ensure_pubchem_cid_smiles(pubchem_cid_smiles_url: &str, path: &Path) -> Result<(), DynError> {
    if path.exists() {
        return Ok(());
    }

    ensure_parent_dir(path)?;
    let temp_path = temporary_path(path)?;
    let mut file = BufWriter::new(File::create(&temp_path)?);
    let client = reqwest::blocking::Client::builder()
        .user_agent(zenodo_user_agent())
        .timeout(Duration::from_secs(600))
        .build()?;
    let mut response = client
        .get(pubchem_cid_smiles_url)
        .send()?
        .error_for_status()?;

    io::copy(&mut response, &mut file)?;
    file.flush()?;
    file.get_ref().sync_all()?;
    fs::rename(&temp_path, path)?;

    let bytes = fs::metadata(path)?.len();
    info!(
        "Downloaded PubChem CID-SMILES.gz to {} ({} bytes)",
        path.display(),
        bytes
    );
    Ok(())
}

fn build_npc_fully_labeled_dataset(raw_path: &Path, prepared_path: &Path) -> Result<(), DynError> {
    ensure_parent_dir(prepared_path)?;

    let reader = open_jsonl_reader(raw_path)?;
    let temp_path = temporary_path(prepared_path)?;
    let file = File::create(&temp_path)?;
    let mut encoder = zstd::Encoder::new(file, 3)?;
    let mut kept = 0usize;
    let mut skipped = 0usize;

    for (line_no, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        let record: NpcReleaseRecord = serde_json::from_str(&line).map_err(|error| {
            io::Error::other(format!(
                "failed to parse NPC release line {} from {}: {error}",
                line_no + 1,
                raw_path.display()
            ))
        })?;

        let is_fully_labeled = !record.smiles.is_empty()
            && !record.pathway_results.is_empty()
            && !record.superclass_results.is_empty()
            && !record.class_results.is_empty();

        if is_fully_labeled {
            encoder.write_all(line.as_bytes())?;
            encoder.write_all(b"\n")?;
            kept += 1;
        } else {
            skipped += 1;
        }
    }

    if kept == 0 {
        return Err(io::Error::other(format!(
            "no fully labeled NPC compounds were found in {}",
            raw_path.display()
        ))
        .into());
    }

    let finished = encoder.finish()?;
    finished.sync_all()?;
    drop(finished);
    fs::rename(&temp_path, prepared_path)?;

    info!(
        "Prepared NPC dataset at {} with {} fully labeled compounds ({} skipped)",
        prepared_path.display(),
        kept,
        skipped
    );
    Ok(())
}

fn build_classyfire_labeled_smiles_dataset(
    raw_path: &Path,
    pubchem_path: &Path,
    prepared_path: &Path,
) -> Result<(), DynError> {
    ensure_parent_dir(prepared_path)?;

    let reader = open_jsonl_reader(raw_path)?;
    let pubchem_file = File::open(pubchem_path)?;
    let pubchem_decoder = GzDecoder::new(pubchem_file);
    let mut pubchem_lines = BufReader::new(pubchem_decoder).lines();
    let mut current_pubchem = next_pubchem_smiles_record(&mut pubchem_lines)?;
    let temp_path = temporary_path(prepared_path)?;
    let file = File::create(&temp_path)?;
    let mut encoder = zstd::Encoder::new(file, 3)?;
    let mut seen_rows = 0usize;
    let mut last_cid = 0u64;
    let mut written = 0usize;
    let mut skipped_missing_smiles = 0usize;

    for (line_no, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        let mut record: ClassyFireReleaseRecord = serde_json::from_str(&line).map_err(|error| {
            io::Error::other(format!(
                "failed to parse ClassyFire release line {} from {}: {error}",
                line_no + 1,
                raw_path.display()
            ))
        })?;

        if seen_rows > 0 && record.cid < last_cid {
            return Err(io::Error::other(format!(
                "ClassyFire release rows are not sorted by CID in {}",
                raw_path.display()
            ))
            .into());
        }
        seen_rows += 1;
        last_cid = record.cid;

        while matches!(&current_pubchem, Some(entry) if entry.cid < record.cid) {
            current_pubchem = next_pubchem_smiles_record(&mut pubchem_lines)?;
        }

        match &current_pubchem {
            Some(entry) if entry.cid == record.cid && !entry.smiles.is_empty() => {
                record
                    .classyfire
                    .insert("smiles".to_owned(), Value::String(entry.smiles.clone()));
                serde_json::to_writer(
                    &mut encoder,
                    &PreparedClassyFireRecord {
                        cid: record.cid,
                        classyfire: record.classyfire,
                    },
                )?;
                encoder.write_all(b"\n")?;
                written += 1;
            }
            _ => {
                skipped_missing_smiles += 1;
            }
        }
    }

    if written == 0 {
        return Err(io::Error::other(format!(
            "no ClassyFire rows could be joined against PubChem CID-SMILES.gz from {}",
            pubchem_path.display()
        ))
        .into());
    }

    let finished = encoder.finish()?;
    finished.sync_all()?;
    drop(finished);
    fs::rename(&temp_path, prepared_path)?;

    info!(
        "Prepared ClassyFire dataset at {} with {} labeled SMILES ({} missing in PubChem CID-SMILES.gz)",
        prepared_path.display(),
        written,
        skipped_missing_smiles
    );
    Ok(())
}

fn next_pubchem_smiles_record(
    lines: &mut impl Iterator<Item = io::Result<String>>,
) -> Result<Option<PubChemSmilesRecord>, DynError> {
    match lines.next() {
        Some(line) => Ok(Some(parse_pubchem_smiles_line(&line?)?)),
        None => Ok(None),
    }
}

fn parse_pubchem_smiles_line(line: &str) -> Result<PubChemSmilesRecord, DynError> {
    let (cid, smiles) = line.split_once('\t').ok_or_else(|| {
        io::Error::other(format!(
            "invalid PubChem CID-SMILES line without tab separator: {line}"
        ))
    })?;

    Ok(PubChemSmilesRecord {
        cid: cid
            .trim()
            .parse()
            .map_err(|error| io::Error::other(format!("invalid PubChem CID `{cid}`: {error}")))?,
        smiles: smiles.trim().to_owned(),
    })
}

fn zenodo_client() -> Result<ZenodoClient, DynError> {
    let token = std::env::var(Auth::TOKEN_ENV_VAR).unwrap_or_default();
    let mut builder = ZenodoClient::builder(Auth::new(token))
        .user_agent(zenodo_user_agent())
        .connect_timeout(Duration::from_secs(15))
        .request_timeout(Duration::from_secs(300));
    if let Some(endpoint) = zenodo_endpoint_override()? {
        builder = builder.endpoint(endpoint);
    }
    Ok(builder.build()?)
}

fn zenodo_runtime() -> Result<tokio::runtime::Runtime, DynError> {
    Ok(Builder::new_current_thread().enable_all().build()?)
}

fn zenodo_user_agent() -> String {
    format!("{}/{}", env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION"))
}

fn load_state(path: &Path) -> SyncState {
    match fs::read(path) {
        Ok(bytes) => match serde_json::from_slice::<SyncState>(&bytes) {
            Ok(state) => state,
            Err(error) => {
                warn!(
                    "Ignoring unreadable sync state at {}: {}",
                    path.display(),
                    error
                );
                SyncState::default()
            }
        },
        Err(error) if error.kind() == io::ErrorKind::NotFound => SyncState::default(),
        Err(error) => {
            warn!("Ignoring sync state at {}: {}", path.display(), error);
            SyncState::default()
        }
    }
}

fn save_state(path: &Path, state: &SyncState) {
    if let Err(error) = save_state_inner(path, state) {
        warn!(
            "Failed to persist sync state at {}: {}",
            path.display(),
            error
        );
    }
}

fn save_state_inner(path: &Path, state: &SyncState) -> Result<(), DynError> {
    ensure_parent_dir(path)?;
    let temp_path = temporary_path(path)?;
    let bytes = serde_json::to_vec_pretty(state)?;
    fs::write(&temp_path, bytes)?;
    fs::rename(&temp_path, path)?;
    Ok(())
}

fn current_unix_timestamp() -> Result<u64, DynError> {
    Ok(SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs())
}

fn ensure_parent_dir(path: &Path) -> Result<(), DynError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    Ok(())
}

fn temporary_path(path: &Path) -> Result<PathBuf, DynError> {
    let parent = path
        .parent()
        .ok_or_else(|| io::Error::other(format!("missing parent for {}", path.display())))?;
    let filename = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| io::Error::other(format!("missing file name for {}", path.display())))?;
    Ok(parent.join(format!(".{filename}.tmp-{}", Uuid::new_v4())))
}

fn use_cached_prepared_dataset(
    dataset_name: &str,
    prepared_path: &Path,
    reason: String,
) -> Result<PathBuf, DynError> {
    if prepared_path.exists() {
        warn!(
            "{} dataset sync failed ({}). Falling back to cached dataset at {}",
            dataset_name,
            reason,
            prepared_path.display()
        );
        Ok(prepared_path.to_path_buf())
    } else {
        Err(io::Error::other(format!(
            "{dataset_name} dataset sync failed and no cached dataset is available: {reason}"
        ))
        .into())
    }
}

fn data_root() -> PathBuf {
    override_env(DATA_ROOT_ENV_VAR)
        .map(PathBuf::from)
        .unwrap_or_else(repo_root)
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn npc_concept_doi() -> String {
    override_env(NPC_CONCEPT_DOI_ENV_VAR).unwrap_or_else(|| NPC_CONCEPT_DOI.to_owned())
}

fn classyfire_concept_doi() -> String {
    override_env(CLASSYFIRE_CONCEPT_DOI_ENV_VAR)
        .unwrap_or_else(|| CLASSYFIRE_CONCEPT_DOI.to_owned())
}

fn pubchem_cid_smiles_url() -> String {
    override_env(PUBCHEM_CID_SMILES_URL_ENV_VAR)
        .unwrap_or_else(|| PUBCHEM_CID_SMILES_URL.to_owned())
}

fn zenodo_endpoint_override() -> Result<Option<Endpoint>, DynError> {
    override_env(ZENODO_ENDPOINT_ENV_VAR)
        .map(|value| Ok(Endpoint::Custom(Url::parse(&value)?)))
        .transpose()
}

fn override_env(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .filter(|value| !value.trim().is_empty())
}

#[cfg(test)]
mod tests {
    use std::fs::{self, File};
    use std::io::{BufRead, BufReader, Write};
    use std::path::Path;
    use std::time::{SystemTime, UNIX_EPOCH};

    use flate2::Compression;
    use flate2::write::GzEncoder;

    use super::{build_classyfire_labeled_smiles_dataset, build_npc_fully_labeled_dataset};

    fn unique_temp_path(label: &str, extension: &str) -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "smarts-evolution-sync-{label}-{}-{nanos}.{extension}",
            std::process::id()
        ))
    }

    fn write_zstd_lines(path: &Path, lines: &[&str]) {
        let file = File::create(path).unwrap();
        let mut encoder = zstd::Encoder::new(file, 0).unwrap();
        for line in lines {
            writeln!(encoder, "{line}").unwrap();
        }
        encoder.finish().unwrap();
    }

    fn read_zstd_lines(path: &Path) -> Vec<String> {
        let file = File::open(path).unwrap();
        let decoder = zstd::Decoder::new(file).unwrap();
        BufReader::new(decoder)
            .lines()
            .collect::<Result<Vec<_>, _>>()
            .unwrap()
    }

    #[test]
    fn npc_builder_keeps_only_fully_labeled_rows() {
        let raw_path = unique_temp_path("npc-raw", "jsonl.zst");
        let prepared_path = unique_temp_path("npc-prepared", "jsonl.zst");

        write_zstd_lines(
            &raw_path,
            &[
                r#"{"cid":1,"smiles":"CC","pathway_results":["PathA"],"superclass_results":["SuperA"],"class_results":["ClassA"]}"#,
                r#"{"cid":2,"smiles":"N","pathway_results":["PathB"],"superclass_results":[],"class_results":["ClassB"]}"#,
            ],
        );

        build_npc_fully_labeled_dataset(&raw_path, &prepared_path).unwrap();

        let lines = read_zstd_lines(&prepared_path);
        assert_eq!(lines.len(), 1);
        assert!(lines[0].contains(r#""cid":1"#));

        fs::remove_file(raw_path).unwrap();
        fs::remove_file(prepared_path).unwrap();
    }

    #[test]
    fn classyfire_builder_joins_smiles_from_pubchem() {
        let raw_path = unique_temp_path("classyfire-raw", "jsonl.zst");
        let pubchem_path = unique_temp_path("cid-smiles", "gz");
        let prepared_path = unique_temp_path("classyfire-prepared", "jsonl.zst");

        write_zstd_lines(
            &raw_path,
            &[
                r#"{"cid":1,"classyfire":{"kingdom":{"name":"Organic compounds"},"direct_parent":{"name":"Alkanes"}}}"#,
                r#"{"cid":3,"classyfire":{"direct_parent":{"name":"Oxides"}}}"#,
            ],
        );

        let pubchem_file = File::create(&pubchem_path).unwrap();
        let mut encoder = GzEncoder::new(pubchem_file, Compression::default());
        writeln!(encoder, "1\tCC").unwrap();
        writeln!(encoder, "2\tN").unwrap();
        encoder.finish().unwrap();

        build_classyfire_labeled_smiles_dataset(&raw_path, &pubchem_path, &prepared_path).unwrap();

        let lines = read_zstd_lines(&prepared_path);
        assert_eq!(lines.len(), 1);
        assert!(lines[0].contains(r#""cid":1"#));
        assert!(lines[0].contains(r#""smiles":"CC""#));
        assert!(lines[0].contains(r#""direct_parent":{"name":"Alkanes"}"#));

        fs::remove_file(raw_path).unwrap();
        fs::remove_file(pubchem_path).unwrap();
        fs::remove_file(prepared_path).unwrap();
    }

    #[test]
    fn classyfire_builder_rejects_unsorted_release_rows() {
        let raw_path = unique_temp_path("classyfire-unsorted-raw", "jsonl.zst");
        let pubchem_path = unique_temp_path("classyfire-unsorted-pubchem", "gz");
        let prepared_path = unique_temp_path("classyfire-unsorted-prepared", "jsonl.zst");

        write_zstd_lines(
            &raw_path,
            &[
                r#"{"cid":2,"classyfire":{"direct_parent":{"name":"B"}}}"#,
                r#"{"cid":1,"classyfire":{"direct_parent":{"name":"A"}}}"#,
            ],
        );

        let pubchem_file = File::create(&pubchem_path).unwrap();
        let mut encoder = GzEncoder::new(pubchem_file, Compression::default());
        writeln!(encoder, "1\tCC").unwrap();
        writeln!(encoder, "2\tNN").unwrap();
        encoder.finish().unwrap();

        let error =
            build_classyfire_labeled_smiles_dataset(&raw_path, &pubchem_path, &prepared_path)
                .unwrap_err()
                .to_string();
        assert!(error.contains("not sorted by CID"));

        fs::remove_file(raw_path).unwrap();
        fs::remove_file(pubchem_path).unwrap();
    }
}
