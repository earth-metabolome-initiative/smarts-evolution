use alloc::boxed::Box;
use alloc::format;
use std::error::Error;
use std::fmt;
use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

pub use log::LevelFilter;

/// Initialize a file-only logger with the default [`FileLogConfig`] settings.
///
/// The default level is [`LevelFilter::Warn`], which records pathologically
/// slow SMARTS evaluations without logging every normal evaluation. Use
/// [`FileLogConfig`] directly when debug-level per-SMARTS logs are useful.
pub fn init_file_logger(path: impl AsRef<Path>) -> Result<(), FileLogInitError> {
    FileLogConfig::new(path.as_ref().to_path_buf()).init()
}

/// Configuration for the built-in file logger.
///
/// This logger is intentionally file-only by default so it does not interfere
/// with terminal progress bars. Set [`Self::mirror_to_stderr`] when stderr
/// output is desired as well.
#[derive(Clone, Debug)]
pub struct FileLogConfig {
    path: PathBuf,
    level: LevelFilter,
    append: bool,
    mirror_to_stderr: bool,
}

impl FileLogConfig {
    /// Create a file logger configuration.
    ///
    /// Defaults:
    ///
    /// - `level`: [`LevelFilter::Warn`]
    /// - `append`: `true`
    /// - `mirror_to_stderr`: `false`
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            level: LevelFilter::Warn,
            append: true,
            mirror_to_stderr: false,
        }
    }

    /// Set the maximum log level written to the file.
    pub fn level(mut self, level: LevelFilter) -> Self {
        self.level = level;
        self
    }

    /// Control whether log records are appended or the file is truncated at initialization.
    pub fn append(mut self, append: bool) -> Self {
        self.append = append;
        self
    }

    /// Mirror each file log record to stderr.
    ///
    /// Leave this disabled when indicatif progress bars own terminal rendering.
    pub fn mirror_to_stderr(mut self, mirror: bool) -> Self {
        self.mirror_to_stderr = mirror;
        self
    }

    /// Install this file logger as the global logger.
    ///
    /// Rust programs can install only one global `log` backend. If the host
    /// application already initialized logging, this returns
    /// [`FileLogInitError::SetLogger`]; in that case, configure the host
    /// logger to also write `smarts-evolution` records to a file.
    pub fn init(self) -> Result<(), FileLogInitError> {
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .append(self.append)
            .truncate(!self.append)
            .open(&self.path)?;

        log::set_boxed_logger(Box::new(FileLogger {
            file: Mutex::new(file),
            level: self.level,
            mirror_to_stderr: self.mirror_to_stderr,
        }))?;
        log::set_max_level(self.level);
        Ok(())
    }
}

/// Error returned when the built-in file logger cannot be initialized.
#[derive(Debug)]
pub enum FileLogInitError {
    /// The log file could not be opened.
    Io(io::Error),
    /// A global logger has already been installed.
    SetLogger(log::SetLoggerError),
}

impl fmt::Display for FileLogInitError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "could not open log file: {error}"),
            Self::SetLogger(error) => write!(formatter, "could not initialize logger: {error}"),
        }
    }
}

impl Error for FileLogInitError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::SetLogger(error) => Some(error),
        }
    }
}

impl From<io::Error> for FileLogInitError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<log::SetLoggerError> for FileLogInitError {
    fn from(error: log::SetLoggerError) -> Self {
        Self::SetLogger(error)
    }
}

struct FileLogger {
    file: Mutex<File>,
    level: LevelFilter,
    mirror_to_stderr: bool,
}

impl log::Log for FileLogger {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        metadata.level() <= self.level
    }

    fn log(&self, record: &log::Record<'_>) {
        if !self.enabled(record.metadata()) {
            return;
        }

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default();
        let line = format!(
            "{}.{:03} {} {} - {}\n",
            timestamp.as_secs(),
            timestamp.subsec_millis(),
            record.level(),
            record.target(),
            record.args(),
        );

        if let Ok(mut file) = self.file.lock() {
            let _ = file.write_all(line.as_bytes());
        }

        if self.mirror_to_stderr {
            let mut stderr = io::stderr().lock();
            let _ = stderr.write_all(line.as_bytes());
        }
    }

    fn flush(&self) {
        if let Ok(mut file) = self.file.lock() {
            let _ = file.flush();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use log::Log;
    use std::fs;

    #[test]
    fn file_log_config_defaults_to_warning_file_only_append() {
        let config = FileLogConfig::new("smarts-evolution.log");

        assert_eq!(config.path, PathBuf::from("smarts-evolution.log"));
        assert_eq!(config.level, LevelFilter::Warn);
        assert!(config.append);
        assert!(!config.mirror_to_stderr);
    }

    #[test]
    fn file_log_config_builder_sets_options() {
        let config = FileLogConfig::new("debug.log")
            .level(LevelFilter::Debug)
            .append(false)
            .mirror_to_stderr(true);

        assert_eq!(config.level, LevelFilter::Debug);
        assert!(!config.append);
        assert!(config.mirror_to_stderr);
    }

    #[test]
    fn file_logger_writes_enabled_records() {
        let path = std::env::temp_dir().join(format!(
            "smarts-evolution-logger-{}.log",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let file = File::create(&path).unwrap();
        let logger = FileLogger {
            file: Mutex::new(file),
            level: LevelFilter::Warn,
            mirror_to_stderr: false,
        };
        let record = log::Record::builder()
            .args(format_args!("slow SMARTS evaluation"))
            .level(log::Level::Warn)
            .target("smarts_evolution::fitness::evaluator")
            .build();

        logger.log(&record);
        logger.flush();

        let logged = fs::read_to_string(&path).unwrap();
        fs::remove_file(path).unwrap();
        assert!(
            logged.contains(" WARN smarts_evolution::fitness::evaluator - slow SMARTS evaluation")
        );
    }
}
