//! Error types for the world crate.

/// Errors that can occur during world operations.
#[derive(Debug, thiserror::Error)]
pub enum WorldError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("save format error: {0}")]
    SaveFormat(String),

    #[error("unsupported save version: {0} (expected <= {1})")]
    UnsupportedVersion(u32, u32),

    #[error("corrupt save data: {0}")]
    CorruptData(String),

    #[error("block registry error: {0}")]
    Registry(String),
}

/// Convenience alias for world operations.
pub type Result<T> = std::result::Result<T, WorldError>;
