use std::{ffi::CStr, fmt};

use rustler::{Encoder, Env, Term};
use thiserror::Error;

use crate::ffi;

rustler::atoms! {
    cancelled,
    unknown,
    invalid_argument,
    deadline_exceeded,
    not_found,
    already_exists,
    permission_denied,
    resource_exhausted,
    failed_precondition,
    aborted,
    out_of_range,
    unimplemented,
    internal,
    unavailable,
    data_loss,
    unauthenticated,
    deferred,
    incompatible
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorKind {
    Cancelled,
    Unknown,
    InvalidArgument,
    DeadlineExceeded,
    NotFound,
    AlreadyExists,
    PermissionDenied,
    ResourceExhausted,
    FailedPrecondition,
    Aborted,
    OutOfRange,
    Unimplemented,
    Internal,
    Unavailable,
    DataLoss,
    Unauthenticated,
    Deferred,
    Incompatible,
}

#[derive(Debug, Error)]
#[error("{message}")]
pub struct TokenizerError {
    pub kind: ErrorKind,
    pub message: String,
}

impl TokenizerError {
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }
}

impl Encoder for TokenizerError {
    fn encode<'a>(&self, env: Env<'a>) -> Term<'a> {
        (kind_to_atom(self.kind), self.message.as_str()).encode(env)
    }
}

pub type Result<T> = std::result::Result<T, TokenizerError>;

pub fn check_status(status: ffi::iree_status_t) -> Result<()> {
    if status.is_null() {
        return Ok(());
    }

    let code = unsafe { ffi::iree_status_consume_code(status) } as u32;
    Err(TokenizerError::new(
        code_to_kind(code),
        status_code_message(code),
    ))
}

pub fn is_resource_exhausted(status: ffi::iree_status_t) -> bool {
    if status.is_null() {
        return false;
    }

    (status as usize & ffi::IREE_STATUS_CODE_MASK as usize)
        == ffi::iree_status_code_e::IREE_STATUS_RESOURCE_EXHAUSTED as usize
}

pub fn status_code_message(code: u32) -> String {
    let ptr = unsafe { ffi::iree_status_code_string(code) };
    if ptr.is_null() {
        return format!("status code {code}");
    }

    unsafe { CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned()
}

fn code_to_kind(code: u32) -> ErrorKind {
    match code {
        x if x == ffi::iree_status_code_e::IREE_STATUS_CANCELLED as u32 => ErrorKind::Cancelled,
        x if x == ffi::iree_status_code_e::IREE_STATUS_UNKNOWN as u32 => ErrorKind::Unknown,
        x if x == ffi::iree_status_code_e::IREE_STATUS_INVALID_ARGUMENT as u32 => {
            ErrorKind::InvalidArgument
        }
        x if x == ffi::iree_status_code_e::IREE_STATUS_DEADLINE_EXCEEDED as u32 => {
            ErrorKind::DeadlineExceeded
        }
        x if x == ffi::iree_status_code_e::IREE_STATUS_NOT_FOUND as u32 => ErrorKind::NotFound,
        x if x == ffi::iree_status_code_e::IREE_STATUS_ALREADY_EXISTS as u32 => {
            ErrorKind::AlreadyExists
        }
        x if x == ffi::iree_status_code_e::IREE_STATUS_PERMISSION_DENIED as u32 => {
            ErrorKind::PermissionDenied
        }
        x if x == ffi::iree_status_code_e::IREE_STATUS_RESOURCE_EXHAUSTED as u32 => {
            ErrorKind::ResourceExhausted
        }
        x if x == ffi::iree_status_code_e::IREE_STATUS_FAILED_PRECONDITION as u32 => {
            ErrorKind::FailedPrecondition
        }
        x if x == ffi::iree_status_code_e::IREE_STATUS_ABORTED as u32 => ErrorKind::Aborted,
        x if x == ffi::iree_status_code_e::IREE_STATUS_OUT_OF_RANGE as u32 => ErrorKind::OutOfRange,
        x if x == ffi::iree_status_code_e::IREE_STATUS_UNIMPLEMENTED as u32 => {
            ErrorKind::Unimplemented
        }
        x if x == ffi::iree_status_code_e::IREE_STATUS_INTERNAL as u32 => ErrorKind::Internal,
        x if x == ffi::iree_status_code_e::IREE_STATUS_UNAVAILABLE as u32 => ErrorKind::Unavailable,
        x if x == ffi::iree_status_code_e::IREE_STATUS_DATA_LOSS as u32 => ErrorKind::DataLoss,
        x if x == ffi::iree_status_code_e::IREE_STATUS_UNAUTHENTICATED as u32 => {
            ErrorKind::Unauthenticated
        }
        x if x == ffi::iree_status_code_e::IREE_STATUS_DEFERRED as u32 => ErrorKind::Deferred,
        x if x == ffi::iree_status_code_e::IREE_STATUS_INCOMPATIBLE as u32 => {
            ErrorKind::Incompatible
        }
        _ => ErrorKind::Internal,
    }
}

fn kind_to_atom(kind: ErrorKind) -> rustler::Atom {
    match kind {
        ErrorKind::Cancelled => cancelled(),
        ErrorKind::Unknown => unknown(),
        ErrorKind::InvalidArgument => invalid_argument(),
        ErrorKind::DeadlineExceeded => deadline_exceeded(),
        ErrorKind::NotFound => not_found(),
        ErrorKind::AlreadyExists => already_exists(),
        ErrorKind::PermissionDenied => permission_denied(),
        ErrorKind::ResourceExhausted => resource_exhausted(),
        ErrorKind::FailedPrecondition => failed_precondition(),
        ErrorKind::Aborted => aborted(),
        ErrorKind::OutOfRange => out_of_range(),
        ErrorKind::Unimplemented => unimplemented(),
        ErrorKind::Internal => internal(),
        ErrorKind::Unavailable => unavailable(),
        ErrorKind::DataLoss => data_loss(),
        ErrorKind::Unauthenticated => unauthenticated(),
        ErrorKind::Deferred => deferred(),
        ErrorKind::Incompatible => incompatible(),
    }
}

impl fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
