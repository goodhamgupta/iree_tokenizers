use std::sync::Mutex;

use rustler::{NifStruct, ResourceArc};

use crate::{
    error::{check_status, is_resource_exhausted, ErrorKind, Result, TokenizerError},
    ffi,
    tokenizer::{invalid_argument, TokenizerResource},
};

pub struct EncodeStreamState {
    _tokenizer: ResourceArc<TokenizerResource>,
    state: *mut ffi::iree_tokenizer_encode_state_t,
    _state_storage: Vec<u8>,
    _transform_buffer: Vec<u8>,
}

unsafe impl Send for EncodeStreamState {}

impl Drop for EncodeStreamState {
    fn drop(&mut self) {
        if !self.state.is_null() {
            unsafe { ffi::iree_tokenizer_encode_state_deinitialize(self.state) };
        }
    }
}

impl EncodeStreamState {
    pub fn new(
        tokenizer: ResourceArc<TokenizerResource>,
        add_special_tokens: bool,
        max_chunk_bytes: usize,
    ) -> Result<Self> {
        let mut state_size = 0usize;
        check_status(unsafe {
            ffi::iree_tokenizer_encode_state_calculate_size(tokenizer.ptr, &mut state_size)
        })?;

        let mut state_storage = vec![0u8; state_size];
        let mut transform_buffer =
            vec![0u8; ffi::transform_buffer_recommended_size(max_chunk_bytes)];
        let mut state = std::ptr::null_mut();
        let mut flags = ffi::encode_flag_bits::AT_INPUT_START;
        if add_special_tokens {
            flags |= ffi::encode_flag_bits::ADD_SPECIAL_TOKENS;
        }

        check_status(unsafe {
            ffi::iree_tokenizer_encode_state_initialize(
                tokenizer.ptr,
                ffi::make_byte_span(&mut state_storage),
                ffi::make_byte_span(&mut transform_buffer),
                ffi::empty_offset_runs(),
                flags,
                &mut state,
            )
        })?;

        Ok(Self {
            _tokenizer: tokenizer,
            state,
            _state_storage: state_storage,
            _transform_buffer: transform_buffer,
        })
    }

    fn feed(&mut self, chunk: &[u8]) -> Result<Vec<i32>> {
        let mut all_ids = Vec::new();
        let mut offset = 0usize;
        let mut ids_buffer = vec![0i32; 1024];

        while offset < chunk.len() {
            let (consumed, produced) =
                encode_feed_once(self.state, &chunk[offset..], &mut ids_buffer)?;

            if consumed == 0 && produced == 0 {
                return Err(invalid_argument("encode stream made no progress"));
            }

            if produced > 0 {
                all_ids.extend_from_slice(&ids_buffer[..produced]);
            }

            offset += consumed;
        }

        Ok(all_ids)
    }

    fn finalize(self) -> Result<Vec<i32>> {
        let mut all_ids = Vec::new();

        loop {
            let mut ids = vec![0i32; 256];

            loop {
                let mut produced = 0usize;
                let status = unsafe {
                    ffi::iree_tokenizer_encode_state_finalize(
                        self.state,
                        ffi::iree_tokenizer_token_output_t {
                            capacity: ids.len(),
                            token_ids: ids.as_mut_ptr(),
                            token_offsets: std::ptr::null_mut(),
                            type_ids: std::ptr::null_mut(),
                        },
                        &mut produced,
                    )
                };

                if is_resource_exhausted(status) {
                    unsafe { ffi::iree_status_ignore(status) };
                    ids.resize(ids.len() * 2, 0);
                    continue;
                }

                check_status(status)?;
                ids.truncate(produced);
                all_ids.extend_from_slice(&ids);

                let has_pending =
                    unsafe { ffi::iree_tokenizer_encode_state_has_pending(self.state) };
                if !has_pending {
                    return Ok(all_ids);
                }

                if produced == 0 {
                    return Err(invalid_argument(
                        "encode stream finalize made no progress while pending data remained",
                    ));
                }

                break;
            }
        }
    }
}

pub struct DecodeStreamState {
    _tokenizer: ResourceArc<TokenizerResource>,
    state: *mut ffi::iree_tokenizer_decode_state_t,
    _state_storage: Vec<u8>,
}

unsafe impl Send for DecodeStreamState {}

impl Drop for DecodeStreamState {
    fn drop(&mut self) {
        if !self.state.is_null() {
            unsafe { ffi::iree_tokenizer_decode_state_deinitialize(self.state) };
        }
    }
}

impl DecodeStreamState {
    pub fn new(
        tokenizer: ResourceArc<TokenizerResource>,
        skip_special_tokens: bool,
    ) -> Result<Self> {
        let mut state_size = 0usize;
        check_status(unsafe {
            ffi::iree_tokenizer_decode_state_calculate_size(tokenizer.ptr, &mut state_size)
        })?;

        let mut state_storage = vec![0u8; state_size];
        let mut state = std::ptr::null_mut();
        let flags = if skip_special_tokens {
            ffi::decode_flag_bits::SKIP_SPECIAL_TOKENS
        } else {
            ffi::decode_flag_bits::NONE
        };

        check_status(unsafe {
            ffi::iree_tokenizer_decode_state_initialize(
                tokenizer.ptr,
                flags,
                ffi::make_byte_span(&mut state_storage),
                &mut state,
            )
        })?;

        Ok(Self {
            _tokenizer: tokenizer,
            state,
            _state_storage: state_storage,
        })
    }

    fn feed(&mut self, ids: &[i32]) -> Result<String> {
        let mut output = String::new();
        let mut offset = 0usize;
        let mut text_buffer = vec![0u8; ffi::IREE_TOKENIZER_DECODE_OUTPUT_RECOMMENDED_SIZE];

        while offset < ids.len() {
            let (consumed, written) =
                decode_feed_once(self.state, &ids[offset..], &mut text_buffer)?;

            if consumed == 0 && written == 0 {
                return Err(invalid_argument("decode stream made no progress"));
            }

            if written > 0 {
                output.push_str(&String::from_utf8(text_buffer[..written].to_vec()).map_err(
                    |err| {
                        TokenizerError::new(
                            ErrorKind::Internal,
                            format!("invalid UTF-8 in decode stream output: {err}"),
                        )
                    },
                )?);
            }

            offset += consumed;
        }

        Ok(output)
    }

    fn finalize(self) -> Result<String> {
        let mut text_buffer = vec![0u8; ffi::IREE_TOKENIZER_DECODE_OUTPUT_RECOMMENDED_SIZE];

        loop {
            let mut written = 0usize;
            let status = unsafe {
                ffi::iree_tokenizer_decode_state_finalize(
                    self.state,
                    ffi::make_mutable_string_view(&mut text_buffer),
                    &mut written,
                )
            };

            if is_resource_exhausted(status) {
                unsafe { ffi::iree_status_ignore(status) };
                text_buffer.resize(text_buffer.len() * 2, 0);
                continue;
            }

            check_status(status)?;
            return String::from_utf8(text_buffer[..written].to_vec()).map_err(|err| {
                TokenizerError::new(
                    ErrorKind::Internal,
                    format!("invalid UTF-8 in decode stream output: {err}"),
                )
            });
        }
    }
}

pub struct EncodeStreamResource {
    pub _tokenizer: ResourceArc<TokenizerResource>,
    pub inner: Mutex<Option<EncodeStreamState>>,
}

#[rustler::resource_impl]
impl rustler::Resource for EncodeStreamResource {}

#[derive(NifStruct)]
#[module = "IREE.Tokenizers.EncodeStream"]
pub struct EncodeStream {
    pub resource: ResourceArc<EncodeStreamResource>,
}

pub struct DecodeStreamResource {
    pub _tokenizer: ResourceArc<TokenizerResource>,
    pub inner: Mutex<Option<DecodeStreamState>>,
}

#[rustler::resource_impl]
impl rustler::Resource for DecodeStreamResource {}

#[derive(NifStruct)]
#[module = "IREE.Tokenizers.DecodeStream"]
pub struct DecodeStream {
    pub resource: ResourceArc<DecodeStreamResource>,
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn encode_stream_feed(stream: EncodeStream, chunk: rustler::Binary) -> Result<Vec<i32>> {
    let mut guard = recover_lock(stream.resource.inner.lock());
    let state = guard
        .as_mut()
        .ok_or_else(|| invalid_argument("stream already finalized"))?;
    state.feed(chunk.as_slice())
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn encode_stream_finalize(stream: EncodeStream) -> Result<Vec<i32>> {
    let mut guard = recover_lock(stream.resource.inner.lock());
    let state = guard
        .take()
        .ok_or_else(|| invalid_argument("stream already finalized"))?;
    state.finalize()
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn decode_stream_feed(stream: DecodeStream, ids: Vec<i32>) -> Result<String> {
    let mut guard = recover_lock(stream.resource.inner.lock());
    let state = guard
        .as_mut()
        .ok_or_else(|| invalid_argument("stream already finalized"))?;
    state.feed(&ids)
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn decode_stream_finalize(stream: DecodeStream) -> Result<String> {
    let mut guard = recover_lock(stream.resource.inner.lock());
    let state = guard
        .take()
        .ok_or_else(|| invalid_argument("stream already finalized"))?;
    state.finalize()
}

fn recover_lock<T>(
    lock: std::sync::LockResult<std::sync::MutexGuard<'_, T>>,
) -> std::sync::MutexGuard<'_, T> {
    lock.unwrap_or_else(|err| err.into_inner())
}

fn encode_feed_once(
    state: *mut ffi::iree_tokenizer_encode_state_t,
    chunk: &[u8],
    ids_buffer: &mut Vec<i32>,
) -> Result<(usize, usize)> {
    loop {
        let mut consumed = 0usize;
        let mut produced = 0usize;
        let status = unsafe {
            ffi::iree_tokenizer_encode_state_feed(
                state,
                ffi::make_string_view(chunk),
                ffi::iree_tokenizer_token_output_t {
                    capacity: ids_buffer.len(),
                    token_ids: ids_buffer.as_mut_ptr(),
                    token_offsets: std::ptr::null_mut(),
                    type_ids: std::ptr::null_mut(),
                },
                &mut consumed,
                &mut produced,
            )
        };

        if is_resource_exhausted(status) {
            unsafe { ffi::iree_status_ignore(status) };
            ids_buffer.resize(ids_buffer.len() * 2, 0);
            continue;
        }

        check_status(status)?;

        if consumed == 0 && produced == 0 && !chunk.is_empty() {
            ids_buffer.resize(ids_buffer.len() * 2, 0);
            continue;
        }

        return Ok((consumed, produced));
    }
}

fn decode_feed_once(
    state: *mut ffi::iree_tokenizer_decode_state_t,
    ids: &[i32],
    text_buffer: &mut Vec<u8>,
) -> Result<(usize, usize)> {
    loop {
        let mut consumed = 0usize;
        let mut written = 0usize;
        let status = unsafe {
            ffi::iree_tokenizer_decode_state_feed(
                state,
                ffi::iree_tokenizer_token_id_list_t {
                    count: ids.len(),
                    values: ids.as_ptr(),
                },
                ffi::make_mutable_string_view(text_buffer.as_mut_slice()),
                &mut consumed,
                &mut written,
            )
        };

        if is_resource_exhausted(status) {
            unsafe { ffi::iree_status_ignore(status) };
            text_buffer.resize(text_buffer.len() * 2, 0);
            continue;
        }

        check_status(status)?;

        if consumed == 0 && written == 0 && !ids.is_empty() {
            text_buffer.resize(text_buffer.len() * 2, 0);
            continue;
        }

        return Ok((consumed, written));
    }
}
