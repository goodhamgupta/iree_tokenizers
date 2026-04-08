use std::{slice, sync::Mutex};

use rustler::{NifStruct, NifTaggedEnum, ResourceArc};

use crate::{
    error::{check_status, is_resource_exhausted, ErrorKind, Result, TokenizerError},
    ffi,
    stream::{DecodeStreamResource, DecodeStreamState, EncodeStreamResource, EncodeStreamState},
};

pub struct TokenizerResource {
    pub(crate) ptr: *mut ffi::iree_tokenizer_t,
    pub(crate) model_type: String,
}

unsafe impl Send for TokenizerResource {}
unsafe impl Sync for TokenizerResource {}

impl Drop for TokenizerResource {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { ffi::iree_tokenizer_free(self.ptr) };
        }
    }
}

#[rustler::resource_impl]
impl rustler::Resource for TokenizerResource {}

#[derive(NifStruct)]
#[module = "IREE.Tokenizers.Tokenizer"]
pub struct Tokenizer {
    pub resource: ResourceArc<TokenizerResource>,
}

#[derive(NifStruct)]
#[module = "IREE.Tokenizers.Encoding"]
pub struct Encoding {
    pub ids: Vec<i32>,
    pub type_ids: Vec<u8>,
    pub offsets: Option<Vec<(u64, u64)>>,
}

#[derive(NifTaggedEnum)]
pub enum EncodeOption {
    AddSpecialTokens(bool),
    TrackOffsets(bool),
}

#[derive(NifTaggedEnum)]
pub enum DecodeOption {
    SkipSpecialTokens(bool),
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn tokenizer_from_buffer(buffer: rustler::Binary) -> Result<Tokenizer> {
    let mut raw = std::ptr::null_mut();
    let status = unsafe {
        ffi::iree_tokenizer_from_huggingface_json(
            ffi::make_string_view(buffer.as_slice()),
            ffi::system_allocator(),
            &mut raw,
        )
    };
    check_status(status)?;

    let model_type = string_view_to_string(unsafe { ffi::iree_tokenizer_model_type_name(raw) });

    Ok(Tokenizer {
        resource: ResourceArc::new(TokenizerResource {
            ptr: raw,
            model_type,
        }),
    })
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn tokenizer_encode(
    tokenizer: Tokenizer,
    text: rustler::Binary,
    opts: Vec<EncodeOption>,
) -> Result<Encoding> {
    let options = parse_encode_options(opts);
    encode_impl(
        &tokenizer.resource,
        text.as_slice(),
        options.add_special_tokens,
        options.track_offsets,
    )
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn tokenizer_encode_batch(
    tokenizer: Tokenizer,
    texts: Vec<rustler::Binary>,
    opts: Vec<EncodeOption>,
) -> Result<Vec<Encoding>> {
    let options = parse_encode_options(opts);

    if texts.is_empty() {
        return Ok(Vec::new());
    }

    let mut state_size = 0usize;
    check_status(unsafe {
        ffi::iree_tokenizer_encode_state_calculate_size(tokenizer.resource.ptr, &mut state_size)
    })?;

    let max_len = texts.iter().map(|item| item.len()).max().unwrap_or(0);
    let transform_size = ffi::transform_buffer_oneshot_size(max_len);
    let mut state_storage = vec![0u8; state_size];
    let mut transform_buffer = vec![0u8; transform_size];
    let flags = encode_flags(options.add_special_tokens, options.track_offsets);

    let mut capacities: Vec<usize> = texts.iter().map(|item| item.len() / 2 + 16).collect();
    let mut ids_bufs: Vec<Vec<i32>> = capacities.iter().map(|&cap| vec![0; cap]).collect();
    let mut type_ids_bufs: Vec<Vec<u8>> = capacities.iter().map(|&cap| vec![0; cap]).collect();
    let mut offsets_bufs: Vec<Vec<ffi::iree_tokenizer_offset_t>> = capacities
        .iter()
        .map(|&cap| vec![ffi::iree_tokenizer_offset_t { start: 0, end: 0 }; cap])
        .collect();

    loop {
        let mut items: Vec<ffi::iree_tokenizer_encode_batch_item_t> = texts
            .iter()
            .enumerate()
            .map(|(index, text)| ffi::iree_tokenizer_encode_batch_item_t {
                text: ffi::make_string_view(text.as_slice()),
                output: ffi::iree_tokenizer_token_output_t {
                    capacity: capacities[index],
                    token_ids: ids_bufs[index].as_mut_ptr(),
                    token_offsets: if options.track_offsets {
                        offsets_bufs[index].as_mut_ptr()
                    } else {
                        std::ptr::null_mut()
                    },
                    type_ids: type_ids_bufs[index].as_mut_ptr(),
                },
                out_token_count: 0,
            })
            .collect();

        let status = unsafe {
            ffi::iree_tokenizer_encode_batch(
                tokenizer.resource.ptr,
                items.as_mut_ptr(),
                items.len(),
                flags,
                ffi::make_byte_span(&mut state_storage),
                ffi::make_byte_span(&mut transform_buffer),
                ffi::empty_offset_runs(),
            )
        };

        if is_resource_exhausted(status) {
            unsafe { ffi::iree_status_ignore(status) };
            for (index, text) in texts.iter().enumerate() {
                capacities[index] = capacities[index].max(text.len() + 64) * 2;
                ids_bufs[index] = vec![0; capacities[index]];
                type_ids_bufs[index] = vec![0; capacities[index]];
                offsets_bufs[index] =
                    vec![ffi::iree_tokenizer_offset_t { start: 0, end: 0 }; capacities[index]];
            }
            continue;
        }

        check_status(status)?;

        let encodings = items
            .into_iter()
            .enumerate()
            .map(|(index, item)| {
                ids_bufs[index].truncate(item.out_token_count);
                type_ids_bufs[index].truncate(item.out_token_count);
                let offsets = if options.track_offsets {
                    offsets_bufs[index].truncate(item.out_token_count);
                    Some(
                        offsets_bufs[index]
                            .iter()
                            .map(|offset| (offset.start as u64, offset.end as u64))
                            .collect(),
                    )
                } else {
                    None
                };

                Encoding {
                    ids: std::mem::take(&mut ids_bufs[index]),
                    type_ids: std::mem::take(&mut type_ids_bufs[index]),
                    offsets,
                }
            })
            .collect();

        return Ok(encodings);
    }
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn tokenizer_decode(
    tokenizer: Tokenizer,
    ids: Vec<i32>,
    opts: Vec<DecodeOption>,
) -> Result<String> {
    let options = parse_decode_options(opts);
    decode_impl(&tokenizer.resource, &ids, options.skip_special_tokens)
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn tokenizer_decode_batch(
    tokenizer: Tokenizer,
    batch_ids: Vec<Vec<i32>>,
    opts: Vec<DecodeOption>,
) -> Result<Vec<String>> {
    let options = parse_decode_options(opts);

    if batch_ids.is_empty() {
        return Ok(Vec::new());
    }

    let mut state_size = 0usize;
    check_status(unsafe {
        ffi::iree_tokenizer_decode_state_calculate_size(tokenizer.resource.ptr, &mut state_size)
    })?;

    let mut state_storage = vec![0u8; state_size];
    let flags = decode_flags(options.skip_special_tokens);
    let mut capacities: Vec<usize> = batch_ids.iter().map(|ids| ids.len() * 4 + 64).collect();
    let mut output_bufs: Vec<Vec<u8>> = capacities.iter().map(|&cap| vec![0; cap]).collect();

    loop {
        let mut items: Vec<ffi::iree_tokenizer_decode_batch_item_t> = batch_ids
            .iter()
            .enumerate()
            .map(|(index, ids)| ffi::iree_tokenizer_decode_batch_item_t {
                tokens: ffi::iree_tokenizer_token_id_list_t {
                    count: ids.len(),
                    values: ids.as_ptr(),
                },
                text_output: ffi::make_mutable_string_view(&mut output_bufs[index]),
                out_text_length: 0,
            })
            .collect();

        let status = unsafe {
            ffi::iree_tokenizer_decode_batch(
                tokenizer.resource.ptr,
                items.as_mut_ptr(),
                items.len(),
                flags,
                ffi::make_byte_span(&mut state_storage),
            )
        };

        if is_resource_exhausted(status) {
            unsafe { ffi::iree_status_ignore(status) };
            for (index, ids) in batch_ids.iter().enumerate() {
                capacities[index] = capacities[index].max(ids.len() * 8 + 128) * 2;
                output_bufs[index] = vec![0; capacities[index]];
            }
            continue;
        }

        check_status(status)?;

        let mut results = Vec::with_capacity(items.len());
        for (index, item) in items.iter().enumerate() {
            results.push(
                String::from_utf8_lossy(&output_bufs[index][..item.out_text_length]).into_owned(),
            );
        }
        return Ok(results);
    }
}

#[rustler::nif]
pub fn tokenizer_vocab_size(tokenizer: Tokenizer) -> usize {
    let vocab = unsafe { ffi::iree_tokenizer_vocab(tokenizer.resource.ptr) };
    unsafe { ffi::iree_tokenizer_vocab_token_count(vocab) }
}

#[rustler::nif]
pub fn tokenizer_model_type(tokenizer: Tokenizer) -> String {
    tokenizer.resource.model_type.clone()
}

#[rustler::nif]
pub fn tokenizer_token_to_id(tokenizer: Tokenizer, token: rustler::Binary) -> Option<i32> {
    let vocab = unsafe { ffi::iree_tokenizer_vocab(tokenizer.resource.ptr) };
    let id =
        unsafe { ffi::iree_tokenizer_vocab_lookup(vocab, ffi::make_string_view(token.as_slice())) };
    (id >= 0).then_some(id)
}

#[rustler::nif]
pub fn tokenizer_id_to_token(tokenizer: Tokenizer, id: i32) -> Option<String> {
    if id < 0 {
        return None;
    }

    let vocab = unsafe { ffi::iree_tokenizer_vocab(tokenizer.resource.ptr) };
    let view = unsafe { ffi::iree_tokenizer_vocab_token_text(vocab, id) };
    (!view.data.is_null() && view.size > 0).then(|| string_view_to_string(view))
}

#[rustler::nif]
pub fn tokenizer_bos_token_id(tokenizer: Tokenizer) -> Option<i32> {
    special_id(&tokenizer.resource, |ids| ids.bos)
}

#[rustler::nif]
pub fn tokenizer_eos_token_id(tokenizer: Tokenizer) -> Option<i32> {
    special_id(&tokenizer.resource, |ids| ids.eos)
}

#[rustler::nif]
pub fn tokenizer_unk_token_id(tokenizer: Tokenizer) -> Option<i32> {
    special_id(&tokenizer.resource, |ids| ids.unk)
}

#[rustler::nif]
pub fn tokenizer_pad_token_id(tokenizer: Tokenizer) -> Option<i32> {
    special_id(&tokenizer.resource, |ids| ids.pad)
}

#[rustler::nif]
pub fn tokenizer_sep_token_id(tokenizer: Tokenizer) -> Option<i32> {
    special_id(&tokenizer.resource, |ids| ids.sep)
}

#[rustler::nif]
pub fn tokenizer_cls_token_id(tokenizer: Tokenizer) -> Option<i32> {
    special_id(&tokenizer.resource, |ids| ids.cls)
}

#[rustler::nif]
pub fn tokenizer_mask_token_id(tokenizer: Tokenizer) -> Option<i32> {
    special_id(&tokenizer.resource, |ids| ids.mask)
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn encode_stream_new(
    tokenizer: Tokenizer,
    opts: Vec<EncodeOption>,
) -> Result<crate::stream::EncodeStream> {
    let options = parse_encode_options(opts);
    let state = EncodeStreamState::new(tokenizer.resource.clone(), options.add_special_tokens)?;
    Ok(crate::stream::EncodeStream {
        resource: ResourceArc::new(EncodeStreamResource {
            _tokenizer: tokenizer.resource,
            inner: Mutex::new(Some(state)),
        }),
    })
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn decode_stream_new(
    tokenizer: Tokenizer,
    opts: Vec<DecodeOption>,
) -> Result<crate::stream::DecodeStream> {
    let options = parse_decode_options(opts);
    let state = DecodeStreamState::new(tokenizer.resource.clone(), options.skip_special_tokens)?;
    Ok(crate::stream::DecodeStream {
        resource: ResourceArc::new(DecodeStreamResource {
            _tokenizer: tokenizer.resource,
            inner: Mutex::new(Some(state)),
        }),
    })
}

pub fn encode_impl(
    tokenizer: &TokenizerResource,
    text: &[u8],
    add_special_tokens: bool,
    track_offsets: bool,
) -> Result<Encoding> {
    let mut capacity = text.len() / 2 + 16;
    let flags = encode_flags(add_special_tokens, track_offsets);

    loop {
        let mut ids = vec![0i32; capacity];
        let mut type_ids = vec![0u8; capacity];
        let mut offsets = vec![ffi::iree_tokenizer_offset_t { start: 0, end: 0 }; capacity];
        let mut token_count = 0usize;

        let status = unsafe {
            ffi::iree_tokenizer_encode(
                tokenizer.ptr,
                ffi::make_string_view(text),
                flags,
                ffi::iree_tokenizer_token_output_t {
                    capacity,
                    token_ids: ids.as_mut_ptr(),
                    token_offsets: if track_offsets {
                        offsets.as_mut_ptr()
                    } else {
                        std::ptr::null_mut()
                    },
                    type_ids: type_ids.as_mut_ptr(),
                },
                ffi::system_allocator(),
                &mut token_count,
            )
        };

        if is_resource_exhausted(status) {
            unsafe { ffi::iree_status_ignore(status) };
            capacity = capacity.max(text.len() + 64) * 2;
            continue;
        }

        check_status(status)?;
        ids.truncate(token_count);
        type_ids.truncate(token_count);

        let offsets = if track_offsets {
            offsets.truncate(token_count);
            Some(
                offsets
                    .into_iter()
                    .map(|offset| (offset.start as u64, offset.end as u64))
                    .collect(),
            )
        } else {
            None
        };

        return Ok(Encoding {
            ids,
            type_ids,
            offsets,
        });
    }
}

pub fn decode_impl(
    tokenizer: &TokenizerResource,
    ids: &[i32],
    skip_special_tokens: bool,
) -> Result<String> {
    let mut capacity = ids.len() * 4 + 64;
    let flags = decode_flags(skip_special_tokens);

    loop {
        let mut bytes = vec![0u8; capacity];
        let mut written = 0usize;
        let status = unsafe {
            ffi::iree_tokenizer_decode(
                tokenizer.ptr,
                ffi::iree_tokenizer_token_id_list_t {
                    count: ids.len(),
                    values: ids.as_ptr(),
                },
                flags,
                ffi::make_mutable_string_view(&mut bytes),
                ffi::system_allocator(),
                &mut written,
            )
        };

        if is_resource_exhausted(status) {
            unsafe { ffi::iree_status_ignore(status) };
            capacity = capacity.max(ids.len() * 8 + 128) * 2;
            continue;
        }

        check_status(status)?;
        bytes.truncate(written);
        return Ok(String::from_utf8_lossy(&bytes).into_owned());
    }
}

fn parse_encode_options(opts: Vec<EncodeOption>) -> ParsedEncodeOptions {
    let mut parsed = ParsedEncodeOptions {
        add_special_tokens: true,
        track_offsets: false,
    };

    for opt in opts {
        match opt {
            EncodeOption::AddSpecialTokens(value) => parsed.add_special_tokens = value,
            EncodeOption::TrackOffsets(value) => parsed.track_offsets = value,
        }
    }

    parsed
}

fn parse_decode_options(opts: Vec<DecodeOption>) -> ParsedDecodeOptions {
    let mut parsed = ParsedDecodeOptions {
        skip_special_tokens: true,
    };

    for opt in opts {
        match opt {
            DecodeOption::SkipSpecialTokens(value) => parsed.skip_special_tokens = value,
        }
    }

    parsed
}

fn encode_flags(add_special_tokens: bool, track_offsets: bool) -> u32 {
    let mut flags = ffi::encode_flag_bits::NONE;
    if add_special_tokens {
        flags |= ffi::encode_flag_bits::ADD_SPECIAL_TOKENS;
    }
    if track_offsets {
        flags |= ffi::encode_flag_bits::TRACK_OFFSETS;
    }
    flags
}

fn decode_flags(skip_special_tokens: bool) -> u32 {
    if skip_special_tokens {
        ffi::decode_flag_bits::SKIP_SPECIAL_TOKENS
    } else {
        ffi::decode_flag_bits::NONE
    }
}

fn string_view_to_string(view: ffi::iree_string_view_t) -> String {
    if view.data.is_null() || view.size == 0 {
        return String::new();
    }

    let bytes = unsafe { slice::from_raw_parts(view.data as *const u8, view.size) };
    String::from_utf8_lossy(bytes).into_owned()
}

fn special_id(
    tokenizer: &TokenizerResource,
    accessor: impl Fn(ffi::iree_tokenizer_special_ids_t) -> i32,
) -> Option<i32> {
    let vocab = unsafe { ffi::iree_tokenizer_vocab(tokenizer.ptr) };
    let ids = unsafe { ffi::iree_tokenizer_vocab_special_ids(vocab) };
    let value = accessor(ids);
    (value >= 0).then_some(value)
}

struct ParsedEncodeOptions {
    add_special_tokens: bool,
    track_offsets: bool,
}

struct ParsedDecodeOptions {
    skip_special_tokens: bool,
}

pub(crate) fn invalid_argument(message: impl Into<String>) -> TokenizerError {
    TokenizerError::new(ErrorKind::InvalidArgument, message)
}
