use std::{slice, sync::Mutex};

use rustler::{NifStruct, NifTaggedEnum, ResourceArc};
use serde_json::Value;

use crate::{
    error::{check_status, is_resource_exhausted, ErrorKind, Result, TokenizerError},
    ffi, sentencepiece,
    stream::{DecodeStreamResource, DecodeStreamState, EncodeStreamResource, EncodeStreamState},
};

const SUPPORTED_TIKTOKEN_ENCODINGS: [&str; 7] = [
    "cl100k_base",
    "o200k_base",
    "o200k_harmony",
    "r50k_base",
    "gpt2",
    "p50k_base",
    "p50k_edit",
];

pub struct TokenizerResource {
    pub(crate) ptr: *mut ffi::iree_tokenizer_t,
    pub(crate) model_type: String,
    pub(crate) decode_strategy: DecodeStrategy,
    pub(crate) stream_encode_strategy: StreamEncodeStrategy,
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
    pub attention_mask: Vec<u8>,
    pub special_tokens_mask: Vec<u8>,
    pub tokens: Vec<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DecodeStrategy {
    Native,
    SentencePieceBpe,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum StreamEncodeStrategy {
    Native,
    BufferedFinalize,
}

#[derive(NifTaggedEnum)]
pub enum EncodeOption {
    AddSpecialTokens(bool),
    TrackOffsets(bool),
    MaxChunkBytes(usize),
}

#[derive(NifTaggedEnum)]
pub enum DecodeOption {
    SkipSpecialTokens(bool),
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn tokenizer_from_buffer(buffer: rustler::Binary) -> Result<Tokenizer> {
    let metadata = tokenizer_metadata_from_hf_json(buffer.as_slice());
    let mut raw = std::ptr::null_mut();
    let status = unsafe {
        ffi::iree_tokenizer_from_huggingface_json(
            ffi::make_string_view(buffer.as_slice()),
            ffi::system_allocator(),
            &mut raw,
        )
    };
    check_status(status)?;

    tokenizer_from_raw(raw, metadata)
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn tokenizer_from_tiktoken_buffer(
    buffer: rustler::Binary,
    encoding: String,
) -> Result<Tokenizer> {
    let config = tiktoken_config_by_name(&encoding)?;
    let mut raw = std::ptr::null_mut();
    let status = unsafe {
        ffi::iree_tokenizer_from_tiktoken(
            ffi::make_string_view(buffer.as_slice()),
            config,
            ffi::system_allocator(),
            &mut raw,
        )
    };
    check_status(status)?;

    tokenizer_from_raw(raw, TokenizerMetadata::default())
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn tokenizer_from_sentencepiece_model(buffer: rustler::Binary) -> Result<Tokenizer> {
    let json = sentencepiece::model_to_tokenizer_json(buffer.as_slice())?;
    let metadata = tokenizer_metadata_from_hf_json(json.as_slice());
    let mut raw = std::ptr::null_mut();
    let status = unsafe {
        ffi::iree_tokenizer_from_huggingface_json(
            ffi::make_string_view(json.as_slice()),
            ffi::system_allocator(),
            &mut raw,
        )
    };
    check_status(status)?;

    tokenizer_from_raw(raw, metadata)
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
    // Always request offsets internally so `encoding_metadata` can substitute
    // source text for UNK ids; only surface them to Elixir when the caller
    // asked for them. See the matching comment in `encode_impl`.
    let flags = encode_flags(options.add_special_tokens, true);

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
                text_pair: ffi::empty_string_view(),
                flags: 0,
                output: ffi::iree_tokenizer_token_output_t {
                    capacity: capacities[index],
                    token_ids: ids_bufs[index].as_mut_ptr(),
                    token_offsets: offsets_bufs[index].as_mut_ptr(),
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

        // Silent-truncation guard: see the comment in `encode_impl`. If any
        // item's output exactly fills its buffer and we still have headroom
        // below the worst-case upper bound, grow those items and retry.
        let suspected_truncation = items.iter().enumerate().any(|(index, item)| {
            item.out_token_count == capacities[index]
                && capacities[index] <= texts[index].len() + 16
        });

        if suspected_truncation {
            for (index, text) in texts.iter().enumerate() {
                if items[index].out_token_count == capacities[index]
                    && capacities[index] <= text.len() + 16
                {
                    capacities[index] = capacities[index].max(text.len() + 64) * 2;
                    ids_bufs[index] = vec![0; capacities[index]];
                    type_ids_bufs[index] = vec![0; capacities[index]];
                    offsets_bufs[index] =
                        vec![ffi::iree_tokenizer_offset_t { start: 0, end: 0 }; capacities[index]];
                }
            }
            continue;
        }

        let encodings = items
            .into_iter()
            .enumerate()
            .map(|(index, item)| {
                ids_bufs[index].truncate(item.out_token_count);
                type_ids_bufs[index].truncate(item.out_token_count);
                offsets_bufs[index].truncate(item.out_token_count);

                let (special_tokens_mask, tokens) = encoding_metadata(
                    &tokenizer.resource,
                    &ids_bufs[index],
                    texts[index].as_slice(),
                    &offsets_bufs[index],
                );

                let returned_offsets = if options.track_offsets {
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
                    offsets: returned_offsets,
                    attention_mask: vec![1; item.out_token_count],
                    special_tokens_mask,
                    tokens,
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
    match tokenizer.resource.decode_strategy {
        DecodeStrategy::Native => {
            decode_impl(&tokenizer.resource, &ids, options.skip_special_tokens)
        }
        DecodeStrategy::SentencePieceBpe => {
            decode_sentencepiece_bpe(&tokenizer.resource, &ids, options.skip_special_tokens)
        }
    }
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

    if tokenizer.resource.decode_strategy == DecodeStrategy::SentencePieceBpe {
        return batch_ids
            .iter()
            .map(|ids| {
                decode_sentencepiece_bpe(&tokenizer.resource, ids, options.skip_special_tokens)
            })
            .collect();
    }

    let mut state_size = 0usize;
    check_status(unsafe {
        ffi::iree_tokenizer_decode_state_calculate_size(tokenizer.resource.ptr, &mut state_size)
    })?;

    let mut state_storage = vec![0u8; state_size];
    let flags = decode_flags(options.skip_special_tokens);
    let mut capacities: Vec<usize> = batch_ids.iter().map(|ids| ids.len() * 8 + 128).collect();
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
                capacities[index] = capacities[index].max(ids.len() * 16 + 256) * 2;
                output_bufs[index] = vec![0; capacities[index]];
            }
            continue;
        }

        check_status(status)?;

        let mut results = Vec::with_capacity(items.len());
        for (index, item) in items.iter().enumerate() {
            results.push(
                String::from_utf8(output_bufs[index][..item.out_text_length].to_vec()).map_err(
                    |err| {
                        TokenizerError::new(
                            ErrorKind::Internal,
                            format!("invalid UTF-8 in decode batch output: {err}"),
                        )
                    },
                )?,
            );
        }
        return Ok(results);
    }
}

// Variant of `tokenizer_decode_batch` that reads token ids from a list of
// pre-packed `u32` little-endian binaries. Avoids cons-cell traversal of
// `Vec<Vec<i32>>`, which is highly heap-layout sensitive (4–6× slowdowns
// observed when callers' input lists are scattered across a fragmented BEAM
// process heap, e.g. immediately after an `encode_batch` call).
#[rustler::nif(schedule = "DirtyCpu")]
pub fn tokenizer_decode_batch_u32<'a>(
    tokenizer: Tokenizer,
    batch_ids: Vec<rustler::Binary<'a>>,
    opts: Vec<DecodeOption>,
) -> Result<Vec<String>> {
    let options = parse_decode_options(opts);

    if batch_ids.is_empty() {
        return Ok(Vec::new());
    }

    for (index, bin) in batch_ids.iter().enumerate() {
        if bin.as_slice().len() % 4 != 0 {
            return Err(TokenizerError::new(
                ErrorKind::InvalidArgument,
                format!(
                    "decode_batch_u32 input {} length {} is not a multiple of 4",
                    index,
                    bin.as_slice().len()
                ),
            ));
        }
    }

    let lens: Vec<usize> = batch_ids.iter().map(|b| b.as_slice().len() / 4).collect();

    // Reinterpret each binary as &[i32]. BEAM binaries are word-aligned and
    // every architecture we ship (x86_64, aarch64) is little-endian and
    // tolerates 4-byte unaligned loads, so the cast is sound.
    let id_slices: Vec<&[i32]> = batch_ids
        .iter()
        .map(|bin| {
            let bytes = bin.as_slice();
            unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const i32, bytes.len() / 4) }
        })
        .collect();

    if tokenizer.resource.decode_strategy == DecodeStrategy::SentencePieceBpe {
        return id_slices
            .iter()
            .map(|ids| decode_sentencepiece_bpe(&tokenizer.resource, ids, options.skip_special_tokens))
            .collect();
    }

    let mut state_size = 0usize;
    check_status(unsafe {
        ffi::iree_tokenizer_decode_state_calculate_size(tokenizer.resource.ptr, &mut state_size)
    })?;

    let mut state_storage = vec![0u8; state_size];
    let flags = decode_flags(options.skip_special_tokens);
    let mut capacities: Vec<usize> = lens.iter().map(|&n| n * 8 + 128).collect();
    let mut output_bufs: Vec<Vec<u8>> = capacities.iter().map(|&cap| vec![0; cap]).collect();

    loop {
        let mut items: Vec<ffi::iree_tokenizer_decode_batch_item_t> = id_slices
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
            for (index, &len) in lens.iter().enumerate() {
                capacities[index] = capacities[index].max(len * 16 + 256) * 2;
                output_bufs[index] = vec![0; capacities[index]];
            }
            continue;
        }

        check_status(status)?;

        let mut results = Vec::with_capacity(items.len());
        for (index, item) in items.iter().enumerate() {
            results.push(
                String::from_utf8(output_bufs[index][..item.out_text_length].to_vec()).map_err(
                    |err| {
                        TokenizerError::new(
                            ErrorKind::Internal,
                            format!("invalid UTF-8 in decode batch output: {err}"),
                        )
                    },
                )?,
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
pub fn tokenizer_vocab_capacity(tokenizer: Tokenizer) -> usize {
    let vocab = unsafe { ffi::iree_tokenizer_vocab(tokenizer.resource.ptr) };
    unsafe { ffi::iree_tokenizer_vocab_capacity(vocab) }
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
    let capacity = unsafe { ffi::iree_tokenizer_vocab_capacity(vocab) };
    if id as usize >= capacity {
        return None;
    }
    let view = unsafe { ffi::iree_tokenizer_vocab_token_text(vocab, id) };
    if view.data.is_null() || view.size == 0 {
        None
    } else {
        string_view_to_string(view).ok()
    }
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
    let state = EncodeStreamState::new(
        tokenizer.resource.clone(),
        options.add_special_tokens,
        options.max_chunk_bytes,
    )?;
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
    // Native encode should never need more than O(input bytes) token slots. If
    // it keeps returning RESOURCE_EXHAUSTED past this bound, surface the native
    // pending-state bug instead of doubling buffers until the BEAM is killed.
    let max_capacity = text.len().saturating_mul(8).saturating_add(1024).max(64);
    // We always tell IREE to populate token offsets even when the caller
    // did not request them, because `encoding_metadata` needs them to map
    // UNK token positions back to their source text slice. The offsets are
    // only surfaced to Elixir when `track_offsets` is true.
    let flags = encode_flags(add_special_tokens, true);

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
                    token_offsets: offsets.as_mut_ptr(),
                    type_ids: type_ids.as_mut_ptr(),
                },
                ffi::system_allocator(),
                &mut token_count,
            )
        };

        if is_resource_exhausted(status) {
            if capacity >= max_capacity {
                let native_error = check_status(status).err().unwrap_or_else(|| {
                    TokenizerError::new(
                        ErrorKind::ResourceExhausted,
                        "native encode exhausted output capacity",
                    )
                });
                return Err(TokenizerError::new(
                    native_error.kind,
                    format!(
                        "{}; bounded output capacity reached ({capacity} >= {max_capacity} token slots for {} input bytes)",
                        native_error.message,
                        text.len()
                    ),
                ));
            }
            unsafe { ffi::iree_status_ignore(status) };
            capacity = capacity
                .max(text.len() + 64)
                .saturating_mul(2)
                .min(max_capacity);
            continue;
        }

        check_status(status)?;

        // The underlying IREE tokenizer does not always return
        // RESOURCE_EXHAUSTED when the output buffer fills exactly: for
        // small-vocab / byte-fallback heavy inputs it stops at `capacity`
        // silently, leaving us with a prefix. Treat a full buffer as
        // suspected truncation and grow until we have headroom above the
        // worst-case upper bound of one token per input byte plus slack for
        // special tokens.
        if token_count == capacity && capacity <= text.len() + 16 {
            if capacity >= max_capacity {
                return Err(TokenizerError::new(
                    ErrorKind::ResourceExhausted,
                    format!(
                        "native encode filled output capacity beyond bounded limit ({capacity} >= {max_capacity} token slots for {} input bytes)",
                        text.len()
                    ),
                ));
            }
            capacity = capacity
                .max(text.len() + 64)
                .saturating_mul(2)
                .min(max_capacity);
            continue;
        }

        ids.truncate(token_count);
        type_ids.truncate(token_count);
        offsets.truncate(token_count);

        let (special_tokens_mask, tokens) = encoding_metadata(tokenizer, &ids, text, &offsets);

        let returned_offsets = if track_offsets {
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
            offsets: returned_offsets,
            attention_mask: vec![1; token_count],
            special_tokens_mask,
            tokens,
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
        return String::from_utf8(bytes).map_err(|err| {
            TokenizerError::new(
                ErrorKind::Internal,
                format!("invalid UTF-8 in decode output: {err}"),
            )
        });
    }
}

fn parse_encode_options(opts: Vec<EncodeOption>) -> ParsedEncodeOptions {
    let mut parsed = ParsedEncodeOptions {
        add_special_tokens: true,
        track_offsets: false,
        max_chunk_bytes: 65_536,
    };

    for opt in opts {
        match opt {
            EncodeOption::AddSpecialTokens(value) => parsed.add_special_tokens = value,
            EncodeOption::TrackOffsets(value) => parsed.track_offsets = value,
            EncodeOption::MaxChunkBytes(value) => parsed.max_chunk_bytes = value,
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

fn string_view_to_string(
    view: ffi::iree_string_view_t,
) -> std::result::Result<String, std::string::FromUtf8Error> {
    if view.data.is_null() || view.size == 0 {
        return Ok(String::new());
    }

    let bytes = unsafe { slice::from_raw_parts(view.data as *const u8, view.size) };
    String::from_utf8(bytes.to_vec())
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
    max_chunk_bytes: usize,
}

struct ParsedDecodeOptions {
    skip_special_tokens: bool,
}

pub(crate) fn invalid_argument(message: impl Into<String>) -> TokenizerError {
    TokenizerError::new(ErrorKind::InvalidArgument, message)
}

// Metaspace marker (Unicode U+2581 LOWER ONE EIGHTH BLOCK) used by
// SentencePiece-style tokenizers (Unigram + SentencePiece BPE) to mark word
// starts inside vocab token texts. UTF-8 encoding is 3 bytes.
const METASPACE_MARKER: &[u8] = "\u{2581}".as_bytes();

fn encoding_metadata(
    tokenizer: &TokenizerResource,
    ids: &[i32],
    text: &[u8],
    offsets: &[ffi::iree_tokenizer_offset_t],
) -> (Vec<u8>, Vec<String>) {
    let vocab = unsafe { ffi::iree_tokenizer_vocab(tokenizer.ptr) };
    let unk_id = unsafe { ffi::iree_tokenizer_vocab_special_ids(vocab) }.unk;

    // Pre-compute vocab token text for every id; we need it both for the
    // returned `tokens` field and for tracking metaspace overhead.
    let vocab_texts: Vec<String> = ids.iter().map(|&id| vocab_token_text(vocab, id)).collect();

    // The native IREE tokenizer (in this vendored bundle) writes per-token
    // offsets in the *transform buffer* — i.e. the post-normalizer string
    // with metaspace `▁` markers injected — not in the original input. To
    // match elixir-nx/tokenizers (which surfaces UNK substring text in the
    // `tokens` field for vocabularies without byte_fallback), we need to
    // map each UNK token's transform-buffer offset back to a slice of the
    // original input. We do that by walking the token sequence and
    // accumulating the byte difference between the transform buffer and
    // the original input that has been introduced by metaspace prefixes
    // up to (but not including) each token. Each metaspace prefix adds 3
    // bytes of `▁` to the transform; if the metaspace is at the start of
    // the input it consumes no source bytes (net +3), otherwise it
    // consumes one source space byte (net +2).
    let metaspace_offsets = compute_metaspace_overheads(&vocab_texts, offsets);

    let mut special_tokens_mask = Vec::with_capacity(ids.len());
    let mut tokens = Vec::with_capacity(ids.len());

    for (index, &id) in ids.iter().enumerate() {
        let attrs = unsafe { ffi::iree_tokenizer_vocab_token_attrs(vocab, id) };
        special_tokens_mask.push(u8::from(
            (attrs & ffi::IREE_TOKENIZER_TOKEN_ATTR_SPECIAL) != 0,
        ));

        // For the per-position `tokens` field: when the model has a UNK
        // token and the current id is UNK, return the source text slice
        // that produced it instead of the literal `<unk>` vocab string.
        // This makes encodings human-readable for vocabularies that do
        // not have byte-fallback coverage (e.g. T5 SentencePiece Unigram)
        // and matches elixir-nx/tokenizers byte-for-byte.
        let token = if unk_id >= 0 && id == unk_id {
            unk_source_slice(
                text,
                offsets.get(index),
                metaspace_offsets.get(index).copied(),
            )
            .unwrap_or_else(|| vocab_texts[index].clone())
        } else {
            vocab_texts[index].clone()
        };

        tokens.push(token);
    }

    (special_tokens_mask, tokens)
}

fn vocab_token_text(vocab: *const ffi::iree_tokenizer_vocab_t, id: i32) -> String {
    let view = unsafe { ffi::iree_tokenizer_vocab_token_text(vocab, id) };
    string_view_to_string(view).unwrap_or_default()
}

// For each token, returns the cumulative number of bytes the transform
// buffer has grown by relative to the original input *before* this token
// starts. Used to translate transform-buffer offsets back to original
// offsets. Only meaningful for SentencePiece-style metaspace tokenizers;
// for tokenizers that do not use the metaspace marker the result is all
// zeros, which leaves transform-buffer offsets unchanged.
fn compute_metaspace_overheads(
    vocab_texts: &[String],
    offsets: &[ffi::iree_tokenizer_offset_t],
) -> Vec<usize> {
    let mut result = Vec::with_capacity(vocab_texts.len());
    let mut overhead: usize = 0;
    let mut seen_metaspace = false;

    for (index, vocab_text) in vocab_texts.iter().enumerate() {
        result.push(overhead);

        if vocab_text.as_bytes().starts_with(METASPACE_MARKER) {
            // First metaspace token whose transform_start is at the very
            // beginning of the buffer corresponds to "start of input" (no
            // source space was consumed): the `▁` is purely additive, so
            // the buffer grew by 3 bytes. Every other metaspace replaces a
            // source space (1 byte) with a `▁` (3 bytes), so the buffer
            // grows by 2 bytes per occurrence.
            let at_input_start = !seen_metaspace
                && offsets
                    .get(index)
                    .map(|offset| offset.start == 0)
                    .unwrap_or(false);

            if at_input_start {
                overhead += METASPACE_MARKER.len();
            } else {
                overhead += METASPACE_MARKER.len() - 1;
            }
            seen_metaspace = true;
        }
    }

    result
}

fn unk_source_slice(
    text: &[u8],
    offset: Option<&ffi::iree_tokenizer_offset_t>,
    metaspace_overhead: Option<usize>,
) -> Option<String> {
    let offset = offset?;
    let overhead = metaspace_overhead?;

    if offset.end <= offset.start {
        return None;
    }

    let original_start = offset.start.checked_sub(overhead)?;
    let original_end = offset.end.checked_sub(overhead)?;
    if original_start >= original_end || original_end > text.len() {
        return None;
    }

    Some(String::from_utf8_lossy(&text[original_start..original_end]).into_owned())
}

#[derive(Clone, Copy, Default)]
pub(crate) struct TokenizerMetadata {
    pub(crate) decode_strategy: DecodeStrategy,
    pub(crate) stream_encode_strategy: StreamEncodeStrategy,
}

impl Default for DecodeStrategy {
    fn default() -> Self {
        Self::Native
    }
}

impl Default for StreamEncodeStrategy {
    fn default() -> Self {
        Self::Native
    }
}

fn tokenizer_from_raw(
    raw: *mut ffi::iree_tokenizer_t,
    metadata: TokenizerMetadata,
) -> Result<Tokenizer> {
    let model_type = string_view_to_string(unsafe { ffi::iree_tokenizer_model_type_name(raw) })
        .map_err(|err| {
            TokenizerError::new(
                ErrorKind::Internal,
                format!("invalid UTF-8 in tokenizer metadata: {err}"),
            )
        })?;

    Ok(Tokenizer {
        resource: ResourceArc::new(TokenizerResource {
            ptr: raw,
            model_type,
            decode_strategy: metadata.decode_strategy,
            stream_encode_strategy: metadata.stream_encode_strategy,
        }),
    })
}

fn tiktoken_config_by_name(encoding: &str) -> Result<*const ffi::iree_tokenizer_tiktoken_config_t> {
    let config = unsafe {
        ffi::iree_tokenizer_tiktoken_config_by_name(ffi::make_string_view(encoding.as_bytes()))
    };

    if config.is_null() {
        Err(invalid_argument(format!(
            "unknown tiktoken encoding {encoding:?}; supported encodings: {}",
            SUPPORTED_TIKTOKEN_ENCODINGS.join(", ")
        )))
    } else {
        Ok(config)
    }
}

pub(crate) fn tokenizer_metadata_from_hf_json(json: &[u8]) -> TokenizerMetadata {
    let Ok(root) = serde_json::from_slice::<Value>(json) else {
        return TokenizerMetadata::default();
    };

    let decode_strategy = if is_sentencepiece_bpe_decoder(&root) {
        DecodeStrategy::SentencePieceBpe
    } else {
        DecodeStrategy::Native
    };

    let stream_encode_strategy = infer_stream_encode_strategy(&root);

    TokenizerMetadata {
        decode_strategy,
        stream_encode_strategy,
    }
}

fn infer_stream_encode_strategy(root: &Value) -> StreamEncodeStrategy {
    let model_type = root
        .get("model")
        .and_then(|model| model.get("type"))
        .and_then(Value::as_str);

    match model_type {
        Some("Unigram") => StreamEncodeStrategy::BufferedFinalize,
        Some("BPE")
            if root.get("pre_tokenizer").is_none()
                || root.get("pre_tokenizer") == Some(&Value::Null)
                || is_sentencepiece_bpe_decoder(root)
                || is_metaspace_byte_fallback_bpe_decoder(root) =>
        {
            StreamEncodeStrategy::BufferedFinalize
        }
        _ => StreamEncodeStrategy::Native,
    }
}

fn is_sentencepiece_bpe_decoder(root: &Value) -> bool {
    let model_type = root
        .get("model")
        .and_then(|model| model.get("type"))
        .and_then(Value::as_str);

    if model_type != Some("BPE") {
        return false;
    }

    let Some(decoder) = root.get("decoder") else {
        return false;
    };

    let Some(decoders) = decoder
        .as_object()
        .filter(|object| object.get("type").and_then(Value::as_str) == Some("Sequence"))
        .and_then(|object| object.get("decoders"))
        .and_then(Value::as_array)
    else {
        return false;
    };

    let has_replace = decoders.iter().any(is_sentencepiece_replace_decoder);
    let has_fuse = decoders
        .iter()
        .any(|decoder| decoder_type(decoder) == Some("Fuse"));
    let has_strip = decoders.iter().any(is_sentencepiece_strip_decoder);

    has_replace && has_fuse && has_strip
}

fn is_metaspace_byte_fallback_bpe_decoder(root: &Value) -> bool {
    let model_type = root
        .get("model")
        .and_then(|model| model.get("type"))
        .and_then(Value::as_str);

    if model_type != Some("BPE") {
        return false;
    }

    let Some(decoder) = root.get("decoder") else {
        return false;
    };

    let Some(decoders) = decoder
        .as_object()
        .filter(|object| object.get("type").and_then(Value::as_str) == Some("Sequence"))
        .and_then(|object| object.get("decoders"))
        .and_then(Value::as_array)
    else {
        return false;
    };

    let has_replace = decoders.iter().any(is_sentencepiece_replace_decoder);
    let has_byte_fallback = decoders
        .iter()
        .any(|decoder| decoder_type(decoder) == Some("ByteFallback"));
    let has_fuse = decoders
        .iter()
        .any(|decoder| decoder_type(decoder) == Some("Fuse"));

    has_replace && has_byte_fallback && has_fuse
}

fn decoder_type(value: &Value) -> Option<&str> {
    value.as_object()?.get("type")?.as_str()
}

fn is_sentencepiece_replace_decoder(value: &Value) -> bool {
    let Some(object) = value.as_object() else {
        return false;
    };

    if object.get("type").and_then(Value::as_str) != Some("Replace") {
        return false;
    }

    object
        .get("pattern")
        .and_then(Value::as_object)
        .and_then(|pattern| pattern.get("String"))
        .and_then(Value::as_str)
        == Some("▁")
        && object.get("content").and_then(Value::as_str) == Some(" ")
}

fn is_sentencepiece_strip_decoder(value: &Value) -> bool {
    let Some(object) = value.as_object() else {
        return false;
    };

    object.get("type").and_then(Value::as_str) == Some("Strip")
        && object.get("content").and_then(Value::as_str) == Some(" ")
        && object.get("start").and_then(Value::as_i64) == Some(1)
        && object.get("stop").and_then(Value::as_i64) == Some(0)
}

pub(crate) fn decode_sentencepiece_bpe(
    tokenizer: &TokenizerResource,
    ids: &[i32],
    skip_special_tokens: bool,
) -> Result<String> {
    let vocab = unsafe { ffi::iree_tokenizer_vocab(tokenizer.ptr) };
    let mut output = String::new();
    let mut pending_bytes = Vec::new();

    for &id in ids {
        if id < 0 {
            continue;
        }

        let attrs = unsafe { ffi::iree_tokenizer_vocab_token_attrs(vocab, id) };
        let is_special = (attrs & ffi::IREE_TOKENIZER_TOKEN_ATTR_SPECIAL) != 0;
        if skip_special_tokens && is_special {
            continue;
        }

        let token =
            string_view_to_string(unsafe { ffi::iree_tokenizer_vocab_token_text(vocab, id) })
                .map_err(|err| {
                    TokenizerError::new(
                        ErrorKind::Internal,
                        format!("invalid UTF-8 in tokenizer metadata: {err}"),
                    )
                })?;

        if let Some(byte) = parse_byte_fallback_token(&token) {
            pending_bytes.push(byte);
            continue;
        }

        flush_pending_bytes(&mut output, &mut pending_bytes);
        output.push_str(&token.replace('▁', " "));
    }

    flush_pending_bytes(&mut output, &mut pending_bytes);

    if output.starts_with(' ') {
        output.remove(0);
    }

    Ok(output)
}

fn parse_byte_fallback_token(token: &str) -> Option<u8> {
    if token.len() != 6 || !token.starts_with("<0x") || !token.ends_with('>') {
        return None;
    }

    u8::from_str_radix(&token[3..5], 16).ok()
}

fn flush_pending_bytes(output: &mut String, pending_bytes: &mut Vec<u8>) {
    if pending_bytes.is_empty() {
        return;
    }

    output.push_str(&String::from_utf8_lossy(pending_bytes));
    pending_bytes.clear();
}
