#![allow(
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals,
    dead_code
)]

use std::ffi::{c_char, c_void};

pub const IREE_STATUS_CODE_MASK: u32 = 0x1F;
pub const IREE_TOKENIZER_TRANSFORM_BUFFER_MAX_SIZE: usize = 16 * 1024;
pub const IREE_TOKENIZER_TRANSFORM_BUFFER_EXPANSION_FACTOR: usize = 3;
pub const IREE_TOKENIZER_TRANSFORM_BUFFER_MIN_SIZE: usize = 4096;
pub const IREE_TOKENIZER_DECODE_OUTPUT_RECOMMENDED_SIZE: usize = 2048;
pub const IREE_TOKENIZER_TOKEN_ATTR_SPECIAL: u16 = 1u16 << 3;

#[repr(C)]
#[derive(Clone, Copy)]
pub enum iree_status_code_e {
    IREE_STATUS_OK = 0,
    IREE_STATUS_CANCELLED = 1,
    IREE_STATUS_UNKNOWN = 2,
    IREE_STATUS_INVALID_ARGUMENT = 3,
    IREE_STATUS_DEADLINE_EXCEEDED = 4,
    IREE_STATUS_NOT_FOUND = 5,
    IREE_STATUS_ALREADY_EXISTS = 6,
    IREE_STATUS_PERMISSION_DENIED = 7,
    IREE_STATUS_RESOURCE_EXHAUSTED = 8,
    IREE_STATUS_FAILED_PRECONDITION = 9,
    IREE_STATUS_ABORTED = 10,
    IREE_STATUS_OUT_OF_RANGE = 11,
    IREE_STATUS_UNIMPLEMENTED = 12,
    IREE_STATUS_INTERNAL = 13,
    IREE_STATUS_UNAVAILABLE = 14,
    IREE_STATUS_DATA_LOSS = 15,
    IREE_STATUS_UNAUTHENTICATED = 16,
    IREE_STATUS_DEFERRED = 17,
    IREE_STATUS_INCOMPATIBLE = 18,
}

#[repr(C)]
pub struct iree_status_handle_t {
    _private: [u8; 0],
}

pub type iree_status_t = *mut iree_status_handle_t;
pub type iree_tokenizer_encode_flags_t = u32;
pub type iree_tokenizer_decode_flags_t = u32;

#[repr(C)]
pub struct iree_allocator_t {
    pub self_: *mut c_void,
    pub ctl: Option<
        unsafe extern "C" fn(
            self_: *mut c_void,
            command: iree_allocator_command_t,
            params: *const c_void,
            inout_ptr: *mut *mut c_void,
        ) -> iree_status_t,
    >,
}

pub type iree_allocator_command_t = u32;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct iree_string_view_t {
    pub data: *const c_char,
    pub size: usize,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct iree_mutable_string_view_t {
    pub data: *mut c_char,
    pub size: usize,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct iree_byte_span_t {
    pub data: *mut u8,
    pub data_length: usize,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct iree_const_byte_span_t {
    pub data: *const u8,
    pub data_length: usize,
}

#[repr(C)]
pub struct iree_tokenizer_t {
    _private: [u8; 0],
}

#[repr(C)]
pub struct iree_tokenizer_encode_state_t {
    _private: [u8; 0],
}

#[repr(C)]
pub struct iree_tokenizer_decode_state_t {
    _private: [u8; 0],
}

#[repr(C)]
pub struct iree_tokenizer_vocab_t {
    _private: [u8; 0],
}

#[repr(C)]
pub struct iree_tokenizer_tiktoken_config_t {
    _private: [u8; 0],
}

pub type iree_tokenizer_token_id_t = i32;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct iree_tokenizer_token_id_list_t {
    pub count: usize,
    pub values: *const iree_tokenizer_token_id_t,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct iree_tokenizer_offset_t {
    pub start: usize,
    pub end: usize,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct iree_tokenizer_token_output_t {
    pub capacity: usize,
    pub token_ids: *mut iree_tokenizer_token_id_t,
    pub token_offsets: *mut iree_tokenizer_offset_t,
    pub type_ids: *mut u8,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct iree_tokenizer_offset_run_t {
    pub transform_position: usize,
    pub original_offset: usize,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct iree_tokenizer_offset_run_list_t {
    pub capacity: usize,
    pub values: *mut iree_tokenizer_offset_run_t,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct iree_tokenizer_encode_batch_item_t {
    pub text: iree_string_view_t,
    pub output: iree_tokenizer_token_output_t,
    pub out_token_count: usize,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct iree_tokenizer_decode_batch_item_t {
    pub tokens: iree_tokenizer_token_id_list_t,
    pub text_output: iree_mutable_string_view_t,
    pub out_text_length: usize,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct iree_tokenizer_special_ids_t {
    pub bos: i32,
    pub eos: i32,
    pub unk: i32,
    pub pad: i32,
    pub sep: i32,
    pub cls: i32,
    pub mask: i32,
}

pub mod encode_flag_bits {
    pub const NONE: u32 = 0;
    pub const AT_INPUT_START: u32 = 1 << 0;
    pub const TRACK_OFFSETS: u32 = 1 << 1;
    pub const ADD_SPECIAL_TOKENS: u32 = 1 << 2;
    pub const NO_SPECIAL_TOKEN_MATCHING: u32 = 1 << 3;
}

pub mod decode_flag_bits {
    pub const NONE: u32 = 0;
    pub const SKIP_SPECIAL_TOKENS: u32 = 1 << 0;
}

unsafe extern "C" {
    pub fn iree_allocator_libc_ctl(
        self_: *mut c_void,
        command: iree_allocator_command_t,
        params: *const c_void,
        inout_ptr: *mut *mut c_void,
    ) -> iree_status_t;

    pub fn iree_status_code_string(code: u32) -> *const c_char;
    pub fn iree_status_ignore(status: iree_status_t) -> iree_status_t;
    pub fn iree_status_consume_code(status: iree_status_t) -> u32;
    pub fn iree_status_to_string(
        status: iree_status_t,
        allocator: *const iree_allocator_t,
        out_buffer: *mut *mut c_char,
        out_buffer_length: *mut usize,
    ) -> bool;
    pub fn iree_allocator_free(allocator: iree_allocator_t, ptr: *mut c_void);

    pub fn iree_tokenizer_from_huggingface_json(
        json: iree_string_view_t,
        allocator: iree_allocator_t,
        out_tokenizer: *mut *mut iree_tokenizer_t,
    ) -> iree_status_t;

    pub fn iree_tokenizer_tiktoken_config_by_name(
        name: iree_string_view_t,
    ) -> *const iree_tokenizer_tiktoken_config_t;

    pub fn iree_tokenizer_from_tiktoken(
        data: iree_string_view_t,
        config: *const iree_tokenizer_tiktoken_config_t,
        allocator: iree_allocator_t,
        out_tokenizer: *mut *mut iree_tokenizer_t,
    ) -> iree_status_t;

    pub fn iree_tokenizer_free(tokenizer: *mut iree_tokenizer_t);
    pub fn iree_tokenizer_vocab(
        tokenizer: *const iree_tokenizer_t,
    ) -> *const iree_tokenizer_vocab_t;
    pub fn iree_tokenizer_model_type_name(tokenizer: *const iree_tokenizer_t)
        -> iree_string_view_t;

    pub fn iree_tokenizer_encode(
        tokenizer: *const iree_tokenizer_t,
        text: iree_string_view_t,
        flags: iree_tokenizer_encode_flags_t,
        output: iree_tokenizer_token_output_t,
        allocator: iree_allocator_t,
        out_token_count: *mut usize,
    ) -> iree_status_t;

    pub fn iree_tokenizer_decode(
        tokenizer: *const iree_tokenizer_t,
        tokens: iree_tokenizer_token_id_list_t,
        flags: iree_tokenizer_decode_flags_t,
        text_output: iree_mutable_string_view_t,
        allocator: iree_allocator_t,
        out_text_length: *mut usize,
    ) -> iree_status_t;

    pub fn iree_tokenizer_encode_batch(
        tokenizer: *const iree_tokenizer_t,
        items: *mut iree_tokenizer_encode_batch_item_t,
        item_count: usize,
        flags: iree_tokenizer_encode_flags_t,
        state_storage: iree_byte_span_t,
        transform_buffer: iree_byte_span_t,
        offset_runs: iree_tokenizer_offset_run_list_t,
    ) -> iree_status_t;

    pub fn iree_tokenizer_decode_batch(
        tokenizer: *const iree_tokenizer_t,
        items: *mut iree_tokenizer_decode_batch_item_t,
        item_count: usize,
        flags: iree_tokenizer_decode_flags_t,
        state_storage: iree_byte_span_t,
    ) -> iree_status_t;

    pub fn iree_tokenizer_encode_state_calculate_size(
        tokenizer: *const iree_tokenizer_t,
        out_size: *mut usize,
    ) -> iree_status_t;

    pub fn iree_tokenizer_decode_state_calculate_size(
        tokenizer: *const iree_tokenizer_t,
        out_size: *mut usize,
    ) -> iree_status_t;

    pub fn iree_tokenizer_encode_state_initialize(
        tokenizer: *const iree_tokenizer_t,
        state_storage: iree_byte_span_t,
        transform_buffer: iree_byte_span_t,
        offset_runs: iree_tokenizer_offset_run_list_t,
        flags: iree_tokenizer_encode_flags_t,
        out_state: *mut *mut iree_tokenizer_encode_state_t,
    ) -> iree_status_t;

    pub fn iree_tokenizer_encode_state_deinitialize(state: *mut iree_tokenizer_encode_state_t);

    pub fn iree_tokenizer_encode_state_feed(
        state: *mut iree_tokenizer_encode_state_t,
        chunk: iree_string_view_t,
        output: iree_tokenizer_token_output_t,
        out_bytes_consumed: *mut usize,
        out_token_count: *mut usize,
    ) -> iree_status_t;

    pub fn iree_tokenizer_encode_state_finalize(
        state: *mut iree_tokenizer_encode_state_t,
        output: iree_tokenizer_token_output_t,
        out_token_count: *mut usize,
    ) -> iree_status_t;

    pub fn iree_tokenizer_encode_state_has_pending(
        state: *const iree_tokenizer_encode_state_t,
    ) -> bool;

    pub fn iree_tokenizer_decode_state_initialize(
        tokenizer: *const iree_tokenizer_t,
        flags: iree_tokenizer_decode_flags_t,
        state_storage: iree_byte_span_t,
        out_state: *mut *mut iree_tokenizer_decode_state_t,
    ) -> iree_status_t;

    pub fn iree_tokenizer_decode_state_deinitialize(state: *mut iree_tokenizer_decode_state_t);

    pub fn iree_tokenizer_decode_state_feed(
        state: *mut iree_tokenizer_decode_state_t,
        tokens: iree_tokenizer_token_id_list_t,
        text_output: iree_mutable_string_view_t,
        out_tokens_consumed: *mut usize,
        out_text_length: *mut usize,
    ) -> iree_status_t;

    pub fn iree_tokenizer_decode_state_finalize(
        state: *mut iree_tokenizer_decode_state_t,
        text_output: iree_mutable_string_view_t,
        out_text_length: *mut usize,
    ) -> iree_status_t;

    pub fn iree_tokenizer_vocab_lookup(
        vocab: *const iree_tokenizer_vocab_t,
        text: iree_string_view_t,
    ) -> i32;

    pub fn iree_tokenizer_vocab_token_text(
        vocab: *const iree_tokenizer_vocab_t,
        token_id: i32,
    ) -> iree_string_view_t;

    pub fn iree_tokenizer_vocab_token_attrs(
        vocab: *const iree_tokenizer_vocab_t,
        token_id: i32,
    ) -> u16;

    pub fn iree_tokenizer_vocab_capacity(vocab: *const iree_tokenizer_vocab_t) -> usize;
    pub fn iree_tokenizer_vocab_token_count(vocab: *const iree_tokenizer_vocab_t) -> usize;
    pub fn iree_tokenizer_vocab_special_ids(
        vocab: *const iree_tokenizer_vocab_t,
    ) -> iree_tokenizer_special_ids_t;
}

pub fn system_allocator() -> iree_allocator_t {
    iree_allocator_t {
        self_: std::ptr::null_mut(),
        ctl: Some(iree_allocator_libc_ctl),
    }
}

pub fn make_string_view(bytes: &[u8]) -> iree_string_view_t {
    iree_string_view_t {
        data: bytes.as_ptr() as *const c_char,
        size: bytes.len(),
    }
}

pub fn make_mutable_string_view(bytes: &mut [u8]) -> iree_mutable_string_view_t {
    iree_mutable_string_view_t {
        data: bytes.as_mut_ptr() as *mut c_char,
        size: bytes.len(),
    }
}

pub fn make_byte_span(bytes: &mut [u8]) -> iree_byte_span_t {
    iree_byte_span_t {
        data: bytes.as_mut_ptr(),
        data_length: bytes.len(),
    }
}

pub fn empty_offset_runs() -> iree_tokenizer_offset_run_list_t {
    iree_tokenizer_offset_run_list_t {
        capacity: 0,
        values: std::ptr::null_mut(),
    }
}

pub fn transform_buffer_recommended_size(text_size: usize) -> usize {
    let expanded = if text_size
        <= IREE_TOKENIZER_TRANSFORM_BUFFER_MAX_SIZE
            / IREE_TOKENIZER_TRANSFORM_BUFFER_EXPANSION_FACTOR
    {
        text_size * IREE_TOKENIZER_TRANSFORM_BUFFER_EXPANSION_FACTOR
    } else {
        IREE_TOKENIZER_TRANSFORM_BUFFER_MAX_SIZE
    };

    expanded
        .max(IREE_TOKENIZER_TRANSFORM_BUFFER_MIN_SIZE)
        .next_power_of_two()
        .min(IREE_TOKENIZER_TRANSFORM_BUFFER_MAX_SIZE)
}

pub fn transform_buffer_oneshot_size(text_size: usize) -> usize {
    text_size
        .saturating_mul(IREE_TOKENIZER_TRANSFORM_BUFFER_EXPANSION_FACTOR)
        .max(IREE_TOKENIZER_TRANSFORM_BUFFER_MIN_SIZE)
        .next_power_of_two()
        .min(IREE_TOKENIZER_TRANSFORM_BUFFER_MAX_SIZE)
}
