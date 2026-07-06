// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/segmenter/thousands.h"

#include <string.h>

typedef struct iree_tokenizer_segmenter_thousands_t {
  iree_tokenizer_segmenter_t base;
  iree_allocator_t allocator;
} iree_tokenizer_segmenter_thousands_t;

typedef struct iree_tokenizer_segmenter_thousands_state_t {
  iree_tokenizer_segmenter_state_t base;
  iree_host_size_t bytes_processed;
  iree_host_size_t last_emit_end;
  bool has_pending;
} iree_tokenizer_segmenter_thousands_state_t;

static const iree_tokenizer_segmenter_vtable_t
    iree_tokenizer_segmenter_thousands_vtable;

static inline bool iree_tokenizer_thousands_is_digit(uint8_t c) {
  return c >= '0' && c <= '9';
}

static inline bool iree_tokenizer_thousands_is_word(uint8_t c) {
  return (c >= '0' && c <= '9') || (c >= 'A' && c <= 'Z') ||
         (c >= 'a' && c <= 'z') || c == '_';
}

static inline bool iree_tokenizer_thousands_emit(
    iree_tokenizer_segment_output_t output, iree_host_size_t chunk_base,
    iree_host_size_t* count, iree_host_size_t start, iree_host_size_t end) {
  if (start >= end) return true;
  if (*count >= output.capacity) return false;
  output.values[*count].start = start - chunk_base;
  output.values[*count].end = end - chunk_base;
  ++*count;
  return true;
}

iree_status_t iree_tokenizer_segmenter_thousands_allocate(
    iree_allocator_t allocator, iree_tokenizer_segmenter_t** out_segmenter) {
  IREE_ASSERT_ARGUMENT(out_segmenter);
  *out_segmenter = NULL;

  iree_tokenizer_segmenter_thousands_t* segmenter = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, sizeof(*segmenter), (void**)&segmenter));

  iree_tokenizer_segmenter_initialize(
      &segmenter->base, &iree_tokenizer_segmenter_thousands_vtable,
      sizeof(iree_tokenizer_segmenter_thousands_state_t));
  segmenter->allocator = allocator;
  *out_segmenter = &segmenter->base;
  return iree_ok_status();
}

static void iree_tokenizer_segmenter_thousands_destroy(
    iree_tokenizer_segmenter_t* segmenter) {
  iree_tokenizer_segmenter_thousands_t* self =
      (iree_tokenizer_segmenter_thousands_t*)segmenter;
  iree_allocator_free(self->allocator, self);
}

static iree_status_t iree_tokenizer_segmenter_thousands_state_initialize(
    const iree_tokenizer_segmenter_t* segmenter, void* storage,
    iree_tokenizer_segmenter_state_t** out_state) {
  iree_tokenizer_segmenter_thousands_state_t* state =
      (iree_tokenizer_segmenter_thousands_state_t*)storage;
  memset(state, 0, sizeof(*state));
  state->base.segmenter = segmenter;
  *out_state = &state->base;
  return iree_ok_status();
}

static void iree_tokenizer_segmenter_thousands_state_deinitialize(
    iree_tokenizer_segmenter_state_t* state) {
  (void)state;
}

static iree_status_t iree_tokenizer_segmenter_thousands_state_process(
    iree_tokenizer_segmenter_state_t* state, iree_string_view_t input,
    iree_tokenizer_segment_output_t output, iree_host_size_t* out_consumed,
    iree_host_size_t* out_segment_count) {
  iree_tokenizer_segmenter_thousands_state_t* self =
      (iree_tokenizer_segmenter_thousands_state_t*)state;
  *out_consumed = 0;
  *out_segment_count = 0;
  if (input.size == 0 || output.capacity == 0) return iree_ok_status();

  iree_host_size_t chunk_base = self->bytes_processed;
  iree_host_size_t count = 0;
  iree_host_size_t last_emit_end = self->last_emit_end;

  for (iree_host_size_t i = 0; i < input.size;) {
    if (!iree_tokenizer_thousands_is_digit((uint8_t)input.data[i])) {
      ++i;
      continue;
    }

    iree_host_size_t run_start = i;
    while (i < input.size &&
           iree_tokenizer_thousands_is_digit((uint8_t)input.data[i])) {
      ++i;
    }
    iree_host_size_t run_end = i;

    // The regex requires a word boundary after the digit run. Treat the
    // current chunk end as a boundary so process() can make progress; streaming
    // callers that require cross-chunk number grouping use the higher-level
    // buffered path.
    if (run_end < input.size &&
        iree_tokenizer_thousands_is_word((uint8_t)input.data[run_end])) {
      continue;
    }

    iree_host_size_t abs_run_start = chunk_base + run_start;
    iree_host_size_t abs_run_end = chunk_base + run_end;
    if (!iree_tokenizer_thousands_emit(output, chunk_base, &count,
                                       last_emit_end, abs_run_start)) {
      break;
    }
    if (last_emit_end < abs_run_start) last_emit_end = abs_run_start;

    iree_host_size_t run_len = run_end - run_start;
    iree_host_size_t first_len = run_len % 3;
    if (first_len == 0) first_len = 3;
    bool full = false;
    for (iree_host_size_t group_start = run_start; group_start < run_end;) {
      iree_host_size_t group_len =
          (group_start == run_start) ? first_len : 3;
      iree_host_size_t group_end = group_start + group_len;
      if (!iree_tokenizer_thousands_emit(output, chunk_base, &count,
                                         chunk_base + group_start,
                                         chunk_base + group_end)) {
        full = true;
        break;
      }
      last_emit_end = chunk_base + group_end;
      group_start = group_end;
    }
    if (full) break;
    last_emit_end = abs_run_end;
  }

  iree_host_size_t input_end = chunk_base + input.size;
  if (iree_tokenizer_thousands_emit(output, chunk_base, &count, last_emit_end,
                                    input_end)) {
    last_emit_end = input_end;
  }
  *out_consumed = last_emit_end - chunk_base;
  *out_segment_count = count;
  self->bytes_processed = last_emit_end;
  self->last_emit_end = last_emit_end;
  self->has_pending = false;
  return iree_ok_status();
}

static iree_status_t iree_tokenizer_segmenter_thousands_state_finalize(
    iree_tokenizer_segmenter_state_t* state, iree_string_view_t input,
    iree_tokenizer_segment_output_t output,
    iree_host_size_t* out_segment_count) {
  iree_tokenizer_segmenter_thousands_state_t* self =
      (iree_tokenizer_segmenter_thousands_state_t*)state;
  *out_segment_count = 0;
  if (output.capacity == 0) return iree_ok_status();

  iree_host_size_t chunk_base = self->bytes_processed;
  iree_host_size_t count = 0;
  iree_host_size_t last_emit_end = self->last_emit_end;

  for (iree_host_size_t i = 0; i < input.size;) {
    if (!iree_tokenizer_thousands_is_digit((uint8_t)input.data[i])) {
      ++i;
      continue;
    }

    iree_host_size_t run_start = i;
    while (i < input.size &&
           iree_tokenizer_thousands_is_digit((uint8_t)input.data[i])) {
      ++i;
    }
    iree_host_size_t run_end = i;
    if (run_end < input.size &&
        iree_tokenizer_thousands_is_word((uint8_t)input.data[run_end])) {
      continue;
    }

    iree_host_size_t abs_run_start = chunk_base + run_start;
    iree_host_size_t abs_run_end = chunk_base + run_end;
    if (!iree_tokenizer_thousands_emit(output, chunk_base, &count,
                                       last_emit_end, abs_run_start)) {
      break;
    }

    iree_host_size_t run_len = run_end - run_start;
    iree_host_size_t first_len = run_len % 3;
    if (first_len == 0) first_len = 3;
    for (iree_host_size_t group_start = run_start; group_start < run_end;) {
      iree_host_size_t group_len =
          (group_start == run_start) ? first_len : 3;
      iree_host_size_t group_end = group_start + group_len;
      if (!iree_tokenizer_thousands_emit(output, chunk_base, &count,
                                         chunk_base + group_start,
                                         chunk_base + group_end)) {
        *out_segment_count = count;
        return iree_ok_status();
      }
      group_start = group_end;
    }
    last_emit_end = abs_run_end;
  }

  iree_host_size_t input_end = chunk_base + input.size;
  iree_tokenizer_thousands_emit(output, chunk_base, &count, last_emit_end,
                                input_end);
  self->bytes_processed = input_end;
  self->last_emit_end = input_end;
  self->has_pending = false;
  *out_segment_count = count;
  return iree_ok_status();
}

static bool iree_tokenizer_segmenter_thousands_state_has_pending(
    const iree_tokenizer_segmenter_state_t* state) {
  const iree_tokenizer_segmenter_thousands_state_t* self =
      (const iree_tokenizer_segmenter_thousands_state_t*)state;
  return self->has_pending;
}

static const iree_tokenizer_segmenter_vtable_t
    iree_tokenizer_segmenter_thousands_vtable = {
        .destroy = iree_tokenizer_segmenter_thousands_destroy,
        .state_initialize = iree_tokenizer_segmenter_thousands_state_initialize,
        .state_deinitialize =
            iree_tokenizer_segmenter_thousands_state_deinitialize,
        .state_process = iree_tokenizer_segmenter_thousands_state_process,
        .state_finalize = iree_tokenizer_segmenter_thousands_state_finalize,
        .state_has_pending =
            iree_tokenizer_segmenter_thousands_state_has_pending,
};
