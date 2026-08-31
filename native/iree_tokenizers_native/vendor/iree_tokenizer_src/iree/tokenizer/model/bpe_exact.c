// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Exact complete-segment BPE using a global rank heap.
//
// Canonical BPE is not online: a lower-rank merge discovered arbitrarily far
// to the right can change which overlapping merges remain valid to its left.
// For the narrow loader-verified shape, this path buffers one complete
// segment, seeds its base tokens, and applies every merge from one global
// min-heap. Scratch grows with the largest segment seen by the state and is
// reused across later segments.

#include "iree/tokenizer/model/bpe_internal.h"

#include "iree/base/internal/math.h"

bool iree_tokenizer_bpe_exact_can_encode(
    const iree_tokenizer_bpe_model_t* model, iree_string_view_t segment) {
  iree_host_size_t i = 0;
  while (i < segment.size) {
    uint8_t byte = (uint8_t)segment.data[i];
    if (model->byte_to_token[byte] < 0) return false;

    if (byte == '\t' || byte == '\n' || byte == '\r') {
      // Legacy Split+ByteLevel tokenizers may carry literal homogeneous runs
      // such as "\n\n". Any raw run token makes mapped one-node-per-byte
      // seeding ambiguous, so keep the whole segment on the legacy path.
      iree_host_size_t run_end = i + 1;
      while (run_end < segment.size &&
             (uint8_t)segment.data[run_end] == byte) {
        ++run_end;
      }
      iree_host_size_t trie_end = run_end;
      if (trie_end - i > model->max_token_length) {
        trie_end = i + model->max_token_length;
      }
      iree_tokenizer_trie_cursor_t cursor;
      iree_tokenizer_trie_cursor_reset(&cursor, model->trie);
      for (iree_host_size_t j = i; j < trie_end; ++j) {
        if (!iree_tokenizer_trie_cursor_advance(&cursor, byte)) break;
        if (iree_tokenizer_trie_cursor_token_id(&cursor) >= 0) return false;
      }
      i = run_end;
    } else {
      ++i;
    }
  }
  return true;
}

static iree_status_t iree_tokenizer_bpe_exact_reserve(
    const iree_tokenizer_bpe_model_t* model, iree_tokenizer_bpe_state_t* state,
    iree_host_size_t segment_size) {
  if (segment_size >= UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "exact BPE segment too large for 32-bit token offsets: %" PRIhsz,
        segment_size);
  }

  if (segment_size > state->exact.node_capacity) {
    void* nodes = state->exact.nodes;
    IREE_RETURN_IF_ERROR(iree_allocator_realloc_array(
        model->allocator, segment_size,
        sizeof(iree_tokenizer_bpe_exact_node_t), &nodes));
    state->exact.nodes = (iree_tokenizer_bpe_exact_node_t*)nodes;
    state->exact.node_capacity = segment_size;
  }

  iree_host_size_t required_heap_capacity = 0;
  if (!iree_host_size_checked_mul(2, segment_size,
                                  &required_heap_capacity)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "exact BPE heap capacity overflow for %" PRIhsz
                            " input bytes",
                            segment_size);
  }
  if (required_heap_capacity < 2) required_heap_capacity = 2;

  if (required_heap_capacity > state->exact.heap_capacity) {
    void* heap_entries = state->exact.heap_entries;
    IREE_RETURN_IF_ERROR(iree_allocator_realloc_array(
        model->allocator, required_heap_capacity,
        sizeof(iree_tokenizer_bpe_heap_entry_t), &heap_entries));
    state->exact.heap_entries =
        (iree_tokenizer_bpe_heap_entry_t*)heap_entries;
    state->exact.heap_capacity = required_heap_capacity;
  }

  return iree_ok_status();
}

static void iree_tokenizer_bpe_exact_append_node(
    iree_tokenizer_bpe_state_t* state, int32_t token_id, uint32_t start_byte,
    uint32_t end_byte) {
  uint32_t index = (uint32_t)state->exact.node_count++;
  iree_tokenizer_bpe_exact_node_t* node = &state->exact.nodes[index];
  node->token_id = token_id;
  node->start_byte = start_byte;
  node->end_byte = end_byte;
  node->prev = state->exact.tail;
  node->next = IREE_TOKENIZER_BPE_EXACT_INVALID_INDEX;

  if (state->exact.tail != IREE_TOKENIZER_BPE_EXACT_INVALID_INDEX) {
    state->exact.nodes[state->exact.tail].next = index;
  } else {
    state->exact.head = index;
  }
  state->exact.tail = index;
}

static void iree_tokenizer_bpe_exact_maybe_add_merge(
    const iree_tokenizer_bpe_model_t* model,
    iree_tokenizer_bpe_exact_state_t* exact, iree_tokenizer_bpe_heap_t* heap,
    uint32_t left_index) {
  if (left_index == IREE_TOKENIZER_BPE_EXACT_INVALID_INDEX ||
      left_index >= exact->node_count) {
    return;
  }

  iree_tokenizer_bpe_exact_node_t* left = &exact->nodes[left_index];
  if (left->token_id < 0 ||
      left->next == IREE_TOKENIZER_BPE_EXACT_INVALID_INDEX) {
    return;
  }
  iree_tokenizer_bpe_exact_node_t* right = &exact->nodes[left->next];
  if (right->token_id < 0) return;

  iree_tokenizer_merge_hash_result_t merge =
      iree_tokenizer_vocab_merge_hash_lookup(model->merge_hash, left->token_id,
                                             right->token_id);
  if (iree_tokenizer_merge_hash_result_is_valid(merge)) {
    // Initial node indices increase with byte position and merged nodes retain
    // their left index, so this preserves rank + leftmost tie-breaking.
    iree_tokenizer_bpe_heap_entry_t entry = {merge.rank, left_index};
    iree_tokenizer_bpe_heap_push(heap, entry);
  }
}

static void iree_tokenizer_bpe_exact_apply_merges(
    const iree_tokenizer_bpe_model_t* model, iree_tokenizer_bpe_state_t* state,
    iree_tokenizer_bpe_heap_t* heap) {
  iree_tokenizer_bpe_exact_state_t* exact = &state->exact;

  while (!iree_tokenizer_bpe_heap_is_empty(heap)) {
    iree_tokenizer_bpe_heap_entry_t entry =
        iree_tokenizer_bpe_heap_pop(heap);
    uint32_t left_index = entry.left_start_byte;
    if (left_index >= exact->node_count) continue;

    iree_tokenizer_bpe_exact_node_t* left = &exact->nodes[left_index];
    if (left->token_id < 0 ||
        left->next == IREE_TOKENIZER_BPE_EXACT_INVALID_INDEX) {
      continue;
    }
    uint32_t right_index = left->next;
    iree_tokenizer_bpe_exact_node_t* right = &exact->nodes[right_index];
    if (right->token_id < 0 || right->prev != left_index) continue;

    iree_tokenizer_merge_hash_result_t merge =
        iree_tokenizer_vocab_merge_hash_lookup(
            model->merge_hash, left->token_id, right->token_id);
    if (!iree_tokenizer_merge_hash_result_is_valid(merge) ||
        merge.rank != entry.rank) {
      continue;
    }

    left->token_id = merge.result_id;
    left->end_byte = right->end_byte;
    left->next = right->next;
    if (right->next != IREE_TOKENIZER_BPE_EXACT_INVALID_INDEX) {
      exact->nodes[right->next].prev = left_index;
    } else {
      exact->tail = left_index;
    }
    right->token_id = -1;
    right->prev = IREE_TOKENIZER_BPE_EXACT_INVALID_INDEX;
    right->next = IREE_TOKENIZER_BPE_EXACT_INVALID_INDEX;

    if (left->prev != IREE_TOKENIZER_BPE_EXACT_INVALID_INDEX) {
      iree_tokenizer_bpe_exact_maybe_add_merge(model, exact, heap, left->prev);
    }
    iree_tokenizer_bpe_exact_maybe_add_merge(model, exact, heap, left_index);
  }
}

iree_status_t iree_tokenizer_bpe_exact_prepare(
    const iree_tokenizer_bpe_model_t* model, iree_tokenizer_bpe_state_t* state,
    iree_string_view_t segment) {
  IREE_RETURN_IF_ERROR(
      iree_tokenizer_bpe_exact_reserve(model, state, segment.size));
  iree_tokenizer_bpe_exact_reset(state);

  for (iree_host_size_t byte_position = 0; byte_position < segment.size;
       ++byte_position) {
    uint8_t input_byte = (uint8_t)segment.data[byte_position];
    int32_t token_id = model->byte_to_token[input_byte];
    IREE_ASSERT(token_id >= 0);
    iree_tokenizer_bpe_exact_append_node(
        state, token_id, (uint32_t)byte_position,
        (uint32_t)(byte_position + 1));
  }

  iree_tokenizer_bpe_heap_t heap;
  iree_tokenizer_bpe_heap_initialize(&heap, state->exact.heap_entries,
                                     state->exact.heap_capacity);
  uint32_t node_index = state->exact.head;
  while (node_index != IREE_TOKENIZER_BPE_EXACT_INVALID_INDEX) {
    iree_tokenizer_bpe_exact_maybe_add_merge(model, &state->exact, &heap,
                                             node_index);
    node_index = state->exact.nodes[node_index].next;
  }
  iree_tokenizer_bpe_exact_apply_merges(model, state, &heap);
  state->exact.emit_index = state->exact.head;
  return iree_ok_status();
}

bool iree_tokenizer_bpe_exact_emit(iree_tokenizer_bpe_state_t* state,
                                   iree_tokenizer_bpe_output_cursor_t* cursor) {
  while (state->exact.emit_index != IREE_TOKENIZER_BPE_EXACT_INVALID_INDEX) {
    iree_tokenizer_bpe_exact_node_t* node =
        &state->exact.nodes[state->exact.emit_index];
    if (!iree_tokenizer_bpe_emit_and_track(state, cursor, node->token_id,
                                           node->start_byte,
                                           node->end_byte)) {
      return false;
    }
    state->exact.emit_index = node->next;
  }
  return true;
}

void iree_tokenizer_bpe_exact_reset(iree_tokenizer_bpe_state_t* state) {
  state->exact.node_count = 0;
  state->exact.head = IREE_TOKENIZER_BPE_EXACT_INVALID_INDEX;
  state->exact.tail = IREE_TOKENIZER_BPE_EXACT_INVALID_INDEX;
  state->exact.emit_index = IREE_TOKENIZER_BPE_EXACT_INVALID_INDEX;
}

void iree_tokenizer_bpe_exact_deinitialize(
    const iree_tokenizer_bpe_model_t* model,
    iree_tokenizer_bpe_state_t* state) {
  iree_allocator_free(model->allocator, state->exact.nodes);
  iree_allocator_free(model->allocator, state->exact.heap_entries);
  state->exact.nodes = NULL;
  state->exact.node_capacity = 0;
  state->exact.heap_entries = NULL;
  state->exact.heap_capacity = 0;
  iree_tokenizer_bpe_exact_reset(state);
}
