// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Segmenter for the Hugging Face Split regex:
//   \d{1,3}(?=(?:\d{3})*\b)
//
// This pattern emits right-aligned groups inside digit runs, e.g.
// "1234567" -> "1", "234", "567". The vendored regex engine does not
// support positive lookahead, so this segmenter implements the exact
// tokenizer pattern directly.

#ifndef IREE_TOKENIZER_SEGMENTER_THOUSANDS_H_
#define IREE_TOKENIZER_SEGMENTER_THOUSANDS_H_

#include "iree/base/api.h"
#include "iree/tokenizer/segmenter.h"

#ifdef __cplusplus
extern "C" {
#endif

iree_status_t iree_tokenizer_segmenter_thousands_allocate(
    iree_allocator_t allocator, iree_tokenizer_segmenter_t** out_segmenter);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_SEGMENTER_THOUSANDS_H_
