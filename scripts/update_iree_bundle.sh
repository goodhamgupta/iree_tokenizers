#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 /path/to/iree-checkout" >&2
  exit 1
fi

IREE_SOURCE_DIR="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENDOR_ROOT="${REPO_ROOT}/native/iree_tokenizers_native/vendor/iree_tokenizer_src"

if [[ ! -f "${IREE_SOURCE_DIR}/runtime/src/iree/tokenizer/tokenizer.h" ]]; then
  echo "expected an IREE checkout at ${IREE_SOURCE_DIR}" >&2
  exit 1
fi

rm -rf "${VENDOR_ROOT}"
mkdir -p "${VENDOR_ROOT}/iree"

cp -R "${IREE_SOURCE_DIR}/runtime/src/iree/base" "${VENDOR_ROOT}/iree/"
cp -R "${IREE_SOURCE_DIR}/runtime/src/iree/tokenizer" "${VENDOR_ROOT}/iree/"
cp "${IREE_SOURCE_DIR}/LICENSE" "${VENDOR_ROOT}/IREE-LICENSE"

mkdir -p "${REPO_ROOT}/native/iree_tokenizers_native/sources"

find "${VENDOR_ROOT}/iree/base" -name '*.c' \
  ! -path '*/tooling/*' \
  ! -name 'allocator_mimalloc.c' \
  | sed "s#^${REPO_ROOT}/native/iree_tokenizers_native/##" \
  | sort > "${REPO_ROOT}/native/iree_tokenizers_native/sources/base_sources.txt"

find "${VENDOR_ROOT}/iree/tokenizer" -name '*.c' \
  ! -path '*/testing/*' \
  ! -path '*/tools/*' \
  ! -path '*/testdata/*' \
  ! -path '*/format/tiktoken/*' \
  | sed "s#^${REPO_ROOT}/native/iree_tokenizers_native/##" \
  | sort > "${REPO_ROOT}/native/iree_tokenizers_native/sources/tokenizer_sources.txt"

echo "Updated vendored IREE tokenizer bundle from ${IREE_SOURCE_DIR}"
