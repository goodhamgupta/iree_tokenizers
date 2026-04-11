#!/usr/bin/env bash
set -euo pipefail

EXPECTED_IREE_COMMIT="71af3a5e41a8e265330bc693194c708cf6df4724"

if [[ $# -lt 1 ]]; then
  echo "usage: $0 /path/to/iree-checkout" >&2
  exit 1
fi

IREE_SOURCE_DIR="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENDOR_ROOT="${REPO_ROOT}/native/iree_tokenizers_native/vendor/iree_tokenizer_src"
COMMIT_FILE="${REPO_ROOT}/native/iree_tokenizers_native/vendor/IREE_COMMIT"

if [[ ! -f "${IREE_SOURCE_DIR}/runtime/src/iree/tokenizer/tokenizer.h" ]]; then
  echo "expected an IREE checkout at ${IREE_SOURCE_DIR}" >&2
  exit 1
fi

ACTUAL_COMMIT="$(git -C "${IREE_SOURCE_DIR}" rev-parse HEAD)"
if [[ "${ACTUAL_COMMIT}" != "${EXPECTED_IREE_COMMIT}" ]]; then
  echo "expected IREE commit ${EXPECTED_IREE_COMMIT}, got ${ACTUAL_COMMIT}" >&2
  exit 1
fi

rm -rf "${VENDOR_ROOT}"
mkdir -p "${VENDOR_ROOT}/iree"

cp -R "${IREE_SOURCE_DIR}/runtime/src/iree/base" "${VENDOR_ROOT}/iree/"
cp -R "${IREE_SOURCE_DIR}/runtime/src/iree/tokenizer" "${VENDOR_ROOT}/iree/"
cp "${IREE_SOURCE_DIR}/LICENSE" "${VENDOR_ROOT}/IREE-LICENSE"
printf '%s\n' "${ACTUAL_COMMIT}" > "${COMMIT_FILE}"

find "${VENDOR_ROOT}" \
  \( -name '*.cc' -o -name '*.cpp' -o -name '*_test.cc' -o -name '*_benchmark.cc' -o -name '*_fuzz.cc' \) \
  -delete
find "${VENDOR_ROOT}" \
  \( -name 'BUILD.bazel' -o -name 'CMakeLists.txt' -o -name '*.dict' -o -name '*.py' \) \
  -delete
rm -rf \
  "${VENDOR_ROOT}/iree/base/testing" \
  "${VENDOR_ROOT}/iree/base/tooling" \
  "${VENDOR_ROOT}/iree/tokenizer/tools" \
  "${VENDOR_ROOT}/iree/tokenizer/testing" \
  "${VENDOR_ROOT}/iree/tokenizer/testdata"

mkdir -p "${REPO_ROOT}/native/iree_tokenizers_native/sources"

find "${VENDOR_ROOT}/iree/base" -name '*.c' \
  ! -path '*/tooling/*' \
  ! -name 'allocator_mimalloc.c' \
  ! -path '*/internal/cpu.c' \
  ! -path '*/threading/*' \
  | sed "s#^${REPO_ROOT}/native/iree_tokenizers_native/##" \
  | sort > "${REPO_ROOT}/native/iree_tokenizers_native/sources/base_sources.txt"

printf '%s\n' \
  "vendor/iree_tokenizer_src/iree/base/threading/mutex.c" \
  >> "${REPO_ROOT}/native/iree_tokenizers_native/sources/base_sources.txt"

find "${VENDOR_ROOT}/iree/tokenizer" -name '*.c' \
  ! -path '*/testing/*' \
  ! -path '*/tools/*' \
  ! -path '*/testdata/*' \
  | sed "s#^${REPO_ROOT}/native/iree_tokenizers_native/##" \
  | sort > "${REPO_ROOT}/native/iree_tokenizers_native/sources/tokenizer_sources.txt"

echo "Updated vendored IREE tokenizer bundle from ${IREE_SOURCE_DIR}"
