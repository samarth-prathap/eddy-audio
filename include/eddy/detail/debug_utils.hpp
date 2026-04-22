// Copyright (C) 2025 Eddy SDK
// SPDX-License-Identifier: Apache-2.0
//
// PRIVATE IMPLEMENTATION HEADER — do not install or include from public API.
// This directory is excluded from the install step (PATTERN "detail" EXCLUDE).

#pragma once

#include <cstdlib>

namespace eddy {

/// Returns true if EDDY_DEBUG environment variable is set.
/// Result is cached on first call for performance.
inline bool is_debug_enabled() {
  static bool cached = (std::getenv("EDDY_DEBUG") != nullptr);
  return cached;
}

}  // namespace eddy
