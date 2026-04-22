// Copyright (C) 2025 Eddy SDK
// SPDX-License-Identifier: Apache-2.0
//
// PRIVATE IMPLEMENTATION HEADER — do not install or include from public API.
// This directory is excluded from the install step (PATTERN "detail" EXCLUDE).

#pragma once

#include <cstdlib>
#include <cctype>
#include <string>

namespace eddy {

/// Returns true if EDDY_DEBUG environment variable is set.
/// Result is cached on first call for performance.
inline bool is_debug_enabled() {
  static bool cached = (std::getenv("EDDY_DEBUG") != nullptr);
  return cached;
}

/// Convert a string to uppercase (ASCII-safe, locale-independent).
inline std::string to_upper(const std::string& s) {
  std::string r = s;
  for (auto& c : r) c = static_cast<char>(::toupper(static_cast<unsigned char>(c)));
  return r;
}

}  // namespace eddy
