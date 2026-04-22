#pragma once

#include "eddy/models/parakeet-v2/parakeet_openvino.hpp"
#include "eddy/models/parakeet-v2/parakeet_encoder.hpp"
#include "eddy/models/parakeet-v2/tokenizer.hpp"

#include <openvino/openvino.hpp>
#include <mutex>
#include <stdexcept>
#include <string>

namespace eddy::parakeet {

// PRIVATE IMPLEMENTATION HEADER - DO NOT INSTALL OR INCLUDE IN PUBLIC API
//
// This file contains internal implementation details for OpenVINOParakeet.
// It is kept in detail/ to hide implementation from users:
// - OpenVINO types (ov::CompiledModel, ov::InferRequest)
// - Internal state management (mutexes, port indices)
// - Implementation can change without breaking API
//
// Only parakeet_*.cpp files should include this header.

// Shared helper: read a scalar length value from a tensor that may be i32 or i64.
// Validates that the length is non-negative.
[[nodiscard]] inline int64_t read_length_scalar(const ov::Tensor& t) {
  const auto et = t.get_element_type();
  int64_t value;

  if (et == ov::element::i64) {
    value = t.data<int64_t>()[0];
  } else if (et == ov::element::i32) {
    const int32_t val32 = t.data<int32_t>()[0];
    if (val32 < 0) {
      throw std::runtime_error("Length tensor value is negative: " + std::to_string(val32));
    }
    value = static_cast<int64_t>(val32);
  } else {
    throw std::runtime_error("Length tensor has unsupported element type (expected i32 or i64)");
  }

  if (value < 0) {
    throw std::runtime_error("Length tensor value is negative: " + std::to_string(value));
  }

  return value;
}

// Implementation struct for OpenVINOParakeet (Pimpl idiom)
// Also used directly by helper functions as ParakeetImpl alias
struct ParakeetImpl {
  std::shared_ptr<eddy::OpenVINOBackend> backend;
  ModelPaths model_paths;
  RuntimeConfig runtime_cfg;

  Tokenizer tokenizer;

  ov::CompiledModel preproc_model;
  ov::InferRequest preproc_request;

  ov::CompiledModel encoder_model;
  ov::InferRequest encoder_request;

  ov::CompiledModel decoder_model;
  ov::InferRequest decoder_request;

  ov::CompiledModel joint_model;
  ov::InferRequest joint_request;

  size_t encoder_expected_frames = 0;
  size_t encoder_hidden_size = 0;
  size_t decoder_hidden_size = 0;
  size_t joint_output_size = 0;

  std::once_flag compile_once;
  std::mutex request_guard;

  // Resolved encoder ports
  EncoderPorts encoder_ports;

  // Output indices for encoder outputs (robust retrieval)
  size_t encoder_output_index = 0;   // [1, hidden, time]
  size_t encoder_length_index = 1;   // [1]
};

// OpenVINOParakeet::Impl simply inherits from ParakeetImpl
// This satisfies the forward declaration while allowing helper functions to use ParakeetImpl directly
struct OpenVINOParakeet::Impl : ParakeetImpl {};

}  // namespace eddy::parakeet
