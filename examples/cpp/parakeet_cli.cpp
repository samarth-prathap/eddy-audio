// Copyright (C) 2025 Eddy SDK
// SPDX-License-Identifier: Apache-2.0

#include "eddy/backends/openvino_backend.hpp"
#include "eddy/core/app_dir.hpp"
#include "eddy/core/model_configs.hpp"
#include "eddy/models/parakeet-v2/parakeet.hpp"
#include "eddy/models/parakeet-v2/parakeet_openvino.hpp"
#include "eddy/utils/ensure_models.hpp"
#include "eddy/utils/audio_utils.hpp"

#include <openvino/openvino.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <algorithm>
#include <cctype>

static constexpr const char* EDDY_CLI_VERSION = "0.1.0";

// Convert string to uppercase (used for case-insensitive device comparison)
static std::string to_upper(const std::string& s) {
    std::string r = s;
    for (auto& c : r) c = static_cast<char>(::toupper(static_cast<unsigned char>(c)));
    return r;
}

// Escape a string for embedding in a JSON double-quoted value.
static std::string json_escape(const std::string& s) {
    std::string r;
    r.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
            case '"':  r += "\\\""; break;
            case '\\': r += "\\\\"; break;
            case '\n': r += "\\n";  break;
            case '\r': r += "\\r";  break;
            case '\t': r += "\\t";  break;
            default:
                if (c < 0x20) {
                    // Control characters — emit \uXXXX
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    r += buf;
                } else {
                    r += static_cast<char>(c);
                }
                break;
        }
    }
    return r;
}

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <audio.wav> [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --model <model>      Model version (default: parakeet-v2)\n";
    std::cout << "                       Options: parakeet-v2, parakeet-v3\n";
    std::cout << "  --device <device>    OpenVINO device (default: CPU)\n";
    std::cout << "                       Examples: CPU, NPU, GPU, AUTO\n";
    std::cout << "  --output-json        Write result as JSON to stdout; route progress to stderr\n";
    std::cout << "  --list-devices       List available OpenVINO devices and exit\n";
    std::cout << "  --version            Print version and exit\n";
    std::cout << "  --help               Show this help message\n\n";
    std::cout << "Requirements:\n";
    std::cout << "  - Audio must be WAV file (any sample rate; resampled to 16 kHz automatically)\n";
    std::cout << "  - Models are downloaded automatically on first run\n\n";
    std::cout << "Examples:\n";
    std::cout << "  " << program_name << " test.wav\n";
    std::cout << "  " << program_name << " test.wav --model parakeet-v3 --device NPU\n";
    std::cout << "  " << program_name << " test.wav --output-json > result.json\n";
}

int main(int argc, char* argv[]) {
    // Force unbuffered output
    std::cout.setf(std::ios::unitbuf);
    std::cerr.setf(std::ios::unitbuf);

    // Parse command line arguments
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string audio_file;
    std::string device = "CPU";
    std::string model_name = "parakeet-v2";
    bool output_json = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--version" || arg == "-v") {
            std::cout << "parakeet_cli version " << EDDY_CLI_VERSION << "\n";
            return 0;
        } else if (arg == "--list-devices") {
            try {
                ov::Core core;
                auto devices = core.get_available_devices();
                std::cout << "Available OpenVINO devices:\n";
                for (const auto& d : devices) {
                    std::string full_name;
                    try {
                        full_name = core.get_property(d, ov::device::full_name);
                    } catch (...) {}
                    std::cout << "  " << d;
                    if (!full_name.empty()) std::cout << "  (" << full_name << ")";
                    std::cout << "\n";
                }
                if (devices.empty()) std::cout << "  (none detected)\n";
            } catch (const std::exception& e) {
                std::cerr << "[ERROR] Could not query devices: " << e.what() << "\n";
                return 1;
            }
            return 0;
        } else if (arg == "--model") {
            if (i + 1 >= argc) {
                std::cerr << "Error: --model requires an argument\n";
                return 1;
            }
            model_name = argv[++i];
            if (model_name != "parakeet-v2" && model_name != "parakeet-v3") {
                std::cerr << "Error: Invalid model '" << model_name
                          << "'. Use 'parakeet-v2' or 'parakeet-v3'\n";
                return 1;
            }
        } else if (arg == "--device") {
            if (i + 1 >= argc) {
                std::cerr << "Error: --device requires an argument\n";
                return 1;
            }
            device = argv[++i];
        } else if (arg == "--output-json") {
            output_json = true;
        } else {
            // Assume it's the audio file
            audio_file = arg;
        }
    }

    if (audio_file.empty()) {
        std::cerr << "Error: No audio file specified\n\n";
        print_usage(argv[0]);
        return 1;
    }

    // When --output-json is set, route all progress/status text to stderr so
    // stdout stays clean for the JSON result.
    std::ostream& info = output_json ? std::cerr : std::cout;

    // Validate device name (skip meta-devices that OpenVINO always accepts)
    {
        const std::string dev_up = to_upper(device);
        const bool is_meta = (dev_up == "AUTO" ||
                              dev_up.rfind("HETERO:", 0) == 0 ||
                              dev_up.rfind("MULTI:", 0) == 0);
        if (!is_meta) {
            try {
                ov::Core probe;
                auto available = probe.get_available_devices();
                bool found = false;
                for (const auto& d : available) {
                    // Accept exact match or prefix match (e.g., "GPU" matches "GPU.0")
                    const std::string d_up = to_upper(d);
                    if (d_up == dev_up || d_up.rfind(dev_up, 0) == 0) {
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    std::cerr << "Error: Device '" << device << "' is not available.\n";
                    std::cerr << "Available devices:";
                    for (const auto& d : available) std::cerr << " " << d;
                    if (available.empty()) std::cerr << " (none detected)";
                    std::cerr << "\n";
                    std::cerr << "Use --list-devices for details, or --device AUTO to let OpenVINO choose.\n";
                    return 1;
                }
            } catch (const std::exception& e) {
                // Non-fatal: continue and let OpenVINO produce its own error if the device is invalid.
                std::cerr << "[WARN] Could not validate device: " << e.what() << "\n";
            }
        }
    }

    info << "=== Parakeet TDT Transcription CLI (" << model_name << ") ===\n\n";

    try {
        // Load audio file
        info << "Loading audio: " << audio_file << " ... ";
        auto audio_samples = eddy::audio::read_wav(audio_file);
        info << "[OK]\n";
        info << "  Samples: " << audio_samples.size() << "\n";
        info << "  Duration: " << std::fixed << std::setprecision(2)
             << (audio_samples.size() / 16000.0) << " seconds\n\n";

        // Create OpenVINO backend
        info << "Initializing OpenVINO backend (" << device << ") ... ";
        auto compiled_cache_dir = eddy::get_model_dir(model_name).string();
        eddy::OpenVINOOptions ov_opts;
        ov_opts.device = device;
        ov_opts.cache_dir = compiled_cache_dir;
        auto backend = std::make_shared<eddy::OpenVINOBackend>(ov_opts);
        info << "[OK]\n";

        // Ensure model files are present — download if necessary
        auto cache_model_dir = eddy::get_model_assets_dir(model_name);
        {
            std::string check_err;
            if (!eddy::parakeet::check_models_available(cache_model_dir, &check_err)) {
                info << "[INFO] " << check_err << "\n";
                info << "Downloading models from HuggingFace (this may take several minutes)...\n";

                auto it = eddy::model_configs::MODEL_MAP.find(model_name);
                if (it == eddy::model_configs::MODEL_MAP.end()) {
                    std::cerr << "[ERROR] No download config found for model: " << model_name << "\n";
                    return 1;
                }

                std::string dl_err;
                bool ok = eddy::parakeet::download_models(
                    it->second, cache_model_dir, &dl_err,
                    [&info](const std::string& fname, int cur, int total) {
                        info << "  [" << cur << "/" << total << "] " << fname << "\n";
                    });

                if (!ok) {
                    std::cerr << "[ERROR] Model download failed: " << dl_err << "\n";
                    std::cerr << "  You can retry by running hf_fetch_models --model " << model_name << "\n";
                    std::cerr << "  or manually place model files in: " << cache_model_dir.string() << "\n";
                    return 1;
                }
                info << "Download complete.\n";
            }
        }

        // Resolve model directory
        std::filesystem::path model_dir;
        auto exists_nonempty = [](const std::filesystem::path& p) -> bool {
            std::error_code ec;
            auto size = std::filesystem::file_size(p, ec);
            return !ec && size > 0;
        };
        if (exists_nonempty(cache_model_dir / "parakeet_encoder.xml")) {
            model_dir = cache_model_dir;
            info << "Using cached models at: " << cache_model_dir.string() << "\n\n";
        } else {
            // Fallback: legacy Windows path (%LOCALAPPDATA%\eddy\cache\models\<name>\files)
#if defined(_WIN32)
            auto legacy_dir = eddy::get_app_data_dir() / "cache" / "models" / model_name / "files";
            if (exists_nonempty(legacy_dir / "parakeet_encoder.xml")) {
                model_dir = legacy_dir;
                info << "Using legacy cached models at: " << legacy_dir.string() << "\n\n";
            } else
#endif
            {
                model_dir = "models/parakeet";
                info << "Using local models at: " << model_dir.string() << "\n";
                info << "Note: Copy models to " << cache_model_dir.string() << " for user cache access\n\n";
            }
        }

        // Configure model paths
        eddy::parakeet::ModelPaths paths{
            .preprocessor = {.path = (model_dir / "parakeet_melspectogram.xml").string()},
            .encoder      = {.path = (model_dir / "parakeet_encoder.xml").string()},
            .decoder      = {.path = (model_dir / "parakeet_decoder.xml").string()},
            .joint        = {.path = (model_dir / "parakeet_joint.xml").string()},
            .tokenizer_json = (model_dir / "parakeet_vocab.json").string()
        };

        // Configure runtime
        // V2: blank_token_id=1024; V3: blank_token_id=8192
        const bool is_v3 = (model_name == "parakeet-v3");
        eddy::parakeet::RuntimeConfig cfg{
            .device = device,
            .blank_token_id = is_v3 ? 8192 : 1024,
            .duration_bins = {0, 1, 2, 3, 4},
            .v3_preprocessor_workaround = is_v3
        };

        // Load models
        info << "Loading Parakeet models ... ";
        auto model = eddy::parakeet::make_openvino_parakeet(backend, paths, cfg);
        info << "[OK]\n";

        // Warmup
        info << "Warming up model ... ";
        auto parakeet_model = std::static_pointer_cast<eddy::parakeet::OpenVINOParakeet>(model);
        parakeet_model->warmup();
        info << "[OK]\n\n";

        // Prepare audio segment
        eddy::parakeet::AudioSegment segment;
        segment.sample_rate = 16000;
        segment.pcm = audio_samples;

        // Run inference
        info << std::string(70, '=') << "\n";
        info << "TRANSCRIBING...\n";
        info << std::string(70, '=') << "\n\n";

        eddy::parakeet::SegmentOptions options;
        auto start = std::chrono::high_resolution_clock::now();
        auto result = model->infer(segment, options);
        auto end   = std::chrono::high_resolution_clock::now();

        // Calculate metrics
        auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        float audio_duration = audio_samples.size() / 16000.0f;
        float rtfx = (duration_ms > 0) ? audio_duration / (duration_ms / 1000.0f) : 0.0f;

        // ---------------------------------------------------------------
        // JSON output path
        // ---------------------------------------------------------------
        if (output_json) {
            std::ostringstream js;
            js << "{\n";
            js << "  \"text\": \"" << json_escape(result.text) << "\",\n";
            js << "  \"model\": \"" << json_escape(model_name) << "\",\n";
            js << "  \"device\": \"" << json_escape(device) << "\",\n";
            js << "  \"latency_ms\": " << std::fixed << std::setprecision(1) << duration_ms << ",\n";
            js << "  \"audio_duration_s\": " << std::fixed << std::setprecision(3) << audio_duration << ",\n";
            js << "  \"rtfx\": " << std::fixed << std::setprecision(2) << rtfx << ",\n";
            js << "  \"confidence\": " << std::fixed << std::setprecision(4)
               << result.overall_confidence << ",\n";
            js << "  \"tokens\": " << result.token_ids.size() << ",\n";
            js << "  \"token_timings\": [\n";
            for (size_t i = 0; i < result.token_timings.size(); ++i) {
                const auto& t = result.token_timings[i];
                js << "    {\"token_id\": " << t.token_id
                   << ", \"time_s\": " << std::fixed << std::setprecision(3) << (t.frame_index * 0.08f)
                   << ", \"confidence\": " << std::fixed << std::setprecision(4) << t.confidence << "}";
                if (i + 1 < result.token_timings.size()) js << ",";
                js << "\n";
            }
            js << "  ]\n";
            js << "}\n";
            std::cout << js.str();
            return 0;
        }

        // ---------------------------------------------------------------
        // Human-readable output path
        // ---------------------------------------------------------------
        std::cout << "Result:\n";
        std::cout << std::string(70, '-') << "\n";
        std::cout << result.text << "\n";
        std::cout << std::string(70, '-') << "\n\n";

        std::cout << "Metrics:\n";
        std::cout << "  Tokens:           " << result.token_ids.size() << "\n";
        std::cout << "  Confidence:       " << std::fixed << std::setprecision(1)
                  << (result.overall_confidence * 100.0f) << "%\n";
        std::cout << "  Processing time:  " << duration_ms << " ms\n";
        std::cout << "  Audio duration:   " << std::fixed << std::setprecision(2)
                  << audio_duration << " s\n";
        std::cout << "  Real-time factor: " << std::fixed << std::setprecision(1)
                  << rtfx << "x\n\n";

        // Show token timings (first 10 tokens as sample)
        if (!result.token_timings.empty()) {
            std::cout << "Token Timings (first 10):\n";
            const size_t num_to_show = std::min(size_t(10), result.token_timings.size());
            for (size_t i = 0; i < num_to_show; ++i) {
                const auto& timing = result.token_timings[i];
                float time_seconds = timing.frame_index * 0.08f;
                std::cout << "  " << std::setw(3) << i+1 << ". "
                          << "t=" << std::fixed << std::setprecision(2) << std::setw(5) << time_seconds << "s "
                          << "conf=" << std::setprecision(1) << std::setw(4) << (timing.confidence * 100.0f) << "% "
                          << "token_id=" << timing.token_id << "\n";
            }
            if (result.token_timings.size() > num_to_show) {
                std::cout << "  ... and " << (result.token_timings.size() - num_to_show) << " more tokens\n";
            }
            std::cout << "\n";
        }

        // Performance assessment
        if (rtfx >= 10.0f) {
            std::cout << "✅ Performance: Excellent (>" << std::setprecision(0) << rtfx << "x real-time)\n";
        } else if (rtfx >= 1.0f) {
            std::cout << "✅ Performance: Good (processing faster than real-time)\n";
        } else {
            std::cout << "⚠️  Performance: Below real-time (consider optimizations)\n";
        }

        std::cout << "\n" << std::string(70, '=') << "\n";
        std::cout << "SUCCESS\n";
        std::cout << std::string(70, '=') << "\n";

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\n[ERROR] " << e.what() << "\n\n";
        std::cerr << "Troubleshooting:\n";
        std::cerr << "  1. Ensure audio file is a supported format (WAV recommended)\n";
        std::cerr << "  2. Check models are in: " << eddy::get_model_assets_dir(model_name).string() << "\n";
        std::cerr << "     or run: hf_fetch_models --model " << model_name << "\n";
        std::cerr << "  3. Verify OpenVINO runtime is properly installed\n";
        std::cerr << "  4. Run --list-devices to see available devices\n";
        std::cerr << "  5. Try --device CPU if another device fails\n";
        return 1;
    }
}
