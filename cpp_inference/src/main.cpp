#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include <yaml-cpp/yaml.h>

namespace {

struct Request {
    std::vector<std::vector<float>> inputs;
};

std::string read_stdin_all() {
    std::ostringstream ss;
    ss << std::cin.rdbuf();
    return ss.str();
}

std::string read_file_all(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Failed to open input file: " + path);
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

Request parse_request_text(const std::string& text, int expected_input_size) {
    YAML::Node root = YAML::Load(text);

    if (!root["inputs"]) {
        throw std::runtime_error("Request JSON/YAML must contain key: inputs");
    }

    const YAML::Node inputs_node = root["inputs"];
    if (!inputs_node.IsSequence() || inputs_node.size() == 0) {
        throw std::runtime_error("inputs must be a non-empty array of arrays");
    }

    Request req;

    for (std::size_t i = 0; i < inputs_node.size(); ++i) {
        const YAML::Node row = inputs_node[i];
        if (!row.IsSequence()) {
            throw std::runtime_error("Each item in inputs must be an array");
        }
        if (static_cast<int>(row.size()) != expected_input_size) {
            throw std::runtime_error(
                "Invalid input length at batch index " + std::to_string(i) +
                ". Expected " + std::to_string(expected_input_size) +
                ", got " + std::to_string(row.size()));
        }

        std::vector<float> sample;
        sample.reserve(row.size());
        for (std::size_t j = 0; j < row.size(); ++j) {
            sample.push_back(row[j].as<float>());
        }
        req.inputs.push_back(std::move(sample));
    }

    return req;
}

std::vector<float> flatten_batch(const std::vector<std::vector<float>>& batch) {
    if (batch.empty()) {
        return {};
    }

    const std::size_t feature_count = batch.front().size();
    std::vector<float> flat;
    flat.reserve(batch.size() * feature_count);

    for (const auto& row : batch) {
        flat.insert(flat.end(), row.begin(), row.end());
    }
    return flat;
}

void print_json_response(const std::vector<float>& output, std::size_t batch_size, std::size_t output_dim) {
    std::cout << "{\n  \"predictions\": [";

    for (std::size_t i = 0; i < batch_size; ++i) {
        if (i > 0) {
            std::cout << ", ";
        }

        if (output_dim == 1) {
            std::cout << output[i];
        } else {
            std::cout << "[";
            for (std::size_t j = 0; j < output_dim; ++j) {
                if (j > 0) {
                    std::cout << ", ";
                }
                std::cout << output[i * output_dim + j];
            }
            std::cout << "]";
        }
    }

    std::cout << "],\n"
              << "  \"batch_size\": " << batch_size << ",\n"
              << "  \"output_dim\": " << output_dim << "\n"
              << "}\n";
}

std::string get_arg_value(int argc, char* argv[], const std::string& flag) {
    for (int i = 1; i < argc - 1; ++i) {
        if (argv[i] == flag) {
            return argv[i + 1];
        }
    }
    return "";
}

bool has_flag(int argc, char* argv[], const std::string& flag) {
    for (int i = 1; i < argc; ++i) {
        if (argv[i] == flag) {
            return true;
        }
    }
    return false;
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        const bool use_stdin = has_flag(argc, argv, "--stdin");
        const std::string input_file = get_arg_value(argc, argv, "--input");

        if (!use_stdin && input_file.empty()) {
            std::cerr
                << "Usage:\n"
                << "  ./onnx_inference --stdin\n"
                << "  ./onnx_inference --input request.json\n";
            return 1;
        }

        YAML::Node config = YAML::LoadFile("../config.yaml");
        const std::string model_path = config["model"]["path"].as<std::string>();
        const int input_size = config["model"]["input_size"].as<int>();
        const int num_threads = config["inference"]["num_threads"].as<int>();

        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNXInference");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(num_threads);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        Ort::Session session(env, model_path.c_str(), session_options);

        const std::string request_text = use_stdin ? read_stdin_all() : read_file_all(input_file);
        const Request req = parse_request_text(request_text, input_size);

        const std::size_t batch_size = req.inputs.size();
        const std::vector<float> flat_input = flatten_batch(req.inputs);
        const std::vector<int64_t> input_shape = {
            static_cast<int64_t>(batch_size),
            static_cast<int64_t>(input_size)
        };

        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            const_cast<float*>(flat_input.data()),
            flat_input.size(),
            input_shape.data(),
            input_shape.size()
        );

        Ort::AllocatorWithDefaultOptions allocator;
        auto input_name_alloc = session.GetInputNameAllocated(0, allocator);
        auto output_name_alloc = session.GetOutputNameAllocated(0, allocator);

        const char* input_names[] = {input_name_alloc.get()};
        const char* output_names[] = {output_name_alloc.get()};

        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr},
            input_names,
            &input_tensor,
            1,
            output_names,
            1
        );

        float* output_data = output_tensors.front().GetTensorMutableData<float>();
        const auto shape_info = output_tensors.front().GetTensorTypeAndShapeInfo();
        const std::vector<int64_t> output_shape = shape_info.GetShape();
        const std::size_t output_count = shape_info.GetElementCount();

        if (output_shape.empty()) {
            throw std::runtime_error("Model output shape is empty");
        }

        const std::size_t inferred_batch = static_cast<std::size_t>(output_shape[0]);
        if (inferred_batch != batch_size) {
            throw std::runtime_error(
                "Output batch size mismatch. Expected " + std::to_string(batch_size) +
                ", got " + std::to_string(inferred_batch));
        }

        const std::size_t output_dim =
            batch_size == 0 ? 0 : (output_count / batch_size);

        std::vector<float> output(output_data, output_data + output_count);
        print_json_response(output, batch_size, output_dim);

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "{\n"
                  << "  \"error\": \"" << ex.what() << "\"\n"
                  << "}\n";
        return 2;
    }
}