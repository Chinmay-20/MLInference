#include <iostream>
#include <vector>
#include <onnxruntime_cxx_api.h>    // ONNX Runtime C++ API, classes like Ort::Env, Ort::Session, Ort::Value
#include <yaml-cpp/yaml.h>                      // YAML parsing library for reading config.yaml

/*
1) Read YAML Configuration (model_path, input_size, num_threads)
2) Initialize ONNX Runtime environment and sesson
3) Creare input dummy tensor
4) Get input/output name, necessary for run() method
5) Execute inference with session.Run()
6) Read and print output
7) Free memory allocated
*/


int main() {
    // ====================== Load configuration ======================
    YAML::Node config = YAML::LoadFile("..//config.yaml");

    // Extract hyper-parameters from YAMl configuration, as Python Hydra
    std::string model_path = config["model"]["path"].as<std::string>();
    int input_size = config["model"]["input_size"].as<int>();
    int num_threads = config["inference"]["num_threads"].as<int>();

    std::cout << "Model path: " << model_path << "\n"
              << "Input size: " << input_size << "\n"
              << "Num threads: " << num_threads << std::endl;

    // ====================== Create ONNX Runtime environment ======================
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNXInference");       // Create global object ONNX Runtime
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(num_threads);

    // Activate all optimization graph mechanism
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // Create ONNX Runtime session, that load the .onnx model.
    // model_path.c_str() -> turn std::string into const char*, required by API
    Ort::Session session(env, model_path.c_str(), session_options);
    std::cout << "ONNX model loaded successfully!" << std::endl;

    // ====================== Create input tensor ======================
    // Allocate float vector of input_size
    std::vector<float> input_tensor_values(input_size, 1.0f);  // Dummy input: all ones

    //Specify tensor ONNX shape.
    std::vector<int64_t> input_shape{1, input_size};            // Batch size = 1

    // Define where the tensor is
    // OrtArenaAllocator: high performance allocator provided by ORT
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Create tensor ONNX Runtime in practice. 
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input_tensor_values.data(),     // Pointer to data
        input_tensor_values.size(),     // Total number of elements
        input_shape.data(),             // Shape as int64 array
        input_shape.size()              // Dimension Shape
    );

    // ====================== Prepare input/output names ======================
    // It is useful to get and free input/output names.
    Ort::AllocatorWithDefaultOptions allocator;

    // Get name of first input and first output of the models
    char* input_name = session.GetInputNameAllocated(0, allocator).release();
    char* output_name = session.GetOutputNameAllocated(0, allocator).release();

    // ONNX Runtime API requres pointer array to char, even if in this case there is just one input
    std::vector<const char*> input_names = {input_name};
    std::vector<const char*> output_names = {output_name};

    // ====================== Run inference ======================
    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr},   // Default run options
        input_names.data(),     // Pointer to input name
        &input_tensor,          // Array of input tensor
        1,                      // Number of inputs
        output_names.data(),    // Pointer to output name
        1                       // Number of output
    );

    // ====================== Read and print output ======================
    // GetTensorMutableData<float>() → Pointer to data output
    float* output_data = output_tensors.front().GetTensorMutableData<float>();

    // GetElementCount() → Total numbef of elements in output tensor
    size_t output_count = output_tensors.front().GetTensorTypeAndShapeInfo().GetElementCount();
    
    //  Print each element of the output tensor
    std::cout << "Model output: ";
    for (size_t i = 0; i < output_count; ++i) {
        std::cout << output_data[i] << " ";
    }
    std::cout << std::endl;

    // ====================== Free allocated strings ======================
    allocator.Free(input_name);
    allocator.Free(output_name);

    return 0;
}
