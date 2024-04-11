#include <torch/extension.h>

std::vector<torch::Tensor> flash_attention_1_forward(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V
);
std::vector<torch::Tensor> flash_attention_1_backward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor O,
    torch::Tensor dO,
    torch::Tensor l,
    torch::Tensor m
);

std::vector<torch::Tensor> flash_attention_2_forward(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V
);
std::vector<torch::Tensor> flash_attention_2_backward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor O,
    torch::Tensor dO,
    torch::Tensor L
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "flash_attention_1_forward",
        torch::wrap_pybind_function(flash_attention_1_forward),
        "flash_attention_1_forward"
    );
    m.def(
        "flash_attention_1_backward",
        torch::wrap_pybind_function(flash_attention_1_backward),
        "flash_attention_1_backward"
    );

    m.def(
        "flash_attention_2_forward",
        torch::wrap_pybind_function(flash_attention_2_forward),
        "flash_attention_2_forward"
    );
    m.def(
        "flash_attention_2_backward",
        torch::wrap_pybind_function(flash_attention_2_backward),
        "flash_attention_2_backward"
    );
}
