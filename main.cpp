#include <torch/extension.h>

std::vector<torch::Tensor> forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V);
std::vector<torch::Tensor> backward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor O,
    torch::Tensor dO,
    torch::Tensor l,
    torch::Tensor m
);

std::vector<torch::Tensor> forward_2(torch::Tensor Q, torch::Tensor K, torch::Tensor V);
std::vector<torch::Tensor> backward_2(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor O,
    torch::Tensor dO,
    torch::Tensor L
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", torch::wrap_pybind_function(forward), "forward");
    m.def("backward", torch::wrap_pybind_function(backward), "backward");

    m.def("forward_2", torch::wrap_pybind_function(forward_2), "forward_2");
    m.def("backward_2", torch::wrap_pybind_function(backward_2), "backward_2");
}
