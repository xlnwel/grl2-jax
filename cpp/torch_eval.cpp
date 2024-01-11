#include <iostream>
#include <fstream>
#include <sstream>
#include <torch/torch.h>
#include <torch/script.h>


std::vector<double> load(const std::string& path) {
	std::ifstream file(path);

	if (!file) {
		std::cout << "Failed to open the mean or std file." << std::endl;
		return {};
	}

	std::vector<double> vec;
	double var;

	while (file >> var) {
		vec.push_back(var);
	}

	return vec;
}


void call_neural_network_API() {
	auto anc_mean_path = "/home/ubuntu/chenxinwei/grl/saved_model/anc_mean.txt";
	auto anc_std_path = "/home/ubuntu/chenxinwei/grl/saved_model/anc_std.txt";

	auto anc_mean = load(anc_mean_path);
	auto anc_std = load(anc_std_path);

	auto mean = torch::from_blob(anc_mean.data(), {static_cast<long>(anc_mean.size())});
	auto std = torch::from_blob(anc_std.data(), {static_cast<long>(anc_std.size())});

	int64_t size = anc_mean.size();
	std::vector<int64_t> shape = {1, size};
	auto obs = torch::arange(0, size);
	obs = obs.reshape(shape);
	obs = obs / size;
	// obs = (obs - mean) / std;
	std::cout << obs << std::endl;

	auto model_path = "/home/ubuntu/chenxinwei/grl/saved_model/torch_model.pt";
	torch::jit::script::Module model;
	try {
			// Load the model
			model = torch::jit::load(model_path);
			std::cout << "Model loaded successfully." << std::endl;
	} catch (const c10::Error& e) {
			std::cerr << "Error loading the model: " << e.msg() << std::endl;
	}

	std::vector<torch::Tensor> outs;
	torch::Tensor out;
	try {
		out = model.forward({obs}).toTensor();
		// for (const auto& o : out->elements()) {
		// 	outs.push_back(o.toTensor());
		// }
	} catch (const c10::Error& e) {
		std::cerr << "Error running the model: " << e.msg() << std::endl;
	}
	std::cout << out << std::endl;
	// std::cout << "Number of output tensors: " << outs.size() << std::endl;
	// for (size_t i = 0; i < outs.size(); ++i) {
	// 		std::cout << "Output Tensor " << i + 1 << ":" << std::endl;
	// 		std::cout << outs[i] << std::endl;
	// }
}

int main(int argc, const char* argv[]) {
	call_neural_network_API();
	return 0;
}