#include <iostream>
#include <fstream>
#include <sstream>
#include <typeinfo>
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

template<typename T>
class error;

bool containsLSTM(const torch::jit::script::Module& module) {
    // Iterate over all the modules in the model
    for (const auto& sub_module : module.named_children()) {
        // Convert the QualifiedName to a string and check if it contains "LSTM" or "lstm"
        std::string type_name = sub_module.value.type()->name().value().name();
				// std::cout << "type name" << type_name << std::endl;
        if (type_name.find("LSTM") != std::string::npos ||
            type_name.find("lstm") != std::string::npos) {
            return true;
        }
        // If the submodule contains other submodules, recursively check them
        if (containsLSTM(sub_module.value)) {
            return true;
        }
    }
    return false;
}


void call_neural_network_API() {
	auto anc_mean_path = "/home/ubuntu/chenxinwei/grl/saved_model/anc_mean.txt";
	auto anc_std_path = "/home/ubuntu/chenxinwei/grl/saved_model/anc_std.txt";

	std::vector<double, std::allocator<double> > anc_mean = load(anc_mean_path);
	std::vector<double, std::allocator<double> > anc_std = load(anc_std_path);

	torch::Tensor mean = torch::from_blob(anc_mean.data(), {static_cast<long>(anc_mean.size())});
	torch::Tensor std = torch::from_blob(anc_std.data(), {static_cast<long>(anc_std.size())});

	std::cout << "Type of mean: " << typeid(mean).name() << std::endl;
	std::cout << "Type of std: " << typeid(std).name() << std::endl;

	int64_t size = anc_mean.size();
	int64_t bs = 1;
	int64_t seqlen = 10;
	int64_t unit = 1;
	std::vector<int64_t> shape = {seqlen, bs, unit, size};
	auto obs = torch::arange(0, bs * seqlen * unit * size);
	obs = obs.reshape(shape);
	obs = obs / size / seqlen;
	std::vector<float> raw_reset;
	for (auto i = 0; i != seqlen; ++i) {
		if (i == 5) {
			raw_reset.push_back(1.0);
		} else {
			raw_reset.push_back(0.0);
		}
	}
	// std::cout << "mean: " << mean << std::endl;
	// std::cout << "std: " << std << std::endl;
	auto reset = torch::from_blob(raw_reset.data(), {static_cast<long>(raw_reset.size())});
	reset = reset.reshape({seqlen, bs, unit});
	// std::cout << "reset: " << reset << std::endl;
	
	// obs = (obs - mean) / std;
	// std::cout << obs << std::endl;

	auto model_path = "/home/ubuntu/chenxinwei/grl/saved_model/torch_model.pt";
	torch::jit::script::Module model;
	try {
			// Load the model
			model = torch::jit::load(model_path);
			std::cout << "Model loaded successfully." << std::endl;
	} catch (const c10::Error& e) {
			std::cerr << "Error loading the model: " << e.msg() << std::endl;
	}

	auto state = std::make_tuple(
		torch::zeros({bs, unit, 64}), torch::zeros({bs, unit, 64}));

	std::vector<torch::Tensor> outs;
	torch::Tensor out;
	
	c10::intrusive_ptr<c10::ivalue::Tuple> out_and_hidden;
	std::cout << "Model contains lstm: " << containsLSTM(model) << std::endl;

	for (auto i = 0; i < seqlen; ++i) {
		if (containsLSTM(model)) {
			try {
				out_and_hidden = model.forward({
					obs[i], state, reset[i]}).toTuple();
			} catch (const c10::Error& e) {
				std::cerr << "Error running the model: " << e.msg() << std::endl;
			}
			out = out_and_hidden->elements()[0].toTensor();
			auto new_state = out_and_hidden->elements()[1].toTuple();
			state = std::make_tuple(
				new_state->elements()[0].toTensor(), 
				new_state->elements()[1].toTensor());
			outs.push_back(out);
			std::cout << i << "-th output: " << out << std::endl;
		}
		else {
			try {
				out = model.forward({obs[i]}).toTensor();
			}
			catch (const c10::Error& e) {
				std::cerr << "Error running the model: " << e.msg() << std::endl;
			}
			outs.push_back(out);
		}
	}
		// for (const auto& o : out->elements()) {
		// 	outs.push_back(o.toTensor());
		// }
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