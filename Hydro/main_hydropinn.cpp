#include <QCoreApplication>
#include <iostream>
#include <string>
#include <ATen/ATen.h>

int main(int argc, char *argv[])
{
    QCoreApplication app(argc, argv);

    at::set_num_threads(1);
    at::set_num_interop_threads(1);

    std::string mode = "ffn";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a.rfind("--mode=", 0) == 0) mode = a.substr(7);
    }

    std::cout << "HydroPINN mode = " << mode << std::endl;

    if (mode == "ffn") {
        std::cout << "Run FFN baseline" << std::endl;
        return 0;
    }
    if (mode == "ffn_pinn") {
        std::cout << "Run FFN-PINN" << std::endl;
        return 0;
    }
    if (mode == "lstm") {
        std::cout << "Run LSTM baseline" << std::endl;
        return 0;
    }
    if (mode == "lstm_pinn") {
        std::cout << "Run LSTM-PINN" << std::endl;
        return 0;
    }

    std::cerr << "Unknown mode: " << mode << std::endl;
    return 1;
}
