#pragma once
#include <string>

inline int arg_int(int argc, char** argv, const std::string& key, int def) {
    for (int i = 1; i + 1 < argc; ++i)
        if (key == argv[i]) return std::stoi(argv[i + 1]);
    return def;
}

inline std::string arg_str(int argc, char** argv, const std::string& key,
                            const std::string& def) {
    for (int i = 1; i + 1 < argc; ++i)
        if (key == std::string(argv[i])) return argv[i + 1];
    return def;
}
