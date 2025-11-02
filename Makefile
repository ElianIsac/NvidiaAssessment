CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2 -Iinclude
SRC = $(wildcard src/*.cpp)
OBJ = $(patsubst src/%.cpp, build/%.o, $(SRC))
BIN = build/wasm_interpreter

all: $(BIN)

$(BIN): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

build/%.o: src/%.cpp
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf build

.PHONY: all clean
