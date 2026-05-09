# Dress to Preprocess (Virtual Try-On)

A virtual try-on system that combines pose estimation and gesture recognition to overlay 3D clothing meshes onto a user in real-time.

## Quickstart
mkdir -p build

cd build

git clone https://github.com/ocornut/imgui.git 

cmake .. && make

./main

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

## Recommended Meshes
1. jeans_denim_jacket
2. tshirt_shortSleeve
3. cute_blazer_with_bears_with_bones (adjusted down to 0.0)
4. shirt_2
5. basic_sweatshirt (adjusted down)
6. fleece_jacket

