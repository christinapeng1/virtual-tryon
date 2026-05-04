# virtual-tryon

mkdir -p build
cd build
git clone https://github.com/ocornut/imgui.git 
cmake .. && make
./main

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
