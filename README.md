# virtual-tryon

mkdir -p build
cd build
cmake .. && make
./main

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
