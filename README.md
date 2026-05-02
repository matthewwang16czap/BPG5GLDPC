Sionna 2.0 has support pytorch, will build a sionna2.0 ver system soon.

## Prerequisites

### `pip`

```bash
pip install -r requirements.txt
```

### `libbpg`

Install all required system packages:

```bash
sudo apt update
sudo apt install -y build-essential cmake yasm libpng-dev libjpeg-dev libsdl1.2-dev
```

Clone and build the library:

```bash
git clone https://github.com/mirrorer/libbpg.git
cd libbpg

make clean
make -j$(nproc)
```

To install system-wide:

```bash
sudo make install
```

If you encounter build issues related to NUMA or system libraries, try:

```bash
sudo apt-get remove libnuma-dev
```

Then rebuild:

```bash
make clean
make -j$(nproc)
sudo make install
```

## Related links

- BPG image format by _Fabrice Bellard_: https://bellard.org/bpg
- Sionna An Open-Source Library for Next-Generation Physical Layer Research: https://github.com/NVlabs/sionna
- Kodak image dataset: http://r0k.us/graphics/kodak/
- BPG Image library and utilities: https://github.com/mirrorer/libbpg
