# Build instructions for Ubuntu 16.04

Make sure to have CUDA >= v9.0 installed. CUDA 8 has compilation errors for sm_61 device.

```
$ sudo apt-get install cmake fasm libboost-dev libboost-system-dev libboost-log-dev libboost-thread-dev
$ git clone git@github.com:dmikushin/nheqminer.git
$ cd nheqminer
$ mkdir build
$ cd build
$ cmake ..
$ make -j48
```

# Run instructions

Parameters:

``` 
	-h		Print this help and quit
	-l [location]	Stratum server:port
	-u [username]	Username (bitcoinaddress)
	-a [port]	Local API port (default: 0 = do not bind)
	-d [level]	Debug print level (0 = print all, 5 = fatal only, default: 2)
	-b [hashes]	Run in benchmark mode (default: 200 iterations)
```

CPU settings

```
	-t [num_thrds]	Number of CPU threads
	-e [ext]	Force CPU ext (0 = SSE2, 1 = AVX, 2 = AVX2)
```

NVIDIA CUDA settings

```
	-ci		CUDA info
	-cd [devices]	Enable CUDA mining on spec. devices
	-cb [blocks]	Number of blocks
	-ct [tpb]	Number of threads per block
```

Example:

```
$ ./nheqminer -cd 0 2 -cb 12 16 -ct 64 128
```

If run without parameters, miner will start mining with 75% of available logical CPU cores. Use parameter -h to learn about available parameters:

Example to run benchmark on your CPU:

```
$ ./nheqminer -b
```
        
Example to mine on your CPU with your own BTC address and worker1 on NiceHash USA server:

```
$ ./nheqminer -l equihash.usa.nicehash.com:3357 -u YOUR_BTC_ADDRESS_HERE.worker1
```

Example to mine on your CPU with your own BTC address and worker1 on EU server, using 6 threads:

```
$ ./nheqminer -l equihash.eu.nicehash.com:3357 -u YOUR_BTC_ADDRESS_HERE.worker1 -t 6
```

<i>Note: if you have a 4-core CPU with hyper threading enabled (total 8 threads) it is best to run with only 6 threads (experimental benchmarks shows that best results are achieved with 75% threads utilized)</i>

Example to mine on your CPU as well on your CUDA GPUs with your own BTC address and worker1 on EU server, using 6 CPU threads and 2 CUDA GPUs:

```
$ ./nheqminer -l equihash.eu.nicehash.com:3357 -u YOUR_BTC_ADDRESS_HERE.worker1 -t 6 -cd 0 1
```

