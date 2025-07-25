## Changes in tRecX
Main changes in timePropagatorOutput.cpp and run_trecx.cpp. Copy paste content of both files, rest is debugging and changing header files...



## Debugging
make checkpoints
```
cd tRecX
cmake -DCMAKE BUILD TYPE=Develop . // or cmake -DCMAKE_BUILD_TYPE=Develop -DCMAKE_CXX_FLAGS_DEVELOP="-g -O0" .
make -j6
gdb
set args tiptoe.inp
run
backtrace
frame 1 // depends wich checkpoint you want to debug
```