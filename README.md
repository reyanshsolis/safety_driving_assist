# drowsy_driver
This keeps maintains an Attention Score based on Blinking Rate and Yawning of Driver, based on which it Warns him takes control of the vehicle if Attention score fall below certain level using CAN protocol to control Vehicle autonomously. 


#DLIB


# dlib C++ library [![Travis Status](https://travis-ci.org/davisking/dlib.svg?branch=master)](https://travis-ci.org/davisking/dlib)

Dlib is a modern C++ toolkit containing machine learning algorithms and tools for creating complex software in C++ to solve real world problems. See [http://dlib.net](http://dlib.net) for the main project documentation and API reference.



## Compiling dlib C++ example programs

Go into the examples folder and type:

```bash
mkdir build; cd build; cmake .. ; cmake --build .
```

That will build all the examples.
If you have a CPU that supports AVX instructions then turn them on like this:

```bash
mkdir build; cd build; cmake .. -DUSE_AVX_INSTRUCTIONS=1; cmake --build .
```

Doing so will make some things run faster.

Finally, Visual Studio users should usually do everything in 64bit mode.  By default Visual Studio is 32bit, both in its outputs and its own execution, so you have to explicitly tell it to use 64bits.  Since it's not the 1990s anymore you probably want to use 64bits.  Do that with a cmake invocation like this:
```bash
cmake .. -G "Visual Studio 14 2015 Win64" -T host=x64 
```

## Compiling your own C++ programs that use dlib

The examples folder has a [CMake tutorial](https://github.com/davisking/dlib/blob/master/examples/CMakeLists.txt) that tells you what to do.  There are also additional instructions on the [dlib web site](http://dlib.net/compile.html).

## Compiling dlib Python API

Before you can run the Python example programs you must compile dlib. Type:

```bash
python setup.py install
```


## Running the unit test suite

Type the following to compile and run the dlib unit test suite:

```bash
cd dlib/test
mkdir build
cd build
cmake ..
cmake --build . --config Release
./dtest --runall
```

#CAN Protocol 
### Link for KVASER drivers- https://www.kvaser.com/linux-drivers-and-sdk/

### Link for canLib installation - https://www.youtube.com/watch?v=Gz-lIVIU7ys&feature=youtu.be
 


