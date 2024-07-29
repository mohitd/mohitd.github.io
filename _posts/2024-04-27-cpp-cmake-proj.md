---
layout: post
title: "Anatomy of a Good-enough Modern CMake Project for C++ Libraries"
excerpt: "CMake is the most common meta-build system used for building C/C++ libraries and applications. In this post, I'll describe the anatomy of a good-enough C++ library project structure and a good-enough way to build it using CMake."
comments: true
---

C++ is one of the most widely used programming languages in the world, from mobile apps to gaming to robotics. Personally, I've used it for at least these things, but there are hundreds and thousands of more uses of the language. Invariably, the majority of those uses will, at some point, involve having to write C++ libraries and executables. 

CMake is one of the most commonly-used ways to create a set of build files to construct the library or executable. It's a meta-build system since does not build anything itself: it creates the files that we *then* use to build, e.g., generating Makefiles to run `make`. Learning CMake is challenging since tutorials and the official CMake documentation and public projects either range from constructing the very basic "Hello World" to constructing multi-platform, multi-compiler submodular libraries. In other words, the complexity is often binary from "let's build this one C++ file!" to "let's build something like Boost!" The majority of times, I've found that a CMake structure somewhere in between tends to be good enough for most projects.

In this post, I'll describe a good-enough C++ library project structure and CMake file that accomplishes enough to build a fairly flexible library for a client to build from scratch and use (or some automated build system to generate binaries). To concretely demonstrate this, I've started on a catch-all miscellaneous C++ library called [bagel](https://github.com/mohitd/bagel), named after an "everything bagel" that I had for breakfast that day 😄, that I'm going to be using as a C++ playground going forward.

I don't intend for this to be a CMake tutorial for complete beginners; I'll assume you have enough CMake knowledge where I won't have to explain syntax or basic commands like `set` or `project`. The purpose of this post is to talk more about how to use that CMake knowledge to create a project structure that makes building easy and flexible.

# A Good-enough Project Structure

Before getting into the CMake file, let's describe a good-enough directory structure for a mid-sized project:

```
.
├── CMakeLists.txt
├── LICENSE
├── Readme.md
├── cmake
│   └── Config.cmake.in
├── examples
│   ├── CMakeLists.txt
│   └── timer.cpp
├── include
│   └── bagel
│       ├── chrono
│       │   └── timer.hpp
│       └── export.hpp
├── src
│   └── chrono
│       └── timer.cpp
└── tests
    ├── CMakeLists.txt
    └── chrono
        └── test_timer.cpp

10 directories, 11 files
```

In this directory, we have a few "required" files like `Readme.md` and `LICENSE` that provide an overall description of the library (among many other things) as well as the legal software license it falls under. Often times open-source libraries have more files like `CONTRIBUTING.md` and `AUTHORS.md` that explain how to contribute to the library and the core authors of the library, respectively.

The crux of building the library is in the `CMakeLists.txt` which is the CMake file that's used by the CMake executable to write the Makefiles used to actually build this library; it contains the actual library definition including things like compile options and where to install the headers and whatnot. When we run the CMake command in a directory like `cmake .`, it will search for `CMakeLists.txt` in that directory and parse and execute it. A related directory we'll cover in the later sections is `cmake`, which tends to store auxiliary CMake files used by the root `CMakeLists.txt`.

The next directory `examples` contains example usage of the library with its own `CMakeLists.txt` that just builds the examples. This allows the builder to control if they want to build examples or not. In this case, `examples` is flat, but it could be more hierarchical if we had a larger library. We'll get to this definition later as well. The `tests` directory contains our tests for the library and it's own `CMakeLists.txt` for the same reason as the `examples` directory. We use GoogleTest to validate our library, but any testing framework will do. I'd highly recommend having tests for your libraries so it helps provide credibility and confidence to users that your library actually does what it intends to do.

The next two directories `include` and `src` contain the actual content of our library. In the case of `include`, we have some subdirectories, the main one being the name of the library `bagel`. Then we have subdirectories for the subcomponents like `chrono`. The reason we use a subdirectory `bagel` with the same name as the project is so that, when we install the header files, e.g., to a place like `/usr/local/include` in a Linux system, that our headers like `timer.hpp` are prefixed by the library folder to avoid overwriting some other file named `timer.hpp` from some other library.

We'll see most of these directories play a part in the project-level `CMakeLists.txt`. The focus for this post is on the CMake required to build our library and not on what the library itself actually does so we won't necessarily talk about *what* `timer.hpp`/`timer.cpp` contains. The contents aren't as important as how we *build* the contents into a library.

# Anatomy of a Good-enough CMakeLists.txt

Building a project starts with the `CMakeLists.txt` file that defines the project, build artifacts, and other options. I like to divide the CMake into several larger sections:

1. **Preamble**: define the entire CMake project as a whole.
2. **Configuration**: check any project-level variables and configure building examples and tests
3. **Build**: define the library and its associated source files, compile options, versions, and other properties
4. **Install**: configure where to install the library and headers
5. **Extra stuff**: recurse into directories for tests and examples as well as build documentation

## Preamble

The preamble defines the minimum CMake binary version as well as defines the project.

```cmake
cmake_minimum_required(VERSION 3.14)
project(bagel
    VERSION 0.1.0
    DESCRIPTION "An everything bagel of C++"
    LANGUAGES CXX)
```

In general, using a too-recent version of CMake can make it difficult for developers to use your library since not everyone might be able to use the latest version of CMake, especially in industry where upgrades to newer build tools can be very slow. For the versioning, [semantic versioning](https://semver.org/) is usually a popular choice.

##  Configuration

After defining the root CMake project, we define some project-level configurations and check some variables. One of the first configurations we'll provide to builders is the ability to build our code into a shared or a static library. A shared library (also called shared object hence the `.so` file extension) is a kind of library that is dynamically loaded into an executable at runtime; these kinds of libraries make the overall executable smaller but, since the library is loaded dynamically at runtime, the executable requires the shared library to be located in the right place in the filesystem otherwise the exectuable fails when you run it. On the other hand, a static library (file extension `.a` for archive) is the other kind of library that is actually built *into* an executable at compile-time; these kinds of libraries make the executable larger but, since they're built into the executable, it ensures the exectuable is self-sufficient.

CMake allows the builder to specific which kind of library they want to build. There's a built-in variable called `BUILD_SHARED_LIBS`. However, since this is general to all CMake libraries and is coupled to other CMake behavior, oftentimes we provide a project-specific override usually called something like `${PROJECT_NAME}_SHARED_LIBS`. If that is defined, then we can use it, otherwise, we can default to whatever the `BUILD_SHARED_LIBS` variable decides. The default option is to build static libraries.

One nuance is that we want the variable to be defined like `BAGEL_BUILD_SHARED_LIBS` not `bagel_BUILD_SHARED_LIBS` for consistencency so we'll define a `${UPPER_PROJECT_NAME}` variable that's just `${PROJECT_NAME}` but uppercase.

```cmake
set(namespace ${PROJECT_NAME})
string(TOUPPER ${PROJECT_NAME} UPPER_PROJECT_NAME)

message(CHECK_START "Checking ${UPPER_PROJECT_NAME}_SHARED_LIBS")
if(DEFINED ${UPPER_PROJECT_NAME}_SHARED_LIBS)
    set(BUILD_SHARED_LIBS ${UPPER_PROJECT_NAME}_SHARED_LIBS)
    message(CHECK_PASS "${${UPPER_PROJECT_NAME}_SHARED_LIBS}")
else()
    message(CHECK_FAIL "${BUILD_SHARED_LIBS}")
endif()

message(CHECK_START "Building shared libraries")
if(BUILD_SHARED_LIBS)
    message(CHECK_PASS "yes")
else()
    message(CHECK_FAIL "no")
endif()
```

We're also defining a `${namespace}` that we'll use later. To write things to the screen, we use the `message` macro but use the `CHECK_START`, `CHECK_PASS`, and `CHECK_FAIL` settings so that CMake formats our message nicely like the following.

```
[cmake] -- Checking BAGEL_SHARED_LIBS
[cmake] -- Checking BAGEL_SHARED_LIBS - ON
[cmake] -- Building shared libraries
[cmake] -- Building shared libraries - yes
```

In CMake, like in bash, there's a difference between a variable existing and not existing and a variable having a value. We first check if the variable `${UPPER_PROJECT_NAME}_SHARED_LIBS` exists. Note that we don't use `${}` around the entire expression since we're not checking if the contents of the variable exist, we want to check if the variable itself exists. If the variable is defined, then we override the value of `BUILD_SHARED_LIBS`, otherwise we default to `BUILD_SHARED_LIBS`. If that also doesn't exist, then we'll use CMake's default (building a static library).

There are several ways to set these variables. One way is to do it using the `cmake` command like `cmake -DMY_VAR` to set define `MY_VAR`.

Another common CMake configuration is the build type. The build type mostly sets the compiler optimizations and options such as debug symbols. The most commonly-used ones are `Debug`, `Release`, and `RelWithDebInfo`. `Debug` has minimal optimizations but retains debug symbols; `Release` has the strongest optimizations but strips any debug symbols for debugging through a debugger like gdb. The last one has the optimizations of release mode but still contains debug symbols. Similar to `BUILD_SHARED_LIBS`, if `CMAKE_BUILD_TYPE` isn't defined, we'll default to Release mode since that's what builders of our library will tend to use.

```cmake
if(NOT DEFINED CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
message(STATUS "Setting build type: ${CMAKE_BUILD_TYPE}")
```

Using the `CACHE` and `FORCE` options, we override whatever user-defined value is set in the cache with this value; this is fine since the user didn't specify a `CMAKE_BUILD_TYPE` in the first place. The `STRING "Build type"` tells CMake that `CMAKE_BUILD_TYPE` is a string.

Next we set some variables for later and define some custom other build options like building examples, tests, and documentation.

```cmake
set(export_header_name "export.hpp")
set(export_file_name "${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME}/${export_header_name}")

include(GNUInstallDirs)
set(cmake_config_dir ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})
set(build_tests ${UPPER_PROJECT_NAME}_BUILD_TESTS)
set(build_examples ${UPPER_PROJECT_NAME}_BUILD_EXAMPLES)
set(build_docs ${UPPER_PROJECT_NAME}_BUILD_DOCS)

option(${build_tests} "Builds tests" OFF)
option(${build_examples} "Builds examples" OFF)
option(${build_docs} "Builds docs" OFF)
```

We use a few CMake variables:

*  `${CMAKE_CURRENT_SOURCE_DIR}`: the directory being processed by CMake; in our case, since our library itself is a top-level CMake project itself, this is the root of the project. This usually refers to the root of the project for single-project CMakes.
* `${CMAKE_INSTALL_LIBDIR}`: the install directory for libraries; in Linux systems, this is usually called `lib` (or sometimes `lib32` and `lib64`). Note that the install prefix is prepended to this folder. Since we used `include(GNUInstallDirs)` earlier, it will set this folder correctly for us.

We'll discuss the export header and config directory later.

## Build

Now we're getting into actually building the library. First thing we'll do is define the library itself and an alias.

```cmake
add_library(${PROJECT_NAME})
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})
```

The alias is so that, if someone was building our library from source and linking it as part of their library, then the `target_link_libraries` would look the same. We're not adding any sources to it yet, just defining the library's existence. After we define the library, we also set the minimum C++ version and provide some compile-time options.

```cmake
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)
target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra)
```

We use `PUBLIC` for the minimum C++ version so that it's visible to users when they try to link against our library. For the compile options, those are `PRIVATE` since they're only applicable to our library; we don't want our library decisions on warnings and errors to be propagated to all of our users!

C++ provides access specifiers like `public` and `private`, but when building a shared library, we also kind of have a notion of "library" visibility. For shared libraries, each class and function defines a symbol in the symbol table of the library. When you link the shared library to an executable (or other library), the linker resolves those symbols to actual memory addresses. Think of them as placeholders and the actual interface that your library itself provides (sometimes called its ABI or Application Binary Interface). By default, *all* defined symbols (except the defined inline ones) are exported by our shared library. However, sometimes we have some internal classes or functions that we don't want to export as part of the shared library interface. It would be better to explicitly mark which symbols should be part of our library's interface and default all other symbols to be hidden. The asymmetry is that, for static libraries, we don't have this distinction since the static library is built into the executable in its entirety; the linker doesn't apply such symbol visibility to static libraries. So we have a few criteria we need to satisfy:

1. By default, hide all symbols
2. Provide a mechanism to manually export symbols
3. Ignore the export symbol mechanism for static libraries

CMake handles this by generating an export header that can create a symbol like `BAGEL_EXPORT` that'll export symbols for shared libraries but it becomes a no-op operation for static libraries.

```cmake
if(NOT BUILD_SHARED_LIBS)
    target_compile_definitions(${PROJECT_NAME} PUBLIC ${UPPER_PROJECT_NAME}_STATIC_DEFINE)
endif()

include(GenerateExportHeader)
generate_export_header(${PROJECT_NAME}
    EXPORT_FILE_NAME ${export_file_name}
)
```

The first part will add a macro definition `BAGEL_STATIC_DEFINE` that will no-op `BAGEL_EXPORT`. The `generate_export_header` will auto-generate a header file at `${export_file_name}` that will define macros to change the visibility of a symbol. To export certain classes or functions, we can import that header and use `BAGEL_EXPORT` right before the symbol name like the following.

```cpp
class BAGEL_EXPORT MyClass {
    ...
};

void BAGEL_EXPORT myFunc() {
    ...
}
```

If we inspect the symbol table of shared library, we'll see only those symbols exported while others won't be. For a class, exporting the class exports all symbols but the export header also defined a `BAGEL_NO_EXPORT` that "un-exports" the symbol again.

The last thing we need to do is to disable exporting all symbols by default.

```cmake
if(NOT DEFINED CMAKE_CXX_VISIBILITY_PRESET)
    set_target_properties(${PROJECT_NAME} PROPERTIES
        CXX_VISIBILITY_PRESET hidden
    )
endif()
if(NOT DEFINED CMAKE_VISIBILITY_INLINES_HIDDEN)
    set_target_properties(${PROJECT_NAME} PROPERTIES
        VISIBILITY_INLINES_HIDDEN ON
    )
endif()
```

That finishes our symbol exporting stuff. Moving on, one minor thing we'll do is also set our library's version based on what we set in the `project()` macro.

```cmake
set_target_properties(${PROJECT_NAME} PROPERTIES
    SOVERSION ${PROJECT_VERSION_MAJOR}
    VERSION ${PROJECT_VERSION}
)
```

After all of that, we're finally ready to actually add header and source files.

```cmake
target_include_directories(${PROJECT_NAME}
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)
target_sources(${PROJECT_NAME} PRIVATE
    src/chrono/timer.cpp)
```

We use `target_include_directories` to add headers to our library. The `PRIVATE` part means that only our source files in our `src` can also access headers in our `src` directory but external users can't (since those are meant to be for library use only). For the `PUBLIC` part, we use CMake generators to specify a build and install interface. When building the library, we can also use headers in the `include` directory directly; for users, they'll use headers wherever we've install them as part of the install stage. Recall that `${CMAKE_INSTALL_INCLUDEDIR}` is just like `${CMAKE_INSTALL_LIBDIR}` but for includes instead of libraries (set to `include` by `GNUInstallDirs`).

`target_sources` adds sources to our library and `PRIVATE` is really the only thing that makes sense here. We could also glob all source files under the `src` directory but I like to be more explicit about which source files are added to the library.

## Install

At this point, we have our library and header files ready and we just need to install them in a way so that users can find the library and link against it. The ideal user experience is to be as simple as possible.

```cmake
find_package(bagel REQUIRED)
target_link_libraries(${PROJECT_NAME} bagel)
```

These two lines should be all that's required to link against the installed library. So how can we accomplish this? First thing we need to do is install the headers. There's a `PUBLIC_HEADERS` field but that doesn't work so nicely for nested directory structures. I've found it easier to just install the entire `include` directory into the right place.

```cmake
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME}"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
```

This does go against my previous sentiment about being more explicit about which files are added to the library but we've already configured out project to find headers in the `src` directory only for our project so we have a mechanism to keep some headers private. The next thing we need to install is our actual library itself and associate the headers with it.

```cmake
install(TARGETS ${PROJECT_NAME}
    EXPORT "${PROJECT_NAME}Targets"
)
```

Installing the library isn't enough: we need to create an export target for our library that describes how to find the header files and library file from the target itself. We'll use the export target we just created and create a corresponding `*Targets.cmake` for it. We'll give it a namespace; this is a more modern way for CMake to know that a particular alias is a build target and not a folder or something else.

```cmake
install(EXPORT "${PROJECT_NAME}Targets"
    FILE "${PROJECT_NAME}Targets.cmake"
    NAMESPACE ${namespace}::
    DESTINATION ${cmake_config_dir}
)
```

We'll get to why we're installing this into `${cmake_config_dir}` in just a second.

The last thing we need is to write a package config so that `find_package` (using pkg-config) in a client `CMakeLists.txt` can actually find it and import the build target. First thing we'll do is write a version config file, but there's a helper we can use.

```cmake
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY SameMajorVersion
)
```

Recall `${CMAKE_CURRENT_BINARY_DIR}` is the location of the build directory; this is fine since we'll be installing these generated files immediately anyways. We're setting the compatibility to be the `SameMajorVersion` since, under our semantic versioning scheme, there are no breaking changes across major versions. Next thing we need to create is a config file that imports our previously-created target file. For that, first we create a separate `Config.cmake.in`. 

```cmake
@PACKAGE_INIT@

include("${CMAKE_CURRENT_LIST_DIR}/@PROJECT_NAME@Targets.cmake")

check_required_components(@PROJECT_NAME@)
```

Some of this is a bit esoteric, but the documentation says to ensure `@PACKAGE_INIT@` is at the start and `check_required_components(@PROJECT_NAME@)` is at the bottom. In the middle, all we have to do is include our targets file. Finally, we install both of these to the right location.

```cmake
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
    DESTINATION ${cmake_config_dir}
)
```

Note that we install the package config files and the targets file to the `${cmake_config_dir}` we defined earlier. This effectively installs to a filepath like `lib/bagel/cmake` on a Linux system. This is where pkg-config looks when you write `find_package(bagel)`: it'll go through the folder of each library stored in `lib` and look for a `cmake` diretory. If we were to put it somewhere else, we'd get some error like the following.

```
CMake Error at CMakeLists.txt:6 (find_package):
  Could not find a package configuration file provided by "bagel" with any of
  the following names:

    bagelConfig.cmake
    bagel-config.cmake

  Add the installation prefix of "bagel" to CMAKE_PREFIX_PATH or set
  "bagel_DIR" to a directory containing one of the above files.  If "bagel"
  provides a separate development package or SDK, be sure it has been
  installed.
```

Alternatively, we could install this anywhere and append to the `CMAKE_PREFIX_PATH` or define a `bagel_DIR`, but it's convenient to have the right suffix location by default so clients don't have to do that extra step. Of course a client could add an install prefix to anywhere but then it's on them to set either of the two variables above.

## Extra stuff

At this point, we technically have everything we need for our library, but let's also provide a way to build examples, tests, and documentation. In the project-level `CMakeLists.txt`, we just need to recurse into the lower-level `CMakeLists.txt`.

```cmake
message(CHECK_START "Building tests")
if(${build_tests})
    message(CHECK_PASS "yes")

    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tests)
else()
    message(CHECK_FAIL "no")
endif()

message(CHECK_START "Building examples")
if(${build_examples})
    message(CHECK_PASS "yes")

    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/examples)
else()
    message(CHECK_FAIL "no")
endif()
```

We'll get into those in a minute but building documentation relies on Doxygen and there are some CMake variables that can be set and the `doxygen_add_docs` command generates docs. One additional thing we can do is to create a dependency in our project to our `generate_docs` target so that, whenever we rebuild the library due to a code change, the documentation will automatically be re-generated too!

```cmake
message(CHECK_START "Building docs")
if(${build_docs})
    message(CHECK_PASS "yes")

    find_package(Doxygen REQUIRED)
    
    set(README_PATH "${CMAKE_CURRENT_SOURCE_DIR}/Readme.md")
    set(DOXYGEN_PROJECT_NAME "${PROJECT_NAME}")
    set(DOXYGEN_PROJECT_BRIEF "${PROJECT_DESCRIPTION}")
    set(DOXYGEN_USE_MDFILE_AS_MAINPAGE "${README_PATH}")
    doxygen_add_docs(generate_docs include "${README_PATH}"
        COMMENT "Generating docs")
    add_dependencies(${PROJECT_NAME} generate_docs)
else()
    message(CHECK_FAIL "no")
endif()
```

Alternatively, we could use a backup documentation generator and not make Doxygen required but that's a choice.

The `CMakeLists.txt` in the examples folder is fairly straightforward

```cmake
cmake_minimum_required(VERSION 3.16)
project(bagel-examples)

add_executable(timer timer.cpp)
target_link_libraries(timer PRIVATE bagel::bagel)
```

Notice how we link our example executable to our library with `bagel::bagel` using `PRIVATE` since we have an executable.

Tests are slightly more complicated beacuse of downloading and using GoogleTest, but still readable.

```cmake
cmake_minimum_required(VERSION 3.16)
project(bagel-tests)

set(INSTALL_GTEST OFF)

enable_testing()

include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

add_executable(test_timer chrono/test_timer.cpp)
target_link_libraries(test_timer
    PRIVATE
        bagel::bagel
        GTest::gtest_main
)
gtest_discover_tests(test_timer)
```

Again notice how we link our library to a test binary (and also to the GoogleTest binary).

To evaluate if we did everything correct, I created a dummy C++ executable for testing purposes. The `main.cpp` simply imports the header and does some work.

```cpp
#include <chrono>
#include <iostream>
#include <thread>
#include <bagel/chrono/timer.hpp>

using namespace std::chrono_literals;

int main(int argc, char** argv) {
    bagel::WallTimer t;
    t.start();
    std::this_thread::sleep_for(10ms);
    auto elapsed = t.stop();
    std::cout << elapsed.count() << "s\n";
    return 0;
}
```

We create a timer, intentionally pause the main thread for about 10ms, stop the timer, and record the value in the timer.

The `CMakeLists.txt` simply defines an executable and links against our library. Since I've installed the library to a custom location for development purposes, I'm manually appending the location to the `CMAKE_PREFIX_PATH`.

```cmake
cmake_minimum_required(VERSION 3.14)
project(hungry)

list(APPEND CMAKE_PREFIX_PATH "/Users/mohit/Developer/bagel/install/")

find_package(bagel CONFIG REQUIRED)

add_executable(hungry main.cpp)

target_link_libraries(hungry PRIVATE bagel::bagel)
```

Now we can create a build directory, run cmake, build our executable, and run it!

```bash
mkdir build && cd build
cmake ..
make
./hungry
```

The output is what we expect: a value close to 10ms (a little off depending on your scheduler).

```
0.012527s
```

# Conclusion

CMake is the most popular meta-build system to build C++ libraries and executables, but it's also one of the most challenging ones to learn well. In this post, we went over a project structure and `CMakeLists.txt` for a medium-sized project with multiple subcomponents. We broke the `CMakeLists.txt` down into a parts: (i) preamble, (ii) configuration, (iii) building, (iv) installing, and (v) extra stuff. In (i), we simply define the project. In (ii), we define some variables that clients can use to configure how they build our library. (iii) is where we actually build the library and set things like include directories. After building the library, (iv) is where we install it and the headers in a way and place where clients can easily link against it. Finally, (v) is where we build optional things like examples, tests, and documentation.

CMake can be pretty complicated to "get right" and there's a lot of variability in how developers use CMake to write libraries and executables. Hopefully this little tutorial provides some guidance on how to provide more structure your `CMakeLists.txt` abiding to some best practices to avoid. If you're working on C++ stuff, try to crystallize some of this guidance into your team or project's standards and let me know how it goes 🙂
