# CMAKE generated file: DO NOT EDIT!
# Generated by "MinGW Makefiles" Generator, CMake Version 3.17

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = "C:\App2\CLion 2020.2.4\bin\cmake\win\bin\cmake.exe"

# The command to remove a file.
RM = "C:\App2\CLion 2020.2.4\bin\cmake\win\bin\cmake.exe" -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = C:\work\ArmSparse

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = C:\work\ArmSparse\cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/ArmSparse.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/ArmSparse.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/ArmSparse.dir/flags.make

CMakeFiles/ArmSparse.dir/sparseMatrixAdd.cpp.obj: CMakeFiles/ArmSparse.dir/flags.make
CMakeFiles/ArmSparse.dir/sparseMatrixAdd.cpp.obj: CMakeFiles/ArmSparse.dir/includes_CXX.rsp
CMakeFiles/ArmSparse.dir/sparseMatrixAdd.cpp.obj: ../sparseMatrixAdd.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\work\ArmSparse\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/ArmSparse.dir/sparseMatrixAdd.cpp.obj"
	C:\PROGRA~2\Dev-Cpp\MinGW64\bin\G__~1.EXE  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles\ArmSparse.dir\sparseMatrixAdd.cpp.obj -c C:\work\ArmSparse\sparseMatrixAdd.cpp

CMakeFiles/ArmSparse.dir/sparseMatrixAdd.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ArmSparse.dir/sparseMatrixAdd.cpp.i"
	C:\PROGRA~2\Dev-Cpp\MinGW64\bin\G__~1.EXE $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\work\ArmSparse\sparseMatrixAdd.cpp > CMakeFiles\ArmSparse.dir\sparseMatrixAdd.cpp.i

CMakeFiles/ArmSparse.dir/sparseMatrixAdd.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ArmSparse.dir/sparseMatrixAdd.cpp.s"
	C:\PROGRA~2\Dev-Cpp\MinGW64\bin\G__~1.EXE $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S C:\work\ArmSparse\sparseMatrixAdd.cpp -o CMakeFiles\ArmSparse.dir\sparseMatrixAdd.cpp.s

# Object files for target ArmSparse
ArmSparse_OBJECTS = \
"CMakeFiles/ArmSparse.dir/sparseMatrixAdd.cpp.obj"

# External object files for target ArmSparse
ArmSparse_EXTERNAL_OBJECTS =

ArmSparse.exe: CMakeFiles/ArmSparse.dir/sparseMatrixAdd.cpp.obj
ArmSparse.exe: CMakeFiles/ArmSparse.dir/build.make
ArmSparse.exe: CMakeFiles/ArmSparse.dir/linklibs.rsp
ArmSparse.exe: CMakeFiles/ArmSparse.dir/objects1.rsp
ArmSparse.exe: CMakeFiles/ArmSparse.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=C:\work\ArmSparse\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ArmSparse.exe"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles\ArmSparse.dir\link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/ArmSparse.dir/build: ArmSparse.exe

.PHONY : CMakeFiles/ArmSparse.dir/build

CMakeFiles/ArmSparse.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles\ArmSparse.dir\cmake_clean.cmake
.PHONY : CMakeFiles/ArmSparse.dir/clean

CMakeFiles/ArmSparse.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" C:\work\ArmSparse C:\work\ArmSparse C:\work\ArmSparse\cmake-build-debug C:\work\ArmSparse\cmake-build-debug C:\work\ArmSparse\cmake-build-debug\CMakeFiles\ArmSparse.dir\DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/ArmSparse.dir/depend

