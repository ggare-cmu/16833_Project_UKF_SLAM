# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/grg/grg/SLAM/MultiCol/MultiCol-SLAM

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/grg/grg/SLAM/MultiCol/MultiCol-SLAM/build

# Include any dependencies generated for this target.
include CMakeFiles/multi_col_slam_lafida.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/multi_col_slam_lafida.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/multi_col_slam_lafida.dir/flags.make

CMakeFiles/multi_col_slam_lafida.dir/Examples/Lafida/mult_col_slam_lafida.o: CMakeFiles/multi_col_slam_lafida.dir/flags.make
CMakeFiles/multi_col_slam_lafida.dir/Examples/Lafida/mult_col_slam_lafida.o: ../Examples/Lafida/mult_col_slam_lafida.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/grg/grg/SLAM/MultiCol/MultiCol-SLAM/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/multi_col_slam_lafida.dir/Examples/Lafida/mult_col_slam_lafida.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/multi_col_slam_lafida.dir/Examples/Lafida/mult_col_slam_lafida.o -c /home/grg/grg/SLAM/MultiCol/MultiCol-SLAM/Examples/Lafida/mult_col_slam_lafida.cpp

CMakeFiles/multi_col_slam_lafida.dir/Examples/Lafida/mult_col_slam_lafida.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/multi_col_slam_lafida.dir/Examples/Lafida/mult_col_slam_lafida.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/grg/grg/SLAM/MultiCol/MultiCol-SLAM/Examples/Lafida/mult_col_slam_lafida.cpp > CMakeFiles/multi_col_slam_lafida.dir/Examples/Lafida/mult_col_slam_lafida.i

CMakeFiles/multi_col_slam_lafida.dir/Examples/Lafida/mult_col_slam_lafida.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/multi_col_slam_lafida.dir/Examples/Lafida/mult_col_slam_lafida.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/grg/grg/SLAM/MultiCol/MultiCol-SLAM/Examples/Lafida/mult_col_slam_lafida.cpp -o CMakeFiles/multi_col_slam_lafida.dir/Examples/Lafida/mult_col_slam_lafida.s

CMakeFiles/multi_col_slam_lafida.dir/Examples/Lafida/mult_col_slam_lafida.o.requires:

.PHONY : CMakeFiles/multi_col_slam_lafida.dir/Examples/Lafida/mult_col_slam_lafida.o.requires

CMakeFiles/multi_col_slam_lafida.dir/Examples/Lafida/mult_col_slam_lafida.o.provides: CMakeFiles/multi_col_slam_lafida.dir/Examples/Lafida/mult_col_slam_lafida.o.requires
	$(MAKE) -f CMakeFiles/multi_col_slam_lafida.dir/build.make CMakeFiles/multi_col_slam_lafida.dir/Examples/Lafida/mult_col_slam_lafida.o.provides.build
.PHONY : CMakeFiles/multi_col_slam_lafida.dir/Examples/Lafida/mult_col_slam_lafida.o.provides

CMakeFiles/multi_col_slam_lafida.dir/Examples/Lafida/mult_col_slam_lafida.o.provides.build: CMakeFiles/multi_col_slam_lafida.dir/Examples/Lafida/mult_col_slam_lafida.o


# Object files for target multi_col_slam_lafida
multi_col_slam_lafida_OBJECTS = \
"CMakeFiles/multi_col_slam_lafida.dir/Examples/Lafida/mult_col_slam_lafida.o"

# External object files for target multi_col_slam_lafida
multi_col_slam_lafida_EXTERNAL_OBJECTS =

../Examples/Lafida/multi_col_slam_lafida: CMakeFiles/multi_col_slam_lafida.dir/Examples/Lafida/mult_col_slam_lafida.o
../Examples/Lafida/multi_col_slam_lafida: CMakeFiles/multi_col_slam_lafida.dir/build.make
../Examples/Lafida/multi_col_slam_lafida: ../lib/libMultiCol-SLAM.so
../Examples/Lafida/multi_col_slam_lafida: /usr/local/lib/libopencv_dnn.so.4.3.0
../Examples/Lafida/multi_col_slam_lafida: /usr/local/lib/libopencv_gapi.so.4.3.0
../Examples/Lafida/multi_col_slam_lafida: /usr/local/lib/libopencv_highgui.so.4.3.0
../Examples/Lafida/multi_col_slam_lafida: /usr/local/lib/libopencv_ml.so.4.3.0
../Examples/Lafida/multi_col_slam_lafida: /usr/local/lib/libopencv_objdetect.so.4.3.0
../Examples/Lafida/multi_col_slam_lafida: /usr/local/lib/libopencv_photo.so.4.3.0
../Examples/Lafida/multi_col_slam_lafida: /usr/local/lib/libopencv_stitching.so.4.3.0
../Examples/Lafida/multi_col_slam_lafida: /usr/local/lib/libopencv_video.so.4.3.0
../Examples/Lafida/multi_col_slam_lafida: /usr/local/lib/libopencv_calib3d.so.4.3.0
../Examples/Lafida/multi_col_slam_lafida: /usr/local/lib/libopencv_features2d.so.4.3.0
../Examples/Lafida/multi_col_slam_lafida: /usr/local/lib/libopencv_flann.so.4.3.0
../Examples/Lafida/multi_col_slam_lafida: /usr/local/lib/libopencv_videoio.so.4.3.0
../Examples/Lafida/multi_col_slam_lafida: /usr/local/lib/libopencv_imgcodecs.so.4.3.0
../Examples/Lafida/multi_col_slam_lafida: /usr/local/lib/libopencv_imgproc.so.4.3.0
../Examples/Lafida/multi_col_slam_lafida: /usr/local/lib/libopencv_core.so.4.3.0
../Examples/Lafida/multi_col_slam_lafida: /usr/local/lib/libpangolin.so
../Examples/Lafida/multi_col_slam_lafida: /usr/lib/x86_64-linux-gnu/libGL.so
../Examples/Lafida/multi_col_slam_lafida: /usr/lib/x86_64-linux-gnu/libGLU.so
../Examples/Lafida/multi_col_slam_lafida: /usr/lib/x86_64-linux-gnu/libGLEW.so
../Examples/Lafida/multi_col_slam_lafida: /usr/lib/x86_64-linux-gnu/libSM.so
../Examples/Lafida/multi_col_slam_lafida: /usr/lib/x86_64-linux-gnu/libICE.so
../Examples/Lafida/multi_col_slam_lafida: /usr/lib/x86_64-linux-gnu/libX11.so
../Examples/Lafida/multi_col_slam_lafida: /usr/lib/x86_64-linux-gnu/libXext.so
../Examples/Lafida/multi_col_slam_lafida: /usr/lib/x86_64-linux-gnu/libdc1394.so
../Examples/Lafida/multi_col_slam_lafida: /usr/lib/x86_64-linux-gnu/libpng.so
../Examples/Lafida/multi_col_slam_lafida: /usr/lib/x86_64-linux-gnu/libz.so
../Examples/Lafida/multi_col_slam_lafida: /usr/lib/x86_64-linux-gnu/libjpeg.so
../Examples/Lafida/multi_col_slam_lafida: /usr/lib/x86_64-linux-gnu/libtiff.so
../Examples/Lafida/multi_col_slam_lafida: /usr/lib/x86_64-linux-gnu/libIlmImf.so
../Examples/Lafida/multi_col_slam_lafida: ../ThirdParty/g2o/lib/libg2o.so
../Examples/Lafida/multi_col_slam_lafida: ../ThirdParty/DBoW2/lib/libDBoW2.so
../Examples/Lafida/multi_col_slam_lafida: ../ThirdParty/OpenGV/build/lib/librandom_generators.a
../Examples/Lafida/multi_col_slam_lafida: ../ThirdParty/OpenGV/build/lib/libopengv.a
../Examples/Lafida/multi_col_slam_lafida: CMakeFiles/multi_col_slam_lafida.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/grg/grg/SLAM/MultiCol/MultiCol-SLAM/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../Examples/Lafida/multi_col_slam_lafida"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/multi_col_slam_lafida.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/multi_col_slam_lafida.dir/build: ../Examples/Lafida/multi_col_slam_lafida

.PHONY : CMakeFiles/multi_col_slam_lafida.dir/build

CMakeFiles/multi_col_slam_lafida.dir/requires: CMakeFiles/multi_col_slam_lafida.dir/Examples/Lafida/mult_col_slam_lafida.o.requires

.PHONY : CMakeFiles/multi_col_slam_lafida.dir/requires

CMakeFiles/multi_col_slam_lafida.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/multi_col_slam_lafida.dir/cmake_clean.cmake
.PHONY : CMakeFiles/multi_col_slam_lafida.dir/clean

CMakeFiles/multi_col_slam_lafida.dir/depend:
	cd /home/grg/grg/SLAM/MultiCol/MultiCol-SLAM/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/grg/grg/SLAM/MultiCol/MultiCol-SLAM /home/grg/grg/SLAM/MultiCol/MultiCol-SLAM /home/grg/grg/SLAM/MultiCol/MultiCol-SLAM/build /home/grg/grg/SLAM/MultiCol/MultiCol-SLAM/build /home/grg/grg/SLAM/MultiCol/MultiCol-SLAM/build/CMakeFiles/multi_col_slam_lafida.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/multi_col_slam_lafida.dir/depend

