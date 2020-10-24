##### android aarch64
export ANDROID_NDK=/home/ziyu/Software/android-ndk-r21d
mkdir -p build-android-aarch64
pushd build-android-aarch64
<<<<<<< HEAD
cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI="arm64-v8a" -DANDROID_PLATFORM=android-21 ..
=======
cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI="arm64-v8a" -DANDROID_PLATFORM=android-21 -DNCNN_AVX2=OFF -DNCNN-ARM82=OFF ..
>>>>>>> 07c83f5a6d3e894bad474e9cf2e18879794b73b2
make -j4
make install
popd

