##### android aarch64
export ANDROID_NDK=/home/ziyu/Software/android-ndk-r21d
mkdir -p build-android-aarch64
pushd build-android-aarch64
cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI="arm64-v8a" -DANDROID_PLATFORM=android-21 -DNCNN_AVX2=OFF -DNCNN-ARM82=OFF ..
make -j4
make install
popd

