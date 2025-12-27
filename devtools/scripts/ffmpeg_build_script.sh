#! /bin/bash

# Setup dependencies as per 
# https://trac.ffmpeg.org/wiki/CompilationGuide/Ubuntu
# https://docs.nvidia.com/video-technologies/video-codec-sdk/12.0/ffmpeg-with-nvidia-gpu/index.html

cd $HOME/ffmpeg_sources/ffmpeg

PKG_CONFIG_PATH="$HOME/ffmpeg_build/lib/pkgconfig:${PKG_CONFIG_PATH}" PATH="$HOME/ffmpeg_bin:$PATH" ./configure \
  --prefix="$HOME/ffmpeg_build" \
  --pkg-config-flags="--static" \
  --extra-cflags="-I$HOME/ffmpeg_build/include" \
  --extra-ldflags="-L$HOME/ffmpeg_build/lib" \
  --extra-libs="-lpthread -lm" \
  --ld="g++" \
  --bindir="$HOME/ffmpeg_bin" \
  --enable-gpl \
  --enable-gnutls \
  --enable-libsvtav1 \
  --enable-libdav1d \
  --enable-libx264 \
  --enable-libx265 \
  --enable-nonfree \
  --enable-cuda-nvcc \
  --enable-libnpp \
  --extra-cflags=-I/usr/local/cuda/include \
  --extra-ldflags=-L/usr/local/cuda/lib64 \
  --enable-static \
  --enable-libvmaf \
  --enable-libdav1d
 