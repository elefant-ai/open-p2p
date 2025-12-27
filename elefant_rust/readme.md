## Elefant rust


## setup ffmpeg
install full shared version from https://ffmpeg.org/download.html

### WSL
`sudo apt install -y clang pkg-config libavcodec-dev libavdevice-dev libavfilter-dev libavformat-dev libavutil-dev libswscale-dev`

add bin to path
set env variable FFMPEG_DIR to the ffmpeg directory like this:
```powershell   
$env.FFMPEG_DIR = 'C:\libs\ffmpeg-7.1.1-full_build-shared'
```

install clang
https://github.com/llvm/llvm-project/releases
set LIBCLANG_PATH to the clang directory like this:
```powershell
$env.LIBCLANG_PATH = 'C:\libs\llvm\bin\'
```