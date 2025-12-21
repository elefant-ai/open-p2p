# Recap

### What is Recap?
Recap is a tool to record a window and user input.
the video is saved as a mp4 encoded with h264 and the input is saved as a json file.

### How to use Recap?
for develop run
`just recap dev`
this will run the app in dev mode.

for release run
`just recap run`

run in trace mode
`just recap trace`

after running the app there will be a simple ui to select a window, location to save the video and location to save the input.
then press the record button to start recording.

### Info
on start recording the app will beep once and on stop recording the app will beep twice then after recording is all saved long beep.


### Debugging
if you run `just recap trace` and set `GST_DEBUG_DUMP_DOT_DIR="some_path"` then the app will dump a dot file in that directory of the pipeline.
