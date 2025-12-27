"""Generated protocol buffer code."""

from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database

_sym_db = _symbol_database.Default()
from . import shared_pb2 as shared__pb2

DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(
    b'\n\x16video_annotation.proto\x12&elefant.training_data.video_annotation\x1a\x0cshared.proto"K\n\x12VideoAnnotationEnv\x12\x0b\n\x03env\x18\x01 \x01(\t\x12\x13\n\x0benv_subtype\x18\x02 \x01(\t\x12\x13\n\x0benv_version\x18\x03 \x01(\t"\xfa\x01\n\x12CaptureDeviceSpecs\x12\x15\n\rrecap_version\x18\x06 \x01(\x05\x12\x12\n\ngit_commit\x18\x07 \x01(\t\x12\n\n\x02os\x18\x01 \x01(\t\x12\x17\n\x0fkeyboard_layout\x18\x02 \x01(\t\x12I\n\x0cwindow_specs\x18\x03 \x01(\x0b23.elefant.training_data.video_annotation.WindowSpecs\x12I\n\x0cscreen_specs\x18\x04 \x03(\x0b23.elefant.training_data.video_annotation.ScreenSpecs"T\n\x0bScreenSpecs\x12\r\n\x05width\x18\x01 \x01(\x05\x12\x0e\n\x06height\x18\x02 \x01(\x05\x12\r\n\x05scale\x18\x03 \x01(\x01\x12\x0b\n\x03dpi\x18\x04 \x01(\x05\x12\n\n\x02id\x18\x05 \x01(\x03"j\n\x0bWindowSpecs\x12\r\n\x05width\x18\x01 \x01(\x05\x12\x0e\n\x06height\x18\x02 \x01(\x05\x12\r\n\x05scale\x18\x03 \x01(\x01\x12\x0b\n\x03dpi\x18\x04 \x01(\x05\x12\r\n\x05title\x18\x05 \x01(\t\x12\x11\n\tscreen_id\x18\x06 \x01(\x03"\xee\x03\n\x17VideoAnnotationMetadata\x12\n\n\x02id\x18\x01 \x01(\t\x12\x11\n\ttimestamp\x18\x02 \x01(\x03\x12\x0c\n\x04user\x18\x03 \x01(\t\x12\r\n\x05tasks\x18\x04 \x03(\t\x12\x19\n\x11frames_per_second\x18\x05 \x01(\x02\x12G\n\x03env\x18\x06 \x01(\x0b2:.elefant.training_data.video_annotation.VideoAnnotationEnv\x12\x12\n\ngroup_name\x18\x07 \x01(\t\x12\x1b\n\x13golden_example_path\x18\x08 \x01(\t\x12R\n\x11video_source_info\x18\t \x01(\x0b27.elefant.training_data.video_annotation.VideoSourceInfo\x12X\n\x14capture_device_specs\x18\n \x01(\x0b2:.elefant.training_data.video_annotation.CaptureDeviceSpecs\x12T\n\x12video_quality_info\x18\x0b \x01(\x0b28.elefant.training_data.video_annotation.VideoQualityInfo"w\n\x10VideoQualityInfo\x12\x13\n\x0bhas_facecam\x18\x01 \x01(\x08\x12N\n\x0caspect_ratio\x18\x02 \x01(\x0b28.elefant.training_data.video_annotation.VideoAspectRatio":\n\x10VideoAspectRatio\x12\x11\n\tnumerator\x18\x01 \x01(\x05\x12\x13\n\x0bdenominator\x18\x02 \x01(\x05"l\n\x0fVideoSourceInfo\x12>\n\x06source\x18\x01 \x01(\x0b2..elefant.training_data.video_annotation.Source\x12\x19\n\x11video_filter_info\x18\x02 \x01(\t"\x80\x02\n\x06Source\x12O\n\x0eyoutube_source\x18\x01 \x01(\x0b25.elefant.training_data.video_annotation.YoutubeSourceH\x00\x12G\n\noai_source\x18\x02 \x01(\x0b21.elefant.training_data.video_annotation.OAISourceH\x00\x12M\n\rtwitch_source\x18\x03 \x01(\x0b24.elefant.training_data.video_annotation.TwitchSourceH\x00B\r\n\x0bsource_type"0\n\rYoutubeSource\x12\x10\n\x08video_id\x18\x01 \x01(\t\x12\r\n\x05query\x18\x02 \x01(\t"C\n\x0cTwitchSource\x12\x10\n\x08video_id\x18\x01 \x01(\t\x12\x0c\n\x04game\x18\x02 \x01(\t\x12\x13\n\x0benv_subtype\x18\x03 \x01(\t"2\n\tOAISource\x12\x11\n\tvideo_url\x18\x01 \x01(\t\x12\x12\n\naction_url\x18\x02 \x01(\t"\xa5\x02\n\x0bMouseAction\x12@\n\x11mouse_absolute_px\x18\x01 \x01(\x0b2%.elefant.training_data.shared.Vec2Int\x12?\n\x0emouse_relative\x18\x02 \x01(\x0b2\'.elefant.training_data.shared.Vec2Float\x12=\n\x0emouse_delta_px\x18\x03 \x01(\x0b2%.elefant.training_data.shared.Vec2Int\x12>\n\x0fscroll_delta_px\x18\x04 \x01(\x0b2%.elefant.training_data.shared.Vec2Int\x12\x14\n\x0cbuttons_down\x18\t \x03(\t"k\n\x15DeprecatedMouseAction\x12\t\n\x01x\x18\x01 \x01(\x01\x12\t\n\x01y\x18\x02 \x01(\x01\x12\n\n\x02dx\x18\x03 \x01(\x01\x12\n\n\x02dy\x18\x04 \x01(\x01\x12\x0f\n\x07buttons\x18\x05 \x03(\x05\x12\x13\n\x0bnew_buttons\x18\x06 \x03(\x05"\x1e\n\x0eKeyboardAction\x12\x0c\n\x04keys\x18\x01 \x03(\t"\xdf\x01\n\x0eGamePadButtons\x12\r\n\x05south\x18\x01 \x01(\x08\x12\r\n\x05north\x18\x02 \x01(\x08\x12\x0c\n\x04east\x18\x03 \x01(\x08\x12\x0c\n\x04west\x18\x04 \x01(\x08\x12\x0f\n\x07dpad_up\x18\x05 \x01(\x08\x12\x11\n\tdpad_down\x18\x06 \x01(\x08\x12\x11\n\tdpad_left\x18\x07 \x01(\x08\x12\x12\n\ndpad_right\x18\x08 \x01(\x08\x12\r\n\x05start\x18\t \x01(\x08\x12\x0e\n\x06select\x18\n \x01(\x08\x12\x13\n\x0bleft_bumper\x18\x0b \x01(\x08\x12\x14\n\x0cright_bumper\x18\x0c \x01(\x08".\n\x05Stick\x12\t\n\x01x\x18\x01 \x01(\x02\x12\t\n\x01y\x18\x02 \x01(\x02\x12\x0f\n\x07pressed\x18\x03 \x01(\x08"\x8c\x02\n\rGamePadAction\x12G\n\x07buttons\x18\x01 \x01(\x0b26.elefant.training_data.video_annotation.GamePadButtons\x12A\n\nleft_stick\x18\x02 \x01(\x0b2-.elefant.training_data.video_annotation.Stick\x12B\n\x0bright_stick\x18\x03 \x01(\x0b2-.elefant.training_data.video_annotation.Stick\x12\x14\n\x0cleft_trigger\x18\x04 \x01(\x02\x12\x15\n\rright_trigger\x18\x05 \x01(\x02"\xd2\x02\n\x0eLowLevelAction\x12W\n\x10mouse_deprecated\x18\x01 \x01(\x0b2=.elefant.training_data.video_annotation.DeprecatedMouseAction\x12H\n\x08keyboard\x18\x02 \x01(\x0b26.elefant.training_data.video_annotation.KeyboardAction\x12\x10\n\x08is_known\x18\x03 \x01(\x08\x12B\n\x05mouse\x18\x04 \x01(\x0b23.elefant.training_data.video_annotation.MouseAction\x12G\n\x08game_pad\x18\x05 \x01(\x0b25.elefant.training_data.video_annotation.GamePadAction"y\n\x0eMinecraftState\x12\x0c\n\x04xpos\x18\x01 \x01(\x01\x12\x0c\n\x04ypos\x18\x02 \x01(\x01\x12\x0c\n\x04zpos\x18\x03 \x01(\x01\x12\x0b\n\x03yaw\x18\x04 \x01(\x01\x12\r\n\x05pitch\x18\x05 \x01(\x01\x12\r\n\x05milli\x18\x06 \x01(\x03\x12\x12\n\nserverTick\x18\x07 \x01(\x03"g\n\x13TwoDBlockChaseState\x12\x13\n\x0bagent_pos_x\x18\x01 \x01(\x05\x12\x13\n\x0bagent_pos_y\x18\x02 \x01(\x05\x12\x12\n\ngoal_pos_x\x18\x03 \x01(\x05\x12\x12\n\ngoal_pos_y\x18\x04 \x01(\x05"\xb5\x01\n\x13MiniWorldAgentState\x12\x13\n\x0bagent_pos_x\x18\x01 \x01(\x01\x12\x13\n\x0bagent_pos_z\x18\x02 \x01(\x01\x12\x13\n\x0bagent_dir_x\x18\x05 \x01(\x01\x12\x13\n\x0bagent_dir_z\x18\x06 \x01(\x01\x12\x14\n\x0ccamera_dir_x\x18\x07 \x01(\x01\x12\x14\n\x0ccamera_dir_z\x18\x08 \x01(\x01\x12\x0e\n\x06health\x18\t \x01(\x03\x12\x0e\n\x06reward\x18\n \x01(\x01";\n\x11MiniWorldEnvState\x12\x12\n\ngoal_pos_x\x18\x01 \x01(\x01\x12\x12\n\ngoal_pos_z\x18\x02 \x01(\x01"\xc4\x01\n\x0eMiniWorldState\x12Z\n\x15miniworld_agent_state\x18\x01 \x01(\x0b2;.elefant.training_data.video_annotation.MiniWorldAgentState\x12V\n\x13miniworld_env_state\x18\x02 \x03(\x0b29.elefant.training_data.video_annotation.MiniWorldEnvState"3\n\x10MouseButtonEvent\x12\x0e\n\x06button\x18\x01 \x01(\t\x12\x0f\n\x07pressed\x18\x02 \x01(\x08"-\n\rKeyboardEvent\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x0f\n\x07pressed\x18\x02 \x01(\x08"5\n\x12GamePadButtonEvent\x12\x0e\n\x06button\x18\x01 \x01(\t\x12\x0f\n\x07pressed\x18\x02 \x01(\x08"/\n\x10GamePadAxisEvent\x12\x0c\n\x04axis\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x02"5\n\x13GamePadTriggerEvent\x12\x0f\n\x07trigger\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x02"\xb2\x05\n\nInputEvent\x12O\n\x0bmouse_event\x18\x01 \x01(\x0b28.elefant.training_data.video_annotation.MouseButtonEventH\x00\x12O\n\x0ekeyboard_event\x18\x02 \x01(\x0b25.elefant.training_data.video_annotation.KeyboardEventH\x00\x12A\n\x10mouse_move_event\x18\x03 \x01(\x0b2%.elefant.training_data.shared.Vec2IntH\x00\x12<\n\x0bwheel_event\x18\x04 \x01(\x0b2%.elefant.training_data.shared.Vec2IntH\x00\x12B\n\x11mouse_delta_event\x18\x05 \x01(\x0b2%.elefant.training_data.shared.Vec2IntH\x00\x12[\n\x15game_pad_button_event\x18\x07 \x01(\x0b2:.elefant.training_data.video_annotation.GamePadButtonEventH\x00\x12W\n\x13game_pad_axis_event\x18\x08 \x01(\x0b28.elefant.training_data.video_annotation.GamePadAxisEventH\x00\x12]\n\x16game_pad_trigger_event\x18\t \x01(\x0b2;.elefant.training_data.video_annotation.GamePadTriggerEventH\x00\x12\x0c\n\x04time\x18\x06 \x01(\x04\x12\x11\n\tsimulated\x18\n \x01(\x08B\x07\n\x05event"\xbd\x05\n\x0fFrameAnnotation\x12K\n\x0buser_action\x18\x01 \x01(\x0b26.elefant.training_data.video_annotation.LowLevelAction\x12M\n\rsystem_action\x18\x06 \x01(\x0b26.elefant.training_data.video_annotation.LowLevelAction\x12Q\n\x0fminecraft_state\x18\x02 \x01(\x0b26.elefant.training_data.video_annotation.MinecraftStateH\x00\x12X\n\x11block_chase_state\x18\x03 \x01(\x0b2;.elefant.training_data.video_annotation.TwoDBlockChaseStateH\x00\x12Q\n\x0fminiworld_state\x18\x04 \x01(\x0b26.elefant.training_data.video_annotation.MiniWorldStateH\x00\x12G\n\x0baction_task\x18\x05 \x01(\x0b22.elefant.training_data.video_annotation.ActionTask\x12Z\n\x15frame_text_annotation\x18\x07 \x03(\x0b2;.elefant.training_data.video_annotation.FrameTextAnnotation\x12\x12\n\nframe_time\x18\x08 \x01(\x04\x12H\n\x0cinput_events\x18\t \x03(\x0b22.elefant.training_data.video_annotation.InputEventB\x0b\n\tenv_state"\x80\x03\n\x13FrameTextAnnotation\x12\x13\n\x0binstruction\x18\x01 \x01(\t\x12X\n\x14frame_text_annotator\x18\x02 \x01(\x0b2:.elefant.training_data.video_annotation.FrameTextAnnotator\x12\x10\n\x08duration\x18\x03 \x01(\x02\x12o\n\x13text_embedding_dict\x18\x04 \x03(\x0b2R.elefant.training_data.video_annotation.FrameTextAnnotation.TextEmbeddingDictEntry\x1aw\n\x16TextEmbeddingDictEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12L\n\x05value\x18\x02 \x01(\x0b2=.elefant.training_data.video_annotation.TokenizerEmbeddingMap:\x028\x01"\xef\x01\n\x15TokenizerEmbeddingMap\x12j\n\x0ftext_embeddings\x18\x01 \x03(\x0b2Q.elefant.training_data.video_annotation.TokenizerEmbeddingMap.TextEmbeddingsEntry\x1aj\n\x13TextEmbeddingsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12B\n\x05value\x18\x02 \x01(\x0b23.elefant.training_data.video_annotation.FloatTensor:\x028\x01",\n\x0bFloatTensor\x12\r\n\x05shape\x18\x01 \x03(\x03\x12\x0e\n\x06values\x18\x02 \x03(\x02"7\n\x12FrameTextAnnotator\x12\x10\n\x08provider\x18\x01 \x01(\t\x12\x0f\n\x07version\x18\x02 \x01(\t"A\n\nActionTask\x12\x18\n\x10lower_level_task\x18\x01 \x01(\t\x12\x19\n\x11higher_level_task\x18\x02 \x01(\t"*\n\x07Point3D\x12\t\n\x01x\x18\x01 \x01(\x01\x12\t\n\x01y\x18\x02 \x01(\x01\x12\t\n\x01z\x18\x03 \x01(\x01"M\n\tChangelog\x12\x11\n\tchange_id\x18\x01 \x01(\t\x12\x1a\n\x12change_description\x18\x02 \x01(\t\x12\x11\n\ttimestamp\x18\x03 \x01(\x03",\n\nVoiceEvent\x12\x0c\n\x04time\x18\x01 \x01(\x04\x12\x10\n\x08speaking\x18\x02 \x01(\x08"\xf2\x04\n\x0fVideoAnnotation\x12Q\n\x08metadata\x18\x02 \x01(\x0b2?.elefant.training_data.video_annotation.VideoAnnotationMetadata\x12R\n\x11frame_annotations\x18\x03 \x03(\x0b27.elefant.training_data.video_annotation.FrameAnnotation\x12T\n\x12task_specific_info\x18\x04 \x01(\x0b28.elefant.training_data.video_annotation.TaskSpecificInfo\x12R\n\x11video_global_task\x18\x05 \x01(\x0b27.elefant.training_data.video_annotation.VideoGlobalTask\x12m\n\x1fvideo_frame_annotation_metadata\x18\x07 \x01(\x0b2D.elefant.training_data.video_annotation.VideoFrameAnnotationMetadata\x12\x0f\n\x07version\x18\x06 \x01(\x05\x12D\n\tchangelog\x18\x08 \x03(\x0b21.elefant.training_data.video_annotation.Changelog\x12H\n\x0cvoice_events\x18\t \x03(\x0b22.elefant.training_data.video_annotation.VoiceEvent"*\n\x0fVideoGlobalTask\x12\x17\n\x0fvideo_narrative\x18\x01 \x01(\t"7\n\x1cVideoFrameAnnotationMetadata\x12\x17\n\x0fvideo_narrative\x18\x01 \x01(\t"\x80\x01\n\x10TaskSpecificInfo\x12_\n\x10pathfinding_info\x18\x01 \x01(\x0b2C.elefant.training_data.video_annotation.PathfindingTaskSpecificInfoH\x00B\x0b\n\ttask_info"n\n\x1bPathfindingTaskSpecificInfo\x12O\n\x16pathfinding_trajectory\x18\x01 \x03(\x0b2/.elefant.training_data.video_annotation.Point3Db\x06proto3'
)
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, "video_annotation_pb2", globals())
if _descriptor._USE_C_DESCRIPTORS == False:
    DESCRIPTOR._options = None
    _FRAMETEXTANNOTATION_TEXTEMBEDDINGDICTENTRY._options = None
    _FRAMETEXTANNOTATION_TEXTEMBEDDINGDICTENTRY._serialized_options = b"8\x01"
    _TOKENIZEREMBEDDINGMAP_TEXTEMBEDDINGSENTRY._options = None
    _TOKENIZEREMBEDDINGMAP_TEXTEMBEDDINGSENTRY._serialized_options = b"8\x01"
    _VIDEOANNOTATIONENV._serialized_start = 80
    _VIDEOANNOTATIONENV._serialized_end = 155
    _CAPTUREDEVICESPECS._serialized_start = 158
    _CAPTUREDEVICESPECS._serialized_end = 408
    _SCREENSPECS._serialized_start = 410
    _SCREENSPECS._serialized_end = 494
    _WINDOWSPECS._serialized_start = 496
    _WINDOWSPECS._serialized_end = 602
    _VIDEOANNOTATIONMETADATA._serialized_start = 605
    _VIDEOANNOTATIONMETADATA._serialized_end = 1099
    _VIDEOQUALITYINFO._serialized_start = 1101
    _VIDEOQUALITYINFO._serialized_end = 1220
    _VIDEOASPECTRATIO._serialized_start = 1222
    _VIDEOASPECTRATIO._serialized_end = 1280
    _VIDEOSOURCEINFO._serialized_start = 1282
    _VIDEOSOURCEINFO._serialized_end = 1390
    _SOURCE._serialized_start = 1393
    _SOURCE._serialized_end = 1649
    _YOUTUBESOURCE._serialized_start = 1651
    _YOUTUBESOURCE._serialized_end = 1699
    _TWITCHSOURCE._serialized_start = 1701
    _TWITCHSOURCE._serialized_end = 1768
    _OAISOURCE._serialized_start = 1770
    _OAISOURCE._serialized_end = 1820
    _MOUSEACTION._serialized_start = 1823
    _MOUSEACTION._serialized_end = 2116
    _DEPRECATEDMOUSEACTION._serialized_start = 2118
    _DEPRECATEDMOUSEACTION._serialized_end = 2225
    _KEYBOARDACTION._serialized_start = 2227
    _KEYBOARDACTION._serialized_end = 2257
    _GAMEPADBUTTONS._serialized_start = 2260
    _GAMEPADBUTTONS._serialized_end = 2483
    _STICK._serialized_start = 2485
    _STICK._serialized_end = 2531
    _GAMEPADACTION._serialized_start = 2534
    _GAMEPADACTION._serialized_end = 2802
    _LOWLEVELACTION._serialized_start = 2805
    _LOWLEVELACTION._serialized_end = 3143
    _MINECRAFTSTATE._serialized_start = 3145
    _MINECRAFTSTATE._serialized_end = 3266
    _TWODBLOCKCHASESTATE._serialized_start = 3268
    _TWODBLOCKCHASESTATE._serialized_end = 3371
    _MINIWORLDAGENTSTATE._serialized_start = 3374
    _MINIWORLDAGENTSTATE._serialized_end = 3555
    _MINIWORLDENVSTATE._serialized_start = 3557
    _MINIWORLDENVSTATE._serialized_end = 3616
    _MINIWORLDSTATE._serialized_start = 3619
    _MINIWORLDSTATE._serialized_end = 3815
    _MOUSEBUTTONEVENT._serialized_start = 3817
    _MOUSEBUTTONEVENT._serialized_end = 3868
    _KEYBOARDEVENT._serialized_start = 3870
    _KEYBOARDEVENT._serialized_end = 3915
    _GAMEPADBUTTONEVENT._serialized_start = 3917
    _GAMEPADBUTTONEVENT._serialized_end = 3970
    _GAMEPADAXISEVENT._serialized_start = 3972
    _GAMEPADAXISEVENT._serialized_end = 4019
    _GAMEPADTRIGGEREVENT._serialized_start = 4021
    _GAMEPADTRIGGEREVENT._serialized_end = 4074
    _INPUTEVENT._serialized_start = 4077
    _INPUTEVENT._serialized_end = 4767
    _FRAMEANNOTATION._serialized_start = 4770
    _FRAMEANNOTATION._serialized_end = 5471
    _FRAMETEXTANNOTATION._serialized_start = 5474
    _FRAMETEXTANNOTATION._serialized_end = 5858
    _FRAMETEXTANNOTATION_TEXTEMBEDDINGDICTENTRY._serialized_start = 5739
    _FRAMETEXTANNOTATION_TEXTEMBEDDINGDICTENTRY._serialized_end = 5858
    _TOKENIZEREMBEDDINGMAP._serialized_start = 5861
    _TOKENIZEREMBEDDINGMAP._serialized_end = 6100
    _TOKENIZEREMBEDDINGMAP_TEXTEMBEDDINGSENTRY._serialized_start = 5994
    _TOKENIZEREMBEDDINGMAP_TEXTEMBEDDINGSENTRY._serialized_end = 6100
    _FLOATTENSOR._serialized_start = 6102
    _FLOATTENSOR._serialized_end = 6146
    _FRAMETEXTANNOTATOR._serialized_start = 6148
    _FRAMETEXTANNOTATOR._serialized_end = 6203
    _ACTIONTASK._serialized_start = 6205
    _ACTIONTASK._serialized_end = 6270
    _POINT3D._serialized_start = 6272
    _POINT3D._serialized_end = 6314
    _CHANGELOG._serialized_start = 6316
    _CHANGELOG._serialized_end = 6393
    _VOICEEVENT._serialized_start = 6395
    _VOICEEVENT._serialized_end = 6439
    _VIDEOANNOTATION._serialized_start = 6442
    _VIDEOANNOTATION._serialized_end = 7068
    _VIDEOGLOBALTASK._serialized_start = 7070
    _VIDEOGLOBALTASK._serialized_end = 7112
    _VIDEOFRAMEANNOTATIONMETADATA._serialized_start = 7114
    _VIDEOFRAMEANNOTATIONMETADATA._serialized_end = 7169
    _TASKSPECIFICINFO._serialized_start = 7172
    _TASKSPECIFICINFO._serialized_end = 7300
    _PATHFINDINGTASKSPECIFICINFO._serialized_start = 7302
    _PATHFINDINGTASKSPECIFICINFO._serialized_end = 7412
