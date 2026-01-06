"""Generated protocol buffer code."""

from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder

_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC, 5, 29, 0, "", "video_inference.proto"
)
_sym_db = _symbol_database.Default()
from . import shared_pb2 as shared__pb2

DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(
    b'\n\x15video_inference.proto\x12%elefant.training_data.video_inference\x1a\x0cshared.proto"@\n\x05Frame\x12\r\n\x05width\x18\x01 \x01(\x05\x12\x0e\n\x06height\x18\x02 \x01(\x05\x12\x0c\n\x04data\x18\x03 \x01(\x0c\x12\n\n\x02id\x18\x04 \x01(\x05"\xf2\x01\n\x0bMouseAction\x12?\n\x0emouse_delta_px\x18\x01 \x01(\x0b2%.elefant.training_data.shared.Vec2IntH\x00\x12<\n\tmouse_pos\x18\x02 \x01(\x0b2\'.elefant.training_data.shared.Vec2FloatH\x00\x12>\n\x0fscroll_delta_px\x18\x03 \x01(\x0b2%.elefant.training_data.shared.Vec2Int\x12\x14\n\x0cbuttons_down\x18\x04 \x03(\tB\x0e\n\x0cmouse_change"l\n\x06Action\x12\x0c\n\x04keys\x18\x01 \x03(\t\x12\n\n\x02id\x18\x02 \x01(\x05\x12H\n\x0cmouse_action\x18\x03 \x01(\x0b22.elefant.training_data.video_inference.MouseAction2\x81\x01\n\x0eVideoInference\x12o\n\nInferVideo\x12,.elefant.training_data.video_inference.Frame\x1a-.elefant.training_data.video_inference.Action"\x00(\x010\x01b\x06proto3'
)
_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, "video_inference_pb2", _globals)
if not _descriptor._USE_C_DESCRIPTORS:
    DESCRIPTOR._loaded_options = None
    _globals["_FRAME"]._serialized_start = 78
    _globals["_FRAME"]._serialized_end = 142
    _globals["_MOUSEACTION"]._serialized_start = 145
    _globals["_MOUSEACTION"]._serialized_end = 387
    _globals["_ACTION"]._serialized_start = 389
    _globals["_ACTION"]._serialized_end = 497
    _globals["_VIDEOINFERENCE"]._serialized_start = 500
    _globals["_VIDEOINFERENCE"]._serialized_end = 629
