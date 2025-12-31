"""Client and server classes corresponding to protobuf-defined services."""

import grpc
import warnings
from . import video_inference_pb2 as video__inference__pb2

GRPC_GENERATED_VERSION = "1.71.2"
GRPC_VERSION = grpc.__version__
_version_not_supported = False
try:
    from grpc._utilities import first_version_is_lower

    _version_not_supported = first_version_is_lower(
        GRPC_VERSION, GRPC_GENERATED_VERSION
    )
except ImportError:
    _version_not_supported = True
if _version_not_supported:
    raise RuntimeError(
        f"The grpc package installed is at version {GRPC_VERSION},"
        + f" but the generated code in video_inference_pb2_grpc.py depends on"
        + f" grpcio>={GRPC_GENERATED_VERSION}."
        + f" Please upgrade your grpc module to grpcio>={GRPC_GENERATED_VERSION}"
        + f" or downgrade your generated code using grpcio-tools<={GRPC_VERSION}."
    )


class VideoInferenceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.InferVideo = channel.stream_stream(
            "/elefant.training_data.video_inference.VideoInference/InferVideo",
            request_serializer=video__inference__pb2.Frame.SerializeToString,
            response_deserializer=video__inference__pb2.Action.FromString,
            _registered_method=True,
        )


class VideoInferenceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def InferVideo(self, request_iterator, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")


def add_VideoInferenceServicer_to_server(servicer, server):
    rpc_method_handlers = {
        "InferVideo": grpc.stream_stream_rpc_method_handler(
            servicer.InferVideo,
            request_deserializer=video__inference__pb2.Frame.FromString,
            response_serializer=video__inference__pb2.Action.SerializeToString,
        )
    }
    generic_handler = grpc.method_handlers_generic_handler(
        "elefant.training_data.video_inference.VideoInference", rpc_method_handlers
    )
    server.add_generic_rpc_handlers((generic_handler,))
    server.add_registered_method_handlers(
        "elefant.training_data.video_inference.VideoInference", rpc_method_handlers
    )


class VideoInference(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def InferVideo(
        request_iterator,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.stream_stream(
            request_iterator,
            target,
            "/elefant.training_data.video_inference.VideoInference/InferVideo",
            video__inference__pb2.Frame.SerializeToString,
            video__inference__pb2.Action.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True,
        )
