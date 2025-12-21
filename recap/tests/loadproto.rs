use video_annotation_proto::video_annotation::VideoAnnotation;
use video_inference_grpc::prost::Message;

const PROTO_PATHS: &[&str] = &[
    "tests/assets/system_and_user.proto",
    "tests/assets/system_only.proto",
    "tests/assets/user_only.proto",
];

#[test]
/// check that all the protos can be read
fn load_proto() {
    for path in PROTO_PATHS {
        let full_path = std::path::PathBuf::from(path);
        let bytes = std::fs::read(full_path).expect("Failed to read proto file");
        let proto = VideoAnnotation::decode(bytes.as_ref()).expect("Failed to decode proto");
        insta::assert_ron_snapshot!(proto);
    }
}
