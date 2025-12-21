use std::{path::PathBuf, str::FromStr};

use clap::Parser;
use rand::seq::IndexedRandom;
use video_annotation_proto::video_annotation::VideoAnnotation;
use video_inference_grpc::prost::Message;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Type {
    Json,
    Proto,
    Ron,
}

impl FromStr for Type {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "json" => Ok(Type::Json),
            "proto" => Ok(Type::Proto),
            "ron" => Ok(Type::Ron),
            _ => Err("Unknown type, must be one of: json, proto, ron"),
        }
    }
}

#[derive(Debug, Parser)]
struct Args {
    /// Input file path
    ///
    /// The path to the input file to convert. if input_type is not specified, it will be inferred from the file extension.
    #[clap(long, short)]
    input: PathBuf,
    /// Output file path
    ///
    /// The path to the output file to create. if output_type is not specified, it will be inferred from the file extension.
    #[clap(long, short)]
    output: PathBuf,

    /// Input Type
    /// Options are
    /// - json
    /// - proto
    /// - ron
    #[clap(long)]
    input_type: Option<Type>,
    /// Output Type
    /// Options are
    /// - json
    /// - proto
    /// - ron
    #[clap(long)]
    output_type: Option<Type>,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let input_type = args.input_type.unwrap_or_else(|| {
        let ext = args.input.extension().unwrap_or_default();
        Type::from_str(ext.to_str().unwrap())
            .unwrap_or_else(|_| panic!("Unknown input type: {}", ext.to_str().unwrap()))
    });

    let output_type = args.output_type.unwrap_or_else(|| {
        let ext = args.output.extension().unwrap_or_default();
        Type::from_str(ext.to_str().unwrap())
            .unwrap_or_else(|_| panic!("Unknown output type: {}", ext.to_str().unwrap()))
    });

    let data = std::fs::read(&args.input)?;

    #[allow(unused_mut)]
    let mut input: VideoAnnotation = match input_type {
        Type::Json => serde_json::de::from_slice(&data)?,
        Type::Proto => VideoAnnotation::decode(data.as_slice())?,
        Type::Ron => ron::de::from_bytes(&data)?,
    };

    // input
    //     .frame_annotations
    //     .iter_mut()
    //     .enumerate()
    //     .for_each(|(i, fa)| {
    //         fa.frame_time = i as u64;

    //         if let Some(user_action) = &fa.user_action
    //             && let Some(system_action) = &fa.system_action
    //         {
    //             let inference_running = system_action.is_known;
    //             if inference_running {
    //                 if let Some(keys) = &user_action.keyboard
    //                     && !keys.keys.is_empty()
    //                 {
    //                     eprintln!("Frame {}: User pressed keys: {:?}", i, keys.keys);
    //                 }
    //             } else {
    //                 if let Some(keys) = &system_action.keyboard
    //                     && !keys.keys.is_empty()
    //                 {
    //                     eprintln!("Frame {}: System pressed keys: {:?}", i, keys.keys);
    //                 }
    //             }
    //         }
    //     });

    let output_data = match output_type {
        Type::Json => serde_json::to_string_pretty(&input)?.into_bytes(),
        Type::Proto => input.encode_to_vec(),
        Type::Ron => ron::to_string(&input)?.into_bytes(),
    };

    std::fs::write(args.output, output_data)?;

    Ok(())
}
