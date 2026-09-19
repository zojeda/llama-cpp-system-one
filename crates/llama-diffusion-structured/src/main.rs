use clap::Parser;
use llama_diffusion_structured::{Engine, ModelConfig, ReadRequest};
use std::path::PathBuf;

#[derive(Parser)]
#[command(about = "Read one restricted DiffusionGemma classification slot")]
struct Args {
    #[arg(short, long, env = "DIFFUSION_MODEL")]
    model: PathBuf,
    #[arg(short, long)]
    prompt: String,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long, default_value_t = -1, allow_hyphen_values = true)]
    gpu_layers: i32,
    #[arg(long, default_value_t = 0)]
    main_gpu: i32,
    #[arg(long, default_value_t = 4096)]
    context_size: u32,
    #[arg(long, default_value_t = 512)]
    batch_size: u32,
    #[arg(long)]
    flash_attention: bool,
    #[arg(long)]
    json: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let mut config = ModelConfig::new(args.model);
    config.gpu_layers = args.gpu_layers;
    config.main_gpu = args.main_gpu;
    config.context_size = args.context_size;
    config.batch_size = args.batch_size;
    config.flash_attention = args.flash_attention;
    let mut engine = Engine::load(&config)?;
    let result = engine.read(&ReadRequest::scm(&args.prompt), args.seed)?;
    if args.json {
        println!("{}", serde_json::to_string_pretty(&result)?);
    } else {
        let slot = &result.slots[0];
        println!("Structured diffusion read\n\nSlot: is_scm\n");
        for (i, label) in ["A (yes)", "B (no)"].iter().enumerate() {
            println!(
                "{label}\n  token: {}\n  logit: {:.6}\n  probability: {:.6}\n",
                slot.candidate_tokens[i], slot.logits[i], slot.probabilities[i]
            );
        }
        println!(
            "prompt tokens: {}\ncanvas tokens: {}",
            result.prompt_tokens, result.canvas_tokens
        );
        println!(
            "slot canvas position: {}\nslot absolute position: {}",
            slot.canvas_position, slot.absolute_position
        );
        println!(
            "seed: {}\nslot initial token: {}\nforward time: {:.2} ms",
            result.seed, slot.initial_token, result.forward_ms
        );
    }
    Ok(())
}
