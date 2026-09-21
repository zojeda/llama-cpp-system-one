//! JSON-lines research driver. Uses the production compiler and response mapping.
use clap::Parser;
use llama_diffusion_structured::{Engine, ModelConfig, ReadLayout, ReadRequest};
use serde_json::{Value, json};
use std::{
    io::{self, BufRead, Write},
    path::PathBuf,
    time::Instant,
};
use system_one::Request;

#[derive(Parser)]
struct Args {
    #[arg(short, long, env = "DIFFUSION_MODEL")]
    model: PathBuf,
    #[arg(long, default_value_t = 8192)]
    context_size: u32,
    #[arg(long)]
    flash_attention: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let mut config = ModelConfig::new(args.model);
    config.context_size = args.context_size;
    config.flash_attention = args.flash_attention;
    let mut engine = Engine::load(&config)?;
    let mut stdout = io::stdout().lock();
    for line in io::stdin().lock().lines() {
        let input: Value = serde_json::from_str(&line?)?;
        let start = Instant::now();
        let result = run(&mut engine, &input);
        let output = match result {
            Ok((response, read)) => json!({"id":input["id"], "response":response, "read":read,
                "latency_s":start.elapsed().as_secs_f64()}),
            Err(error) => json!({"id":input["id"], "error":error.to_string(),
                "latency_s":start.elapsed().as_secs_f64()}),
        };
        serde_json::to_writer(&mut stdout, &output)?;
        writeln!(stdout)?;
        stdout.flush()?;
    }
    Ok(())
}

fn run(engine: &mut Engine, input: &Value) -> Result<(Value, Value), Box<dyn std::error::Error>> {
    let request = Request::parse(input["request"].clone())?;
    let compiled: ReadRequest = if input.get("compiled").is_some() {
        serde_json::from_value(input["compiled"].clone())?
    } else {
        request.compile(engine.codes())?
    };
    let layout: ReadLayout =
        serde_json::from_value(input.get("layout").cloned().unwrap_or(json!({})))?;
    let seed = input.get("seed").and_then(Value::as_u64).unwrap_or(42);
    let read = engine.read_with_layout(
        &compiled,
        seed,
        request.options(),
        request.images(),
        &layout,
    )?;
    Ok((
        serde_json::to_value(request.response("research", &read)?)?,
        serde_json::to_value(read)?,
    ))
}
