use anyhow::Result;
use std::io::{self, Write};
use std::path::PathBuf;

use catetin_core::config::AppConfig;
use catetin_core::i18n::UiLanguage;
use catetin_core::logging::init_logger;
use catetin_core::pipeline::DiarizationMode;

fn read_line(prompt: &str) -> String {
    print!("{}", prompt);
    let _ = io::stdout().flush();
    let mut line = String::new();
    let _ = io::stdin().read_line(&mut line);
    line.trim().to_string()
}

/// Interactive terminal configuration wizard.
/// Guides the user through language, diarization mode, Whisper model, and logging,
/// automatically saving choices to the configuration file without manual editing.
fn run_interactive_wizard(config: &mut AppConfig) -> Result<()> {
    let mut lang = config.ui_language.clone();

    println!("============================================================");
    println!("  Catetin - {}", lang.text("wizard.title"));
    println!("============================================================");

    // Step 1: Language
    println!("{}", lang.text("wizard.step_lang"));
    println!("  [1] Bahasa Indonesia (id) [Bawaan/Default]");
    println!("  [2] English (en)");
    let lang_input = read_line(&format!("  {} [1]: ", lang.text("wizard.prompt_choice")));
    match lang_input.as_str() {
        "2" | "en" | "english" => {
            lang = UiLanguage::English;
        }
        "1" | "id" | "indonesian" | "" => {
            lang = UiLanguage::Indonesian;
        }
        custom => {
            lang = UiLanguage::from_code(custom);
        }
    }
    config.ui_language = lang.clone();
    println!(
        "  -> {}: {} ({})\n",
        lang.text("label.language"),
        lang.display_name(),
        lang.code()
    );

    // Step 2: Diarization Mode
    println!("{}", lang.text("wizard.step_mode"));
    println!("  [1] {} [Bawaan/Default]", lang.text("label.mode_online"));
    println!("  [2] {}", lang.text("label.mode_in_person"));
    println!("  [3] {}", lang.text("label.mode_hybrid"));
    let mode_input = read_line(&format!("  {} [1]: ", lang.text("wizard.prompt_choice")));
    match mode_input.as_str() {
        "2" | "in-person" | "roundtable" => {
            config.diarization_mode = DiarizationMode::InPerson;
        }
        "3" | "hybrid" => {
            config.diarization_mode = DiarizationMode::Hybrid;
        }
        _ => {
            config.diarization_mode = DiarizationMode::OnlineMeeting;
        }
    }
    let mode_desc = match config.diarization_mode {
        DiarizationMode::OnlineMeeting => lang.text("label.mode_online"),
        DiarizationMode::InPerson => lang.text("label.mode_in_person"),
        DiarizationMode::Hybrid => lang.text("label.mode_hybrid"),
    };
    println!("  -> {}: {}\n", lang.text("label.mode"), mode_desc);

    // Step 3: Whisper Model Tier
    println!("{}", lang.text("wizard.step_model"));
    println!("  [1] tiny   (~75 MB  - Sangat cepat / Ultra fast)");
    println!("  [2] base   (~140 MB - Seimbang / Balanced) [Bawaan/Default]");
    println!("  [3] small  (~460 MB - Akurasi tinggi / High accuracy)");
    println!("  [4] medium (~1.5 GB - Akurasi maksimal / Best accuracy)");
    let model_input = read_line(&format!("  {} [2]: ", lang.text("wizard.prompt_choice")));
    match model_input.as_str() {
        "1" | "tiny" => config.model_tier = "tiny".to_string(),
        "3" | "small" => config.model_tier = "small".to_string(),
        "4" | "medium" => config.model_tier = "medium".to_string(),
        _ => config.model_tier = "base".to_string(),
    }
    println!(
        "  -> {}: {} ({})\n",
        lang.text("label.model_tier"),
        config.model_tier,
        config.model_filename()
    );

    // Step 4: Verbose logging
    println!("{}", lang.text("wizard.step_verbose"));
    let verbose_input = read_line(&format!("  {} [N]: ", lang.text("wizard.prompt_choice")));
    config.verbose = matches!(verbose_input.to_lowercase().as_str(), "y" | "yes" | "ya");
    println!("  -> Verbose stderr: {}\n", config.verbose);

    // Step 5: Save & Exit
    config.save()?;
    println!("============================================================");
    println!("  [OK] {}", lang.text("wizard.saved_success"));
    println!("  {}", AppConfig::config_file_path().display());
    println!("============================================================");

    Ok(())
}

fn print_help(lang: &UiLanguage) {
    println!("Catetin CLI - {}", lang.text("app.subtitle"));
    println!();
    println!("{} catetin [OPTIONS]", lang.text("cli.usage"));
    println!();
    println!("{}", lang.text("cli.options"));
    println!("  --config, -c [path]      {}", lang.text("cli.help_config"));
    println!("  --lang, -l <code>        {}", lang.text("cli.help_lang"));
    println!("  --mode, -m <mode>        {}", lang.text("cli.help_mode"));
    println!("  --model <tier>           {}", lang.text("cli.help_model"));
    println!("  --verbose, -v            {}", lang.text("cli.help_verbose"));
    println!("  --help, -h               Show this help message / Tampilkan bantuan");
    println!("  --version                Show version information / Tampilkan versi");
    println!();
    println!("Examples / Contoh:");
    println!("  catetin --config              # Jalankan wizard konfigurasi interaktif");
    println!("  catetin                       # Jalankan dengan konfigurasi tersimpan");
    println!("  catetin --lang en             # Jalankan sementara dalam bahasa Inggris");
    println!("  catetin --mode in-person      # Mode rapat tatap muka / conference mic");
    println!("  catetin -c ./my-meeting.json  # Muat konfigurasi khusus");
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    // 1. Check if user wants interactive config wizard or custom file path
    let mut custom_config_path: Option<PathBuf> = None;
    let mut trigger_wizard = false;

    let mut i = 1;
    while i < args.len() {
        if args[i] == "--config" || args[i] == "-c" || args[i] == "config" {
            if i + 1 < args.len() && !args[i + 1].starts_with('-') {
                custom_config_path = Some(PathBuf::from(&args[i + 1]));
            } else {
                trigger_wizard = true;
            }
            break;
        }
        i += 1;
    }

    // 2. Load base configuration
    let mut config = if let Some(ref path) = custom_config_path {
        AppConfig::load_from_path(path).unwrap_or_else(|_| AppConfig::default())
    } else {
        AppConfig::load_or_default()
    };

    // 3. Launch interactive wizard if requested
    if trigger_wizard {
        return run_interactive_wizard(&mut config);
    }

    // 4. Parse one-off command-line flag overrides
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--config" | "-c" | "config" => {
                if i + 1 < args.len() && !args[i + 1].starts_with('-') {
                    i += 1;
                }
            }
            "--lang" | "-l" => {
                if i + 1 < args.len() {
                    config.ui_language = UiLanguage::from_code(&args[i + 1]);
                    i += 1;
                }
            }
            "--mode" | "-m" => {
                if i + 1 < args.len() {
                    match args[i + 1].to_lowercase().as_str() {
                        "in-person" | "roundtable" | "offline" => {
                            config.diarization_mode = DiarizationMode::InPerson;
                        }
                        "hybrid" => {
                            config.diarization_mode = DiarizationMode::Hybrid;
                        }
                        _ => {
                            config.diarization_mode = DiarizationMode::OnlineMeeting;
                        }
                    }
                    i += 1;
                }
            }
            "--model" => {
                if i + 1 < args.len() {
                    config.model_tier = args[i + 1].clone();
                    i += 1;
                }
            }
            "--verbose" | "-v" => {
                config.verbose = true;
            }
            "--help" | "-h" => {
                print_help(&config.ui_language);
                return Ok(());
            }
            "--version" => {
                println!("Catetin v{}", env!("CARGO_PKG_VERSION"));
                return Ok(());
            }
            _ => {}
        }
        i += 1;
    }

    // 5. Initialize dedicated file logger (with automatic local fallback)
    let log_path = init_logger(&config, config.verbose)?;

    let lang = &config.ui_language;
    let mode_desc = match config.diarization_mode {
        DiarizationMode::OnlineMeeting => lang.text("label.mode_online"),
        DiarizationMode::InPerson => lang.text("label.mode_in_person"),
        DiarizationMode::Hybrid => lang.text("label.mode_hybrid"),
    };

    println!("============================================================");
    println!(
        "  Catetin v{} - \"{}\"",
        env!("CARGO_PKG_VERSION"),
        lang.text("app.tagline")
    );
    println!("  {}", lang.text("app.subtitle"));
    println!("============================================================");
    println!(
        "  {}:     {} ({})",
        lang.text("label.language"),
        lang.display_name(),
        lang.code()
    );
    println!("  {}:         {}", lang.text("label.mode"), mode_desc);
    println!(
        "  {}:   {} ({})",
        lang.text("label.model_tier"),
        config.model_tier,
        config.model_filename()
    );
    println!("  {}:     {}", lang.text("label.log_file"), log_path.display());
    println!("  {}:       {}", lang.text("label.status"), lang.text("status.ready"));
    println!("  {}:          {}", lang.text("label.tip"), lang.text("prompt.press_quit"));
    println!("============================================================");

    log::info!(
        "CLI initialized with mode: {:?}, lang: {:?}, config: {:?}",
        config.diarization_mode,
        config.ui_language,
        custom_config_path.as_deref().unwrap_or(&AppConfig::config_file_path())
    );

    Ok(())
}
