use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::path::Path;
use std::sync::{OnceLock, RwLock};

/// A parsed locale file containing translation key-value mappings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocaleBundle {
    pub code: String,
    pub name: String,
    pub strings: HashMap<String, String>,
}

/// The global multi-language translation catalog.
#[derive(Debug)]
pub struct LocaleCatalog {
    bundles: HashMap<String, LocaleBundle>,
}

impl Default for LocaleCatalog {
    fn default() -> Self {
        let mut catalog = Self {
            bundles: HashMap::new(),
        };
        catalog.load_builtin();
        catalog
    }
}

impl LocaleCatalog {
    /// Load built-in embedded locales (Bahasa Indonesia & English)
    pub fn load_builtin(&mut self) {
        let id_json = include_str!("../../locales/id.json");
        let en_json = include_str!("../../locales/en.json");

        if let Ok(bundle) = serde_json::from_str::<LocaleBundle>(id_json) {
            self.bundles.insert(bundle.code.to_lowercase(), bundle);
        }
        if let Ok(bundle) = serde_json::from_str::<LocaleBundle>(en_json) {
            self.bundles.insert(bundle.code.to_lowercase(), bundle);
        }
    }

    /// Register a newly parsed locale bundle into the catalog
    pub fn register(&mut self, bundle: LocaleBundle) {
        self.bundles.insert(bundle.code.to_lowercase(), bundle);
    }

    /// Auto-discover and load any custom locale `.json` files from disk.
    ///
    /// Contributors and users can add new languages simply by dropping a `.json`
    /// file into `./locales/` or the application's config folder.
    pub fn auto_load_from_dir<P: AsRef<Path>>(&mut self, dir: P) -> usize {
        let path = dir.as_ref();
        if !path.exists() || !path.is_dir() {
            return 0;
        }

        let mut loaded_count = 0;
        if let Ok(entries) = std::fs::read_dir(path) {
            for entry in entries.flatten() {
                let entry_path = entry.path();
                if entry_path.extension().and_then(|s| s.to_str()) == Some("json") {
                    if let Ok(content) = std::fs::read_to_string(&entry_path) {
                        if let Ok(bundle) = serde_json::from_str::<LocaleBundle>(&content) {
                            self.register(bundle);
                            loaded_count += 1;
                        }
                    }
                }
            }
        }
        loaded_count
    }

    /// Translate a key for a given language code with multi-tier fallback:
    /// Requested Language -> Indonesian -> English -> Raw Key
    pub fn get(&self, lang_code: &str, key: &str) -> String {
        let target = lang_code.to_lowercase();

        // 1. Target language lookup
        if let Some(bundle) = self.bundles.get(&target) {
            if let Some(val) = bundle.strings.get(key) {
                return val.clone();
            }
        }

        // 2. Indonesian fallback
        if target != "id" {
            if let Some(bundle) = self.bundles.get("id") {
                if let Some(val) = bundle.strings.get(key) {
                    return val.clone();
                }
            }
        }

        // 3. English fallback
        if target != "en" {
            if let Some(bundle) = self.bundles.get("en") {
                if let Some(val) = bundle.strings.get(key) {
                    return val.clone();
                }
            }
        }

        // 4. Return the key itself as safe fallback
        key.to_string()
    }

    /// List all registered language codes
    pub fn available_languages(&self) -> Vec<(&str, &str)> {
        self.bundles
            .values()
            .map(|b| (b.code.as_str(), b.name.as_str()))
            .collect()
    }
}

/// Global lazy-initialized locale catalog
static CATALOG: OnceLock<RwLock<LocaleCatalog>> = OnceLock::new();

fn global_catalog() -> &'static RwLock<LocaleCatalog> {
    CATALOG.get_or_init(|| {
        let mut cat = LocaleCatalog::default();
        // Auto-load from local ./locales/ directory if present
        cat.auto_load_from_dir("locales");
        cat.auto_load_from_dir("../locales");
        cat.auto_load_from_dir("crates/catetin-core/locales");
        // Auto-load from system AppData directory
        cat.auto_load_from_dir(crate::config::default_app_dir().join("locales"));
        RwLock::new(cat)
    })
}

/// Explicitly trigger auto-loading locales from a specific folder
pub fn auto_load_locales<P: AsRef<Path>>(dir: P) -> usize {
    if let Ok(mut lock) = global_catalog().write() {
        lock.auto_load_from_dir(dir)
    } else {
        0
    }
}

/// Supported user interface and speech recognition prompt languages.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum UiLanguage {
    /// Bahasa Indonesia (default: "Tolong catetin rapat ini ya!")
    #[serde(alias = "id", alias = "indonesian", alias = "in")]
    Indonesian,

    /// English
    #[serde(alias = "en", alias = "english")]
    English,

    /// Dynamically discovered contributor language (e.g. "es", "jv", "ja", "de")
    Custom(String),
}

impl Default for UiLanguage {
    fn default() -> Self {
        UiLanguage::Indonesian
    }
}

impl UiLanguage {
    /// Parse language from ISO 639-1 code or common name.
    pub fn from_code(code: &str) -> Self {
        let clean = code.trim().to_lowercase();
        match clean.as_str() {
            "id" | "indonesian" | "in" => UiLanguage::Indonesian,
            "en" | "english" => UiLanguage::English,
            custom => UiLanguage::Custom(custom.to_string()),
        }
    }

    /// Two-letter ISO language code (e.g. "id", "en", "es").
    pub fn code(&self) -> &str {
        match self {
            UiLanguage::Indonesian => "id",
            UiLanguage::English => "en",
            UiLanguage::Custom(ref s) => s.as_str(),
        }
    }

    /// Display name of the language.
    pub fn display_name(&self) -> String {
        if let Ok(lock) = global_catalog().read() {
            if let Some(bundle) = lock.bundles.get(self.code()) {
                return bundle.name.clone();
            }
        }
        match self {
            UiLanguage::Indonesian => "Bahasa Indonesia".to_string(),
            UiLanguage::English => "English".to_string(),
            UiLanguage::Custom(code) => code.to_uppercase(),
        }
    }

    /// Translate a string key into the current language.
    pub fn text(&self, key: &str) -> String {
        if let Ok(lock) = global_catalog().read() {
            lock.get(self.code(), key)
        } else {
            key.to_string()
        }
    }

    /// Format a string with arguments {0}, {1}, etc.
    pub fn text_fmt(&self, key: &str, args: &[&str]) -> String {
        let mut result = self.text(key);
        for (i, arg) in args.iter().enumerate() {
            let placeholder = format!("{{{}}}", i);
            result = result.replace(&placeholder, arg);
        }
        result
    }
}

impl fmt::Display for UiLanguage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.code())
    }
}

/// Global translate lookup function
pub fn t(lang: &UiLanguage, key: &str) -> String {
    lang.text(key)
}

/// Helper function to format numbered speaker labels according to language.
/// e.g. "Pembicara 1" (id) vs "Speaker 1" (en).
pub fn format_speaker_label(speaker_num: u32, lang: &UiLanguage) -> String {
    let prefix = lang.text("speaker.prefix");
    format!("{} {}", prefix, speaker_num)
}

/// Macro for ergonomic translation lookups:
/// `tr!(lang, "speaker.you")` -> "Anda"
/// `tr!(lang, "speaker.format", 1)` -> "Pembicara 1"
#[macro_export]
macro_rules! tr {
    ($lang:expr, $key:expr) => {
        $lang.text($key)
    };
    ($lang:expr, $key:expr, $($arg:expr),+) => {
        $lang.text_fmt($key, &[$(&$arg.to_string()),+])
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builtin_localization() {
        let id = UiLanguage::Indonesian;
        let en = UiLanguage::English;

        assert_eq!(id.text("app.tagline"), "Tolong catetin rapat ini ya!");
        assert_eq!(en.text("app.tagline"), "Please take notes for this meeting!");

        assert_eq!(id.text("speaker.you"), "Anda");
        assert_eq!(en.text("speaker.you"), "You");

        assert_eq!(format_speaker_label(1, &id), "Pembicara 1");
        assert_eq!(format_speaker_label(2, &en), "Speaker 2");
    }

    #[test]
    fn test_custom_contributor_bundle_auto_load() {
        let mut catalog = LocaleCatalog::default();
        let es_bundle = LocaleBundle {
            code: "es".to_string(),
            name: "Español".to_string(),
            strings: HashMap::from([
                ("app.tagline".to_string(), "¡Por favor toma notas!".to_string()),
                ("speaker.you".to_string(), "Tú".to_string()),
                ("speaker.prefix".to_string(), "Hablante".to_string()),
            ]),
        };

        catalog.register(es_bundle);

        assert_eq!(catalog.get("es", "speaker.you"), "Tú");
        assert_eq!(catalog.get("es", "app.tagline"), "¡Por favor toma notas!");
        // Fallback test: key not in Spanish -> falls back to Indonesian/English
        assert_eq!(catalog.get("es", "status.ready"), "Siap merekam rapat");
    }

    #[test]
    fn test_dynamic_language_parsing() {
        let es = UiLanguage::from_code("es");
        assert_eq!(es.code(), "es");
        match es {
            UiLanguage::Custom(c) => assert_eq!(c, "es"),
            _ => panic!("Expected UiLanguage::Custom"),
        }
    }
}
