# Catetin Localization (i18n)

Welcome! You can easily add support for your native language in Catetin without modifying any Rust engine code.

## How to Add a New Language

1. Copy `en.json` to `<language_code>.json` in this directory (e.g., `es.json` for Spanish, `jv.json` for Javanese, `ja.json` for Japanese, `de.json` for German, `fr.json` for French).
2. Set the `code` and `name` attributes:
   ```json
   {
     "code": "es",
     "name": "Español",
     "strings": {
       "app.tagline": "¡Por favor toma notas de esta reunión!",
       "speaker.you": "Tú",
       "speaker.prefix": "Hablante",
       ...
     }
   }
   ```
3. Submit a Pull Request!

## Automatic Loading at Runtime (No Recompilation Needed)

Users can also drop any `<language_code>.json` file into:
- Windows: `%APPDATA%\catetin\locales\`
- Linux/macOS: `~/.catetin/locales/`
- Local directory: `./locales/`

Catetin automatically scans and loads all `.json` files in these directories on startup. Users can immediately launch Catetin with:
```bash
catetin --lang <language_code>
```
