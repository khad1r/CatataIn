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
