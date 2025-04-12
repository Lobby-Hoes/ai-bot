import os, json, requests, re, asyncio, logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    import google.generativeai as genai
    from google.generativeai.types import GenerationConfig
except ImportError:
    logging.error("google.generativeai nicht gefunden. Bitte installieren: pip install google-generativeai python-dotenv")
    exit()
except AttributeError:
    logging.error("Konnte GenerationConfig nicht importieren. Überprüfen Sie die google-generativeai Version.")
    exit()


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SEARCH_API_URL = os.getenv("SEARCH_API_URL", "https://localhost/")
CACHE_DIR = os.getenv("CACHE_DIR", "cache/forward")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-pro-exp-03-25")

if not GOOGLE_API_KEY:
    logging.error("GOOGLE_API_KEY Umgebungsvariable nicht gesetzt!")
    exit()

os.makedirs(CACHE_DIR, exist_ok=True)
logging.info(f"Cache-Verzeichnis: {os.path.abspath(CACHE_DIR)}")
logging.info(f"Such-API URL: {SEARCH_API_URL}")

genai_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global genai_client
    if GOOGLE_API_KEY:
        try:
            genai.configure(api_key=GOOGLE_API_KEY)
            genai_client = genai.GenerativeModel(model_name=GEMINI_MODEL,)
            logging.info("Google GenAI Client erfolgreich initialisiert.")
        except AttributeError as e:
             logging.error(f"Fehler bei der Initialisierung des Google GenAI Clients (Attribut nicht gefunden): {e}. Bitte überprüfen Sie Ihre 'google-generativeai' Version und Installation.")
             genai_client = None
        except Exception as e:
            logging.error(f"Allgemeiner Fehler bei der Initialisierung des Google GenAI Clients: {e}")
            genai_client = None
    else:
        logging.warning("Kein GOOGLE_API_KEY gefunden. KI-Funktionen sind deaktiviert.")
    yield
    logging.info("Anwendung wird beendet.")


app = FastAPI(lifespan=lifespan)

def clean_text(text): return re.sub(r'[^\w\säöüÄÖÜß-]', '', text).strip().lower().replace(" ", "_")

def calculate_cost(prompt_token_count, candidates_token_count):
    if GEMINI_MODEL == "gemini-2.5-pro-exp-03-25":
        prompt_cost_per_1M_tokens = 0
        candidates_cost_per_1M_tokens = 0
    elif GEMINI_MODEL == "gemini-2.5-pro-preview-03-25":
        prompt_cost_per_1M_tokens = 1.25
        candidates_cost_per_1M_tokens = 10
    else:
        logging.warning(f"Unbekanntes Modell '{GEMINI_MODEL}'. Verwende Standardpreise von 1€ pro 1M Tokens und 10€ pro 1M Tokens.")
        prompt_cost_per_1M_tokens = 1
        candidates_cost_per_1M_tokens = 10
    prompt_cost = (prompt_cost_per_1M_tokens / 1000000) * prompt_token_count
    candidates_cost = (candidates_cost_per_1M_tokens / 1000000) * candidates_token_count
    total_cost = prompt_cost + candidates_cost
    return total_cost

def extract_first_json(text):
    first_brace = text.find('{')
    first_bracket = text.find('[')
    start_index = -1

    if first_brace == -1 and first_bracket == -1: return None
    elif first_brace == -1: start_index = first_bracket
    elif first_bracket == -1: start_index = first_brace
    else: start_index = min(first_brace, first_bracket)

    if start_index == -1: return None

    balance = 0
    in_string = False
    escape = False
    end_index = -1

    char_start = text[start_index]
    char_end = '}' if char_start == '{' else ']'

    for i in range(start_index, len(text)):
        char = text[i]

        if in_string:
            if char == '"' and not escape: in_string = False
            elif char == '\\': escape = not escape
            else: escape = False
        else:
            if char == '"':
                in_string = True
                escape = False
            elif char == char_start: balance += 1
            elif char == char_end: balance -= 1

        if balance == 0 and i >= start_index:
            end_index = i
            substring = text[start_index : end_index + 1]
            try:
                result = json.loads(substring)
                return result
            except json.JSONDecodeError as e:
                logging.warning(f"Konnte kein valides JSON parsen, obwohl Balance 0 erreicht wurde bei Index {i}. Substring: {substring[:100]}...")
                return None

    logging.warning(f"Konnte kein valides JSON in Text finden, der mit '{char_start}' beginnt und endet.")
    return None


async def web_requests_async(keyword: str):
    clean_keyword = clean_text(text=keyword)
    if not clean_keyword:
        logging.warning(f"Keyword '{keyword}' wurde zu einem leeren String bereinigt. Überspringe.")
        return None
    filepath = os.path.join(CACHE_DIR, f"{clean_keyword}.json")
    data = None

    if os.path.exists(filepath):
        try:
            loop = asyncio.get_running_loop()
            with open(filepath, 'r', encoding='utf-8') as f: content = await loop.run_in_executor(None, f.read)
            data = json.loads(content)
            logging.info(f"'{keyword}' aus lokalem Cache geladen: {filepath}")
            return data
        except FileNotFoundError:
            logging.warning(f"Cache-Datei {filepath} nicht gefunden (Race Condition?). Fahre mit API fort.")
            data = None
        except Exception as e:
            logging.warning(f"Fehler beim Lesen/Parsen der Cache-Datei {filepath}: {e}")
            data = None

    encoded_keyword = requests.utils.quote(keyword)
    api_url = f"{SEARCH_API_URL}?word={encoded_keyword}"
    alle_ergebnisse = []
    try:
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: requests.get(api_url, timeout=20)
        )
        response.raise_for_status()
        data = response.json()
        try:
            loop = asyncio.get_running_loop()
            with open(filepath, 'w', encoding='utf-8') as f:
                await loop.run_in_executor(None, json.dump, data, f, ensure_ascii=False, indent=4)
        except Exception as e: pass
        await asyncio.sleep(0.25)
        print(f"Antwort größe für {keyword}: {len(str(data))}")
        alle_ergebnisse.append(data)
    except requests.exceptions.Timeout: logging.error(f"Timeout bei der API-Anfrage für '{keyword}' an {api_url}")
    except requests.exceptions.RequestException as e: logging.error(f"Fehler bei der API-Anfrage für '{keyword}' an {api_url}: {e}")
    except json.JSONDecodeError as e: logging.error(f"Fehler beim Parsen der JSON-Antwort von API für '{keyword}': {e}. Antworttext: {response.text[:200]}")
    except Exception as e: logging.error(f"Unerwarteter Fehler bei web_requests für '{keyword}': {e}")
    try:
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: requests.get(f"{api_url}&sort_by_time=forward", timeout=20)
        )
        response.raise_for_status()
        data = response.json()
        try:
            loop = asyncio.get_running_loop()
            with open(filepath, 'w', encoding='utf-8') as f:
                await loop.run_in_executor(None, json.dump, data, f, ensure_ascii=False, indent=4)
        except Exception as e: pass
        await asyncio.sleep(0.25)
        print(f"Antwort größe für {keyword}: {len(str(data))}")
        alle_ergebnisse.append(data)
    except requests.exceptions.Timeout: logging.error(f"Timeout bei der API-Anfrage für '{keyword}' an {api_url}")
    except requests.exceptions.RequestException as e: logging.error(f"Fehler bei der API-Anfrage für '{keyword}' an {api_url}: {e}")
    except json.JSONDecodeError as e: logging.error(f"Fehler beim Parsen der JSON-Antwort von API für '{keyword}': {e}. Antworttext: {response.text[:200]}")
    except Exception as e: logging.error(f"Unerwarteter Fehler bei web_requests für '{keyword}': {e}")
    try:
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: requests.get(f"{api_url}&sort_by_time=reverse", timeout=20)
        )
        response.raise_for_status()
        data = response.json()
        try:
            loop = asyncio.get_running_loop()
            with open(filepath, 'w', encoding='utf-8') as f:
                await loop.run_in_executor(None, json.dump, data, f, ensure_ascii=False, indent=4)
        except Exception as e: pass
        await asyncio.sleep(0.25)
        print(f"Antwort größe für {keyword}: {len(str(data))}")
        alle_ergebnisse.append(data)
    except requests.exceptions.Timeout: logging.error(f"Timeout bei der API-Anfrage für '{keyword}' an {api_url}")
    except requests.exceptions.RequestException as e: logging.error(f"Fehler bei der API-Anfrage für '{keyword}' an {api_url}: {e}")
    except json.JSONDecodeError as e: logging.error(f"Fehler beim Parsen der JSON-Antwort von API für '{keyword}': {e}. Antworttext: {response.text[:200]}")
    except Exception as e: logging.error(f"Unerwarteter Fehler bei web_requests für '{keyword}': {e}")
    return alle_ergebnisse

async def get_keywords(frage: str, websocket: WebSocket):
    if not genai_client:
        error_msg = "Fehler: Google GenAI Client nicht initialisiert (API-Schlüssel fehlt oder Initialisierung fehlgeschlagen)."
        await websocket.send_text(error_msg)
        logging.warning(error_msg)
        await websocket.send_text("__ENDOFTASK__")
        return []

    await websocket.send_text("Status: Generiere Keywords...")
    logging.info(f"Generiere Keywords für: {frage}")

    system_instruction = """Du bist eine KI, die verstehen soll, wonach der Benutzer sucht, und die Schlüsselwörter dafür finden soll. Diese Schlüsselwörter werden verwendet, um den deutschen Podcast Hobbylos von Rezo und Ju (Julien Bam) nach der vom Benutzer gesuchten Stelle zu durchsuchen. Gib die Schlüsselwörter im JSON-Format aus: {"keywords":["keyword1", "keyword2"...]}. Schließe auch Synonyme ein. Gib NUR das JSON-Objekt aus, nichts davor oder danach."""

    try:
        generation_config_keywords=GenerationConfig(
            temperature=0.8,
            response_mime_type="text/plain"
        )
    except NameError:
        logging.error("GenerationConfig Klasse nicht verfügbar. GenAI Initialisierung möglicherweise fehlgeschlagen.")
        await websocket.send_text("Fehler: Interne Konfiguration für KI fehlgeschlagen.")
        await websocket.send_text("__ENDOFTASK__")
        return []

    full_response_text = ""
    keywords = []
    try:
        response_stream = await genai_client.generate_content_async(
            generation_config=generation_config_keywords,
            stream=True,
            safety_settings={'HARASSMENT':'block_none', 'HATE_SPEECH':'block_none','SEXUAL':'block_none','DANGEROUS':'block_none'},
            contents=f"{system_instruction}\n\nUser Frage: {frage}",
        )

        if response_stream.usage_metadata:
            total_cost = calculate_cost(response_stream.usage_metadata.prompt_token_count, response_stream.usage_metadata.candidates_token_count)
            logging.info(f"prompt_token_count: {response_stream.usage_metadata.prompt_token_count}")
            logging.info(f"candidates_token_count: {response_stream.usage_metadata.candidates_token_count}")
            logging.info(f"total_token_count: {response_stream.usage_metadata.total_token_count}")
            logging.info(f"Kosten der Anfrage: {total_cost}€")
            await websocket.send_text(f"Status: Kosten der Anfrage: {total_cost}€")
        else: print("\nKeine Nutzungsmetadaten in dieser Antwort verfügbar.")

        async for chunk in response_stream:
            if hasattr(chunk, 'text') and chunk.text: full_response_text += chunk.text
            elif hasattr(chunk, 'prompt_feedback') and chunk.prompt_feedback.block_reason:
                 reason = chunk.prompt_feedback.block_reason
                 logging.error(f"Keyword-Generierung blockiert durch Sicherheitsfilter: {reason}")
                 await websocket.send_text(f"Fehler: Anfrage wurde aus Sicherheitsgründen blockiert ({reason}).")
                 await websocket.send_text("__ENDOFTASK__")
                 return []

    except Exception as e:
        logging.exception(f"Fehler bei der Keyword-Generierung mit GenAI: {e}") 
        await websocket.send_text(f"Fehler: Konnte Keywords nicht generieren: {e}")
        await websocket.send_text("__ENDOFTASK__")
        return []

    logging.info(f"Vollständige Keyword-Antwort von GenAI: {full_response_text}")

    keyword_json = extract_first_json(full_response_text)

    if keyword_json and isinstance(keyword_json.get("keywords"), list):
        keywords = keyword_json["keywords"]
        keywords = [kw for kw in keywords if kw and isinstance(kw, str)]
        if not keywords:
        #     logging.info(f"Extrahierte Keywords: {keywords}")
        #     pass
        # else:
            logging.warning(f"Keyword-Liste war leer oder enthielt nur ungültige Einträge im JSON: {keyword_json}")
            await websocket.send_text("Warnung: Keine gültigen Keywords im JSON gefunden.")
            await websocket.send_text("__ENDOFTASK__")
            keywords = []
    else:
        logging.warning(f"Konnte kein valides Keyword-JSON extrahieren oder 'keywords' war keine Liste in: {full_response_text}")
        await websocket.send_text("Warnung: Konnte Keywords nicht automatisch extrahieren.")
        await websocket.send_text("__ENDOFTASK__")
        keywords = []

    return keywords


async def get_answer(antworten_json_string: str, frage: str, websocket: WebSocket):
    if not genai_client:
        error_msg = "Fehler: Google GenAI Client nicht initialisiert."
        await websocket.send_text(error_msg)
        logging.warning(error_msg)
        await websocket.send_text("__ENDOFTASK__")
        return

    await websocket.send_text("Status: Generiere finale Antwort...")
    logging.info("Generiere finale Antwort...")

    system_instruction_answer = """Du bist eine KI, die verwendet wird, um die JSON-Ausgabe einer Suchmaschine für den deutschen Podcast Hobbylos von Rezo und Ju (Julien Bam) zu verarbeiten und die 1-10 am besten passenden Ergebnisse als Link mit diesem Schema (<a href="https://open.spotify.com/episode/{f}?go=1&t={s}&utm_source=search.hobbylos.online" target="_blank" rel="noopener noreferrer">Beschreibung</a>) auszugeben. Überlege sorgfältig, welche Positionen am besten zur Anfrage des Benutzers passen. Antworte auf Deutsch. Gib vor der Ausgabe der Links eine kurze Antwort auf das Gesuchte, falls möglich. Formatiere die Ausgabe mit Markdown. Stelle sicher, dass Links in einem neuen Tab geöffnet werden."""

    try: generation_config_answer=GenerationConfig(response_mime_type="text/plain")
    except NameError:
        logging.error("GenerationConfig Klasse nicht verfügbar. GenAI Initialisierung möglicherweise fehlgeschlagen.")
        await websocket.send_text("Fehler: Interne Konfiguration für KI fehlgeschlagen.")
        await websocket.send_text("__ENDOFTASK__")
        return

    prompt = f"{system_instruction_answer}\n\nBenutzerfrage: {frage}\n\nSuchergebnisse (JSON):\n{antworten_json_string}\n\nBitte werte diese Ergebnisse aus und gib die relevantesten Stellen gemäß den Anweisungen als Markdown formatierten Text mit HTML-Links aus."

    try:
        response_stream = await genai_client.generate_content_async(
            contents=prompt,
            generation_config=generation_config_answer,
            stream=True,
            safety_settings={'HARASSMENT':'block_none', 'HATE_SPEECH':'block_none','SEXUAL':'block_none','DANGEROUS':'block_none'}
        )

        if response_stream.usage_metadata:
            total_cost = calculate_cost(response_stream.usage_metadata.prompt_token_count, response_stream.usage_metadata.candidates_token_count)
            logging.info(f"prompt_token_count: {response_stream.usage_metadata.prompt_token_count}")
            logging.info(f"candidates_token_count: {response_stream.usage_metadata.candidates_token_count}")
            logging.info(f"total_token_count: {response_stream.usage_metadata.total_token_count}")
            logging.info(f"Kosten der Anfrage: {total_cost}€")
            await websocket.send_text(f"Status: Kosten der Anfrage: {total_cost}€")
        else: print("\nKeine Nutzungsmetadaten in dieser Antwort verfügbar.")

        async for chunk in response_stream:
            if hasattr(chunk, 'text') and chunk.text:
                await websocket.send_text(chunk.text)
            elif hasattr(chunk, 'prompt_feedback') and chunk.prompt_feedback.block_reason:
                 reason = chunk.prompt_feedback.block_reason
                 logging.error(f"Antwort-Generierung blockiert durch Sicherheitsfilter: {reason}")
                 await websocket.send_text(f"\nFehler: Antwortgenerierung wurde aus Sicherheitsgründen blockiert ({reason}).")
                 await websocket.send_text("__ENDOFTASK__")
                 return

        await websocket.send_text("__ENDOFTASK__")
        logging.info("Antwort-Stream erfolgreich beendet.")

    except Exception as e:
        logging.exception(f"Fehler bei der Antwort-Generierung mit GenAI: {e}")
        try:
            await websocket.send_text(f"\nFehler: Konnte finale Antwort nicht generieren: {e}")
            await websocket.send_text("__ENDOFTASK__")
        except Exception as ws_err:
            logging.error(f"Konnte Fehler nicht an WebSocket senden: {ws_err}")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logging.info(f"WebSocket-Verbindung hergestellt von: {websocket.client.host}:{websocket.client.port}")
    try:
        while True:
            frage = await websocket.receive_text()
            logging.info(f"Nachricht empfangen: {frage}")

            if not frage.strip(): continue

            keywords = await get_keywords(frage, websocket)

            if not keywords:
                logging.info("Keine Keywords gefunden oder Fehler in get_keywords. Warte auf nächste Nachricht.")
                continue
            
            await websocket.send_text(f"Status: Suche mit '{SEARCH_API_URL}' nach: {', '.join(keywords)}...")
            search_tasks = [asyncio.create_task(web_requests_async(key), name=f"search_{key}") for key in keywords]
            done, pending = await asyncio.wait(search_tasks, return_when=asyncio.ALL_COMPLETED)

            search_results = []
            for task in done:
                try:
                    result = task.result()
                    if result is not None: search_results.append(result)
                except Exception as e:
                    logging.error(f"Fehler im Such-Task '{task.get_name()}': {e}")

            if not search_results:
                logging.warning("Keine gültigen Suchergebnisse von der API erhalten.")
                await websocket.send_text("Status: Keine Suchergebnisse für die Keywords gefunden.")
                await websocket.send_text("__ENDOFTASK__")
                continue

            try:
                MAX_JSON_LENGTH = 1000000
                antworten_json_string = json.dumps(search_results, ensure_ascii=False, indent=2)
                if len(antworten_json_string) > MAX_JSON_LENGTH:
                    logging.warning(f"Suchergebnis-JSON ({len(antworten_json_string)} Zeichen) überschreitet Limit ({MAX_JSON_LENGTH}). Kürze...")
                    antworten_json_string = antworten_json_string[:MAX_JSON_LENGTH] + "...]}"

                logging.info(f"Suchergebnisse erfolgreich zu JSON zusammengefasst (Länge: {len(antworten_json_string)} Zeichen).")

            except Exception as e:
                 logging.error(f"Fehler beim Umwandeln der Suchergebnisse in JSON: {e}")
                 await websocket.send_text("Fehler: Konnte Suchergebnisse nicht verarbeiten.")
                 await websocket.send_text("__ENDOFTASK__")
                 continue

            await get_answer(antworten_json_string, frage, websocket)

    except WebSocketDisconnect: logging.info(f"WebSocket-Verbindung geschlossen von: {websocket.client.host}:{websocket.client.port}")
    except Exception as e:
        logging.exception(f"Unerwarteter WebSocket-Fehler: {e}")
        try:
            await websocket.send_text(f"Ein interner Serverfehler ist aufgetreten. Bitte versuchen Sie es später erneut.")
            await websocket.send_text("__ENDOFTASK__")
        except Exception: pass
    finally:
        try:
            await websocket.close()
            logging.info(f"WebSocket-Verbindung explizit geschlossen für: {websocket.client.host}:{websocket.client.port}")
        except Exception: pass

@app.get("/", response_class=HTMLResponse)
async def get_root():
    html_file_path = "frontend.html"
    try:
        loop = asyncio.get_running_loop()
        with open(html_file_path, "r", encoding="utf-8") as f: html_content = await loop.run_in_executor(None, f.read)
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        logging.error(f"{html_file_path} nicht gefunden!")
        error_content = "<html><head><title>Fehler</title></head><body><h1>Fehler 500</h1><p>Entschuldigung, die Frontend-Datei konnte nicht gefunden werden.</p></body></html>"
        return HTMLResponse(content=error_content, status_code=500)
    except Exception as e:
        logging.error(f"Fehler beim Lesen von {html_file_path}: {e}")
        error_content = "<html><head><title>Fehler</title></head><body><h1>Fehler 500</h1><p>Entschuldigung, das Frontend konnte nicht geladen werden.</p></body></html>"
        return HTMLResponse(content=error_content, status_code=500)

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 80))
    host = os.getenv("HOST", "0.0.0.0")
    reload_status = os.getenv("RELOAD", "true").lower() == "true"

    logging.info(f"Starte Server auf {host}:{port} mit Reload={reload_status}")
    logging.info("Stelle sicher, dass die Firewall eingehende Verbindungen auf Port {port} erlaubt.")

    uvicorn.run("main:app", host=host, port=port, reload=reload_status)