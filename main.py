import sys, os, re, json, subprocess, threading
from datetime import timedelta

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QTextEdit, QProgressBar,
    QLineEdit, QMessageBox, QFrame, QComboBox, QStackedWidget,
    QScrollArea, QCheckBox, QSpinBox, QListWidget, QListWidgetItem
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QCursor, QPixmap
from PyQt5.QtCore import QUrl

try:
    from faster_whisper import WhisperModel
    WHISPER_OK = True
except ImportError:
    WHISPER_OK = False

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas as pdf_canvas
    PDF_OK = True
except ImportError:
    PDF_OK = False

try:
    from youtube_transcript_api import YouTubeTranscriptApi
    YT_OK = True
except ImportError:
    YT_OK = False

try:
    import yt_dlp
    YTDLP_OK = True
except ImportError:
    YTDLP_OK = False

try:
    import urllib.request
    URLLIB_OK = True
except ImportError:
    URLLIB_OK = False

import ollama

STYLE = """
* { font-family: 'Segoe UI', Arial; }
QMainWindow, QWidget#root { background: #0d0d1a; }
QWidget#sidebar { background: #12122a; border-right: 1px solid #1e1e3a; min-width: 200px; max-width: 200px; }
QPushButton#navBtn { background: transparent; color: #8888aa; border: none; border-radius: 10px; padding: 12px 16px; text-align: left; font-size: 14px; }
QPushButton#navBtn:hover { background: #1e1e3a; color: #fff; }
QPushButton#navBtn[active="true"] { background: #2d1b69; color: #a78bfa; font-weight: bold; }
QFrame#card { background: #12122a; border: 1px solid #1e1e3a; border-radius: 14px; }
QFrame#infoCard { background: #1a1a35; border: 1px solid #2a2a50; border-radius: 10px; }
QFrame#uploadZone { background: #0f0f22; border: 2px dashed #3a3a6a; border-radius: 16px; min-height: 160px; }
QFrame#uploadZone:hover { border-color: #7c3aed; }
QPushButton#primaryBtn { background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #7c3aed, stop:1 #4f46e5); color: white; border: none; border-radius: 10px; padding: 11px 24px; font-size: 14px; font-weight: bold; }
QPushButton#primaryBtn:hover { background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #9333ea, stop:1 #6366f1); }
QPushButton#primaryBtn:disabled { background: #2a2a4a; color: #555; }
QPushButton#secondaryBtn { background: #1e1e3a; color: #a78bfa; border: 1px solid #3a3a6a; border-radius: 10px; padding: 9px 20px; font-size: 13px; }
QPushButton#secondaryBtn:hover { background: #2a2a50; }
QPushButton#successBtn { background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #059669, stop:1 #0d9488); color: white; border: none; border-radius: 10px; padding: 9px 20px; font-size: 13px; font-weight: bold; }
QLineEdit, QTextEdit { background: #1a1a35; color: #e0e0f0; border: 1px solid #2a2a50; border-radius: 10px; padding: 10px; font-size: 13px; }
QLineEdit:focus, QTextEdit:focus { border: 1px solid #7c3aed; }
QProgressBar { background: #1a1a35; border: none; border-radius: 8px; height: 12px; text-align: center; color: transparent; }
QProgressBar::chunk { background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #7c3aed, stop:1 #06b6d4); border-radius: 8px; }
QLabel#pageTitle { font-size: 22px; font-weight: bold; color: #a78bfa; }
QLabel#sectionTitle { font-size: 14px; font-weight: bold; color: #c4b5fd; }
QLabel#metaLabel { font-size: 13px; color: #6666aa; }
QLabel#valueLabel { font-size: 14px; color: #e0e0f0; font-weight: bold; }
QLabel#statusLabel { font-size: 12px; color: #6666aa; }
QLabel#appName { font-size: 15px; font-weight: bold; color: #a78bfa; padding: 16px; }
QComboBox { background: #1a1a35; color: #e0e0f0; border: 1px solid #2a2a50; border-radius: 8px; padding: 7px 12px; }
QComboBox::drop-down { border: none; width: 24px; }
QComboBox QAbstractItemView { background: #1a1a35; color: #e0e0f0; border: 1px solid #2a2a50; }
QScrollBar:vertical { background: #0d0d1a; width: 6px; }
QScrollBar::handle:vertical { background: #3a3a6a; border-radius: 3px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QPushButton#optBtn { background: #1a1a35; color: #e0e0f0; border: 1px solid #2a2a50; border-radius: 10px; padding: 12px 16px; font-size: 13px; text-align: left; }
QPushButton#optBtn:hover:enabled { background: #2d1b69; border-color: #7c3aed; }
QCheckBox { color: #e0e0f0; spacing: 8px; }
QCheckBox::indicator { width: 16px; height: 16px; border: 1px solid #2a2a50; border-radius: 4px; background: #1a1a35; }
QCheckBox::indicator:checked { background: #7c3aed; border: 1px solid #7c3aed; }
QSpinBox { background: #1a1a35; color: #e0e0f0; border: 1px solid #2a2a50; border-radius: 8px; padding: 5px; }
QListWidget { background: #1a1a35; border: 1px solid #2a2a50; border-radius: 8px; color: #e0e0f0; }
QListWidget::item:hover { background: #2d1b69; }
QListWidget::item:selected { background: #7c3aed; }
"""

# ------------------------------------------------------------------ #
#  Worker Thread                                                       #
# ------------------------------------------------------------------ #

class Worker(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict)
    error    = pyqtSignal(str)

    def __init__(self, task, **kw):
        super().__init__()
        self.task = task
        self.kw   = kw

    def run(self):
        try:
            if self.task == "video":
                self._video()
            elif self.task == "youtube":
                self._youtube()
        except Exception as e:
            self.error.emit(str(e))

    # ---------------------------------------------------------------- #
    #  Local Video                                                       #
    # ---------------------------------------------------------------- #
    def _video(self):
        path  = self.kw["path"]
        model = self.kw["model"]
        lang  = self.kw.get("lang", "en")

        self.progress.emit(5, "Extracting thumbnail…")
        thumbnail = self._extract_thumbnail(path)

        self.progress.emit(10, "Extracting audio…")
        audio = path.rsplit(".", 1)[0] + "_tmp.wav"
        ret = os.system(
            f'ffmpeg -y -i "{path}" -vn -acodec pcm_s16le -ar 16000 -ac 1 "{audio}" -loglevel quiet'
        )
        if ret != 0:
            self.error.emit("ffmpeg not found. Please install ffmpeg:\nhttps://ffmpeg.org/download.html")
            return

        self.progress.emit(35, "Transcribing speech…")
        if not WHISPER_OK:
            self.error.emit("Run: pip install faster-whisper")
            return

        wm = WhisperModel("base", device="cpu", compute_type="int8")
        segs, _ = wm.transcribe(audio, language=None if lang == "auto" else lang)
        transcript = ""
        ts = []
        for s in segs:
            transcript += s.text + " "
            ts.append((s.start, s.text.strip()))

        try:
            os.remove(audio)
        except:
            pass

        self.progress.emit(65, "AI is analyzing…")
        result = self._ollama(transcript, model, ts, lang)

        try:
            probe = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json",
                 "-show_format", "-show_streams", path],
                capture_output=True, text=True
            )
            info_json = json.loads(probe.stdout)
            duration  = float(info_json["format"].get("duration", 0))
            result["duration"]  = str(timedelta(seconds=int(duration)))
            result["size"]      = f"{os.path.getsize(path) / (1024 * 1024):.1f} MB"
            result["filename"]  = os.path.basename(path)
            result["resolution"] = "N/A"
            for stream in info_json.get("streams", []):
                if stream.get("codec_type") == "video":
                    result["resolution"] = f"{stream.get('width','N/A')}x{stream.get('height','N/A')}"
                    break
        except:
            result["duration"]   = "N/A"
            result["size"]       = f"{os.path.getsize(path) / (1024 * 1024):.1f} MB"
            result["filename"]   = os.path.basename(path)
            result["resolution"] = "N/A"

        result["thumbnail"] = thumbnail
        self.progress.emit(100, "Done!")
        self.finished.emit(result)

    # ---------------------------------------------------------------- #
    #  YouTube                                                           #
    # ---------------------------------------------------------------- #
    def _youtube(self):
        url   = self.kw["url"]
        model = self.kw["model"]
        lang  = self.kw.get("lang", "en")
        vid   = self._vid_id(url)

        if not vid:
            self.error.emit("Invalid YouTube URL!")
            return

        self.progress.emit(20, "Fetching transcript…")
        transcript = ""
        ts         = []

        if YT_OK:
            # Method 1 — new API style (v0.9+)
            try:
                fetched = YouTubeTranscriptApi().fetch(vid, languages=[lang, "hi", "en"])
                transcript, ts = self._parse_transcript(fetched)
            except Exception:
                pass

            # Method 2 — old static API (v0.6–v0.8)
            if not transcript.strip():
                try:
                    fetched = YouTubeTranscriptApi.get_transcript(vid, languages=[lang, "hi", "en"])
                    transcript, ts = self._parse_transcript(fetched)
                except Exception:
                    pass

            # Method 3 — list_transcripts fallback
            if not transcript.strip():
                try:
                    tlist = YouTubeTranscriptApi.list_transcripts(vid)
                    t_obj = None
                    for candidate_langs in [[lang, "hi", "en", "en-US", "en-GB"],
                                            [lang, "hi", "en", "en-US", "en-GB"]]:
                        try:
                            t_obj = tlist.find_manually_created_transcript(candidate_langs)
                            break
                        except:
                            pass
                        try:
                            t_obj = tlist.find_generated_transcript(candidate_langs)
                            break
                        except:
                            pass
                    if t_obj is None:
                        for t_obj in tlist:
                            break
                    if t_obj:
                        fetched = t_obj.fetch()
                        transcript, ts = self._parse_transcript(fetched)
                except Exception:
                    pass

        if not transcript.strip():
            self.error.emit(
                "Could not fetch transcript.\n\n"
                "Possible reasons & solutions:\n"
                "1️⃣  Run:  pip install --upgrade youtube-transcript-api\n"
                "2️⃣  Video mein subtitles/captions enabled nahi hain\n"
                "3️⃣  Aapka IP YouTube se temporarily rate-limited hai\n"
                "4️⃣  Koi aur video try karein"
            )
            return

        self.progress.emit(60, "AI analyzing…")
        result = self._ollama(transcript, model, ts, lang)

        thumb_path = self._download_yt_thumb(vid)

        result["filename"]   = f"YouTube: {vid}"
        result["duration"]   = "N/A"
        result["size"]       = "Streaming"
        result["url"]        = url
        result["resolution"] = "N/A"
        result["thumbnail"]  = thumb_path or f"https://img.youtube.com/vi/{vid}/0.jpg"

        self.progress.emit(100, "Done!")
        self.finished.emit(result)

    def _parse_transcript(self, fetched):
        transcript = ""
        ts = []
        for e in fetched:
            if hasattr(e, 'text') and hasattr(e, 'start'):
                transcript += e.text + " "
                ts.append((e.start, e.text.strip()))
            elif isinstance(e, dict):
                transcript += e.get("text", "") + " "
                ts.append((e.get("start", 0), e.get("text", "").strip()))
            elif isinstance(e, (list, tuple)) and len(e) >= 2:
                transcript += str(e[1]) + " "
                ts.append((e[0], str(e[1]).strip()))
        return transcript, ts

    def _download_yt_thumb(self, vid):
        try:
            url  = f"https://img.youtube.com/vi/{vid}/hqdefault.jpg"
            path = os.path.join(os.path.expanduser("~"), f"yt_thumb_{vid}.jpg")
            urllib.request.urlretrieve(url, path)
            return path
        except:
            return None

    def _extract_thumbnail(self, path):
        try:
            thumb = path.rsplit(".", 1)[0] + "_thumb.jpg"
            os.system(
                f'ffmpeg -y -i "{path}" -vf "select=eq(n\\,0)" -q:v 3 "{thumb}" -loglevel quiet 2>/dev/null'
            )
            if os.path.exists(thumb):
                return thumb
        except:
            pass
        return None

    # ---------------------------------------------------------------- #
    #  JSON cleaner                                                      #
    # ---------------------------------------------------------------- #
    def _clean_json(self, raw):
        text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
        text = re.sub(r"```(?:json)?", "", text)
        text = text.replace("```", "")
        text = text.strip()
        text = re.sub(r'^[^{\[]+', '', text)

        for start_char, end_char in [('{', '}'), ('[', ']')]:
            idx = text.find(start_char)
            if idx == -1:
                continue
            depth  = 0
            in_str = False
            escape = False
            for i, ch in enumerate(text[idx:], start=idx):
                if escape:
                    escape = False
                    continue
                if ch == '\\' and in_str:
                    escape = True
                    continue
                if ch == '"' and not escape:
                    in_str = not in_str
                    continue
                if in_str:
                    continue
                if ch == start_char:
                    depth += 1
                elif ch == end_char:
                    depth -= 1
                    if depth == 0:
                        candidate = text[idx: i + 1]
                        try:
                            json.loads(candidate)
                            return candidate
                        except:
                            break

        return text.strip()

    def _parse_chapters_markdown(self, raw):
        chapters = []
        blocks = re.split(
            r"\n(?=\*\*Chapter\s+\d+|##\s*Chapter\s+\d+|Chapter\s+\d+\s*:)",
            raw,
            flags=re.IGNORECASE
        )
        for block in blocks:
            if not block.strip():
                continue
            title_match = re.search(
                r"(?:\*\*|##\s*)?Chapter\s+\d+[:\-–]?\s*([^\n*#]+?)(?:\*\*)?(?:\n|$)",
                block, re.IGNORECASE
            )
            title = title_match.group(1).strip() if title_match else ""
            if not title:
                num_match = re.search(r"Chapter\s+(\d+)", block, re.IGNORECASE)
                title = f"Chapter {num_match.group(1)}" if num_match else "Chapter"
            time_match = re.search(
                r"(?:start\s*time|start)[:\s*]*\**\s*([\d]+:[\d]{2})",
                block, re.IGNORECASE
            )
            if not time_match:
                time_match = re.search(r"\b(\d{1,2}:\d{2}(?::\d{2})?)\b", block)
            start = time_match.group(1) if time_match else "0:00"
            content_match = re.search(
                r"(?:content|summary|description)[:\s*]+\**\s*(.+?)(?=\n\*\*|\n##|\n-\s|\Z)",
                block, re.IGNORECASE | re.DOTALL
            )
            summary = content_match.group(1).strip() if content_match else block.strip()
            summary = re.sub(r"\*{1,2}(.*?)\*{1,2}", r"\1", summary)
            summary = re.sub(r"[#`_]", "", summary).strip()
            if title:
                chapters.append({
                    "title": title,
                    "start": start,
                    "summary": summary[:400],
                    "description": summary[:400],
                })
        return chapters

    # ---------------------------------------------------------------- #
    #  Ollama AI                                                         #
    # ---------------------------------------------------------------- #
    def _ollama(self, transcript, model, ts, lang="en"):
        sys_msg = (
            "You are a JSON-only response bot. "
            "STRICT RULES: "
            "1. Output ONLY raw JSON. "
            "2. No markdown, no ```json, no explanations, no greetings. "
            "3. No text before or after JSON. "
            "4. First character must be { or [. "
            "5. Last character must be } or ]. "
            "If you cannot follow these rules, output: {}"
        )
        quiz_count  = self.kw.get("quiz_count", 5)
        trans_short = transcript[:4000]
        results     = {}

        # ---------- Summary ----------
        def get_summary():
            try:
                self.progress.emit(30, "📝 Summary generating…")
                r = ollama.chat(model=model, messages=[
                    {"role": "system", "content": sys_msg},
                    {"role": "user",   "content": f"""Return this exact JSON structure filled with content:
{{
  "title": "video title max 10 words",
  "short_summary": "2-3 sentence summary",
  "detailed_summary": "5-7 sentence detailed summary",
  "key_points": ["point1","point2","point3","point4","point5"],
  "topics": ["topic1","topic2","topic3"],
  "action_items": ["action1","action2"]
}}
TRANSCRIPT: {trans_short}"""}
                ])["message"]["content"]
                cleaned = self._clean_json(r)
                results["summary"] = json.loads(cleaned)
            except Exception:
                results["summary"] = {
                    "title": "Summary unavailable",
                    "short_summary": "",
                    "detailed_summary": "",
                    "key_points": [],
                    "topics": [],
                    "action_items": []
                }

        # ---------- Quiz ----------
        def get_quiz():
            try:
                self.progress.emit(55, f"❓ Generating {quiz_count} quiz questions…")
                r = ollama.chat(model=model, messages=[
                    {"role": "system", "content": sys_msg},
                    {"role": "user",   "content": f"""Create EXACTLY {quiz_count} multiple choice questions.
Output a raw JSON array only. No text before or after. No markdown.
Each item must follow this structure exactly:
{{"question":"question text","options":["A) option","B) option","C) option","D) option"],"answer":0,"explanation":"why this answer"}}
The "answer" field is the 0-based index of the correct option (0=A, 1=B, 2=C, 3=D).
Example of valid output:
[{{"question":"What is X?","options":["A) foo","B) bar","C) baz","D) qux"],"answer":1,"explanation":"Because bar is correct."}}]
TRANSCRIPT: {trans_short}"""}
                ])["message"]["content"]

                cleaned = self._clean_json(r)
                parsed = None

                try:
                    parsed = json.loads(cleaned)
                except json.JSONDecodeError:
                    pass

                if parsed is None:
                    arr_match = re.search(r'\[.*\]', cleaned, re.DOTALL)
                    if arr_match:
                        try:
                            parsed = json.loads(arr_match.group())
                        except json.JSONDecodeError:
                            pass

                if parsed is None:
                    results["quiz"] = []
                    return

                if isinstance(parsed, dict):
                    for key in ("questions", "quiz", "mcq", "items", "data"):
                        if key in parsed and isinstance(parsed[key], list):
                            parsed = parsed[key]
                            break
                    else:
                        for v in parsed.values():
                            if isinstance(v, list):
                                parsed = v
                                break

                if not isinstance(parsed, list):
                    results["quiz"] = []
                    return

                clean_quiz = []
                for item in parsed:
                    if not isinstance(item, dict):
                        continue
                    question    = item.get("question", "").strip()
                    options     = item.get("options", [])
                    answer      = item.get("answer", 0)
                    explanation = item.get("explanation", "")

                    if not question or not options:
                        continue

                    try:
                        answer = int(answer)
                    except (ValueError, TypeError):
                        if isinstance(answer, str) and answer.strip().upper() in ("A","B","C","D"):
                            answer = ord(answer.strip().upper()) - ord('A')
                        elif isinstance(answer, str) and answer.strip() and answer.strip()[0].upper() in ("A","B","C","D"):
                            answer = ord(answer.strip()[0].upper()) - ord('A')
                        else:
                            answer = 0

                    answer = max(0, min(answer, len(options) - 1))

                    clean_quiz.append({
                        "question":    question,
                        "options":     options,
                        "answer":      answer,
                        "explanation": explanation,
                    })

                results["quiz"] = clean_quiz

            except Exception:
                results["quiz"] = []

        # ---------- Chapters ----------
        def get_chapters():
            try:
                self.progress.emit(75, "📖 Detecting chapters…")

                prompt = (
                    'Output ONLY a JSON object. No explanation, no markdown, no think tags.\n'
                    'Rules:\n'
                    '1. Create exactly 5 chapters covering the full video.\n'
                    '2. Each "summary" and "description" MUST be a non-empty sentence (min 10 words).\n'
                    '3. Start times must be in "M:SS" format.\n'
                    'Format: {"chapters":[{"title":"...","start":"0:00","summary":"2-3 sentence summary of this section","description":"what happens in this section"}]}\n'
                    f'TRANSCRIPT: {trans_short}'
                )

                r = ollama.chat(
                    model=model,
                    messages=[
                        {"role": "system", "content": sys_msg},
                        {"role": "user", "content": prompt}
                    ]
                )["message"]["content"]

                raw_chapters = []

                try:
                    cleaned = self._clean_json(r)
                    ch_data = json.loads(cleaned)

                    if isinstance(ch_data, list):
                        raw_chapters = ch_data
                    elif isinstance(ch_data, dict):
                        raw_chapters = ch_data.get("chapters", [])
                    else:
                        raw_chapters = []

                except Exception:
                    raw_chapters = self._parse_chapters_markdown(r)

                fixed = []
                for i, ch in enumerate(raw_chapters):
                    # Skip anything that is not a dict
                    if not isinstance(ch, dict):
                        continue

                    raw_start = ch.get("start", ch.get("start_time", ch.get("time", "0:00")))

                    if isinstance(raw_start, (int, float)):
                        secs = int(raw_start)
                        start_str = f"{secs // 60}:{secs % 60:02d}"
                    else:
                        start_str = re.sub(r"[*_`]", "", str(raw_start)).strip()

                    summary     = ch.get("summary", ch.get("description", ""))
                    description = ch.get("description", ch.get("summary", ""))

                    if not summary or summary.strip() in ["", "No summary.", "No description."]:
                        summary = "This chapter covers content from the video."
                    if not description or description.strip() in ["", "No summary.", "No description."]:
                        description = summary

                    fixed.append({
                        "title":       re.sub(r"[*_`#]", "", ch.get("title", f"Chapter {i+1}")).strip(),
                        "start":       start_str,
                        "summary":     summary,
                        "description": description,
                    })

                results["chapters"] = fixed if fixed else [{
                    "title":       "Full Video",
                    "start":       "0:00",
                    "summary":     "Chapter generation failed. Please retry with a different model.",
                    "description": "AI could not generate chapters for this content."
                }]

            except Exception:
                results["chapters"] = [{
                    "title":       "Full Video",
                    "start":       "0:00",
                    "summary":     "Chapter generation failed.",
                    "description": "AI could not generate chapters for this content."
                }]

        # Run all three in parallel
        self.progress.emit(25, "🚀 AI analysis starting…")
        t1 = threading.Thread(target=get_summary,  daemon=True)
        t2 = threading.Thread(target=get_quiz,     daemon=True)
        t3 = threading.Thread(target=get_chapters, daemon=True)
        t1.start(); t2.start(); t3.start()
        t1.join()
        t2.join(timeout=300)
        t3.join(timeout=300)

        data             = results.get("summary", {})
        data["quiz"]     = results.get("quiz",     [])
        data["chapters"] = results.get("chapters", [])

        step       = max(1, len(ts) // 5)
        timestamps = [
            {"time": str(timedelta(seconds=int(t[0]))), "text": t[1]}
            for t in ts[::step]
        ][:5]

        self.progress.emit(90, "✨ Almost done…")
        return {**data, "timestamps": timestamps, "transcript": transcript}

    def _vid_id(self, url):
        for p in [r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", r"^([A-Za-z0-9_-]{11})$"]:
            m = re.search(p, url)
            if m:
                return m.group(1)
        return None


# ------------------------------------------------------------------ #
#  Helper factories                                                    #
# ------------------------------------------------------------------ #

def make_card(layout_type="V", margin=16, spacing=12):
    frame = QFrame()
    frame.setObjectName("card")
    lay = QVBoxLayout(frame) if layout_type == "V" else QHBoxLayout(frame)
    lay.setContentsMargins(margin, margin, margin, margin)
    lay.setSpacing(spacing)
    return frame, lay

def label(text, obj="", wrap=False):
    l = QLabel(text)
    if obj:  l.setObjectName(obj)
    if wrap: l.setWordWrap(True)
    return l

def btn(text, obj="primaryBtn"):
    b = QPushButton(text)
    b.setObjectName(obj)
    b.setCursor(QCursor(Qt.PointingHandCursor))
    return b


# ------------------------------------------------------------------ #
#  Drag-drop upload zone                                               #
# ------------------------------------------------------------------ #

class DragDropWidget(QFrame):
    file_dropped = pyqtSignal(str)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.setDropAction(Qt.CopyAction)
            event.accept()
            self.setProperty("drag-over", True)
            self.style().unpolish(self)
            self.style().polish(self)

    def dragLeaveEvent(self, event):
        self.setProperty("drag-over", False)
        self.style().unpolish(self)
        self.style().polish(self)

    def dropEvent(self, event):
        self.setProperty("drag-over", False)
        self.style().unpolish(self)
        self.style().polish(self)
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path.lower().endswith(('.mp4', '.mkv', '.avi', '.mov')):
                self.file_dropped.emit(path)
                break


# ------------------------------------------------------------------ #
#  Main Window                                                         #
# ------------------------------------------------------------------ #

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎬 AI Video Summarizer")
        self.setMinimumSize(1050, 680)
        self.result  = {}
        self.cur_q   = 0
        self.answers = []
        self._build()
        self.setStyleSheet(STYLE)

    def _build(self):
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        main = QHBoxLayout(root)
        main.setContentsMargins(0, 0, 0, 0)
        main.setSpacing(0)

        # Sidebar
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sv = QVBoxLayout(sidebar)
        sv.setContentsMargins(12, 12, 12, 12)
        sv.setSpacing(4)
        sv.addWidget(label("🎬 AI Summarizer", "appName"))

        self.nav_btns = []
        pages = [
            ("📁  Upload Video", 0),
            ("🎥  YouTube Link", 1),
            ("📄  Summary",      2),
            ("📌  Key Points",   3),
            ("📖  Chapters",     4),
            ("❓  Quiz",         5),
            ("⚙️  Settings",     6),
        ]
        for txt, idx in pages:
            b = btn(txt, "navBtn")
            b.clicked.connect(lambda _, i=idx: self._goto(i))
            sv.addWidget(b)
            self.nav_btns.append(b)

        sv.addStretch()
        main.addWidget(sidebar)

        self.stack = QStackedWidget()
        self.stack.addWidget(self._page_upload())
        self.stack.addWidget(self._page_youtube())
        self.stack.addWidget(self._page_summary())
        self.stack.addWidget(self._page_keypoints())
        self.stack.addWidget(self._page_chapters())
        self.stack.addWidget(self._page_quiz())
        self.stack.addWidget(self._page_settings())
        main.addWidget(self.stack, 1)

        self._goto(0)

    def _goto(self, idx):
        self.stack.setCurrentIndex(idx)
        for i, b in enumerate(self.nav_btns):
            b.setProperty("active", str(i == idx).lower())
            b.style().unpolish(b)
            b.style().polish(b)

    def _scroll_page(self, inner):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(inner)
        return scroll

    # ---------------------------------------------------------------- #
    #  Page: Upload Video                                                #
    # ---------------------------------------------------------------- #
    def _page_upload(self):
        inner = QWidget()
        v = QVBoxLayout(inner)
        v.setContentsMargins(28, 24, 28, 24)
        v.setSpacing(16)

        v.addWidget(label("📁  Upload Video", "pageTitle"))
        v.addWidget(label("Supports MP4, MKV, AVI, MOV – Drag & Drop or Browse", "metaLabel"))

        zone = DragDropWidget()
        zone.setObjectName("uploadZone")
        zone.setAcceptDrops(True)
        zv = QVBoxLayout(zone)
        zv.setAlignment(Qt.AlignCenter)
        zv.setSpacing(10)
        self.up_icon = label("🎬", "")
        self.up_icon.setAlignment(Qt.AlignCenter)
        self.up_icon.setStyleSheet("font-size:48px;")
        self.up_name = label("Drag & drop video here or click Browse", "metaLabel")
        self.up_name.setAlignment(Qt.AlignCenter)
        up_browse = btn("  Browse File", "primaryBtn")
        up_browse.setFixedWidth(160)
        up_browse.clicked.connect(self._pick_file)
        zone.file_dropped.connect(self._file_dropped)
        zv.addWidget(self.up_icon)
        zv.addWidget(self.up_name)
        zv.addWidget(up_browse)
        v.addWidget(zone)

        thumb_card, tcl = make_card("H")
        tcl.addWidget(label("🎬 Thumbnail:", "sectionTitle"))
        self.thumb_label = QLabel("—")
        self.thumb_label.setAlignment(Qt.AlignCenter)
        self.thumb_label.setStyleSheet("color:#6666aa; font-size:12px;")
        self.thumb_label.setMaximumHeight(120)
        tcl.addWidget(self.thumb_label)
        v.addWidget(thumb_card)

        info, il = make_card("H")
        self.fi_name = self._info_tile("📹 File Name", "—")
        self.fi_dur  = self._info_tile("⏱ Duration",   "—")
        self.fi_size = self._info_tile("💾 Size",       "—")
        self.fi_res  = self._info_tile("🖥️ Resolution", "—")
        il.addWidget(self.fi_name)
        il.addWidget(self.fi_dur)
        il.addWidget(self.fi_size)
        il.addWidget(self.fi_res)
        v.addWidget(info)

        self.up_prog   = QProgressBar()
        self.up_prog.setValue(0)
        self.up_status = label("", "statusLabel")
        v.addWidget(self.up_prog)
        v.addWidget(self.up_status)

        self.up_btn = btn("🚀  Analyze Video", "primaryBtn")
        self.up_btn.setEnabled(False)
        self.up_btn.clicked.connect(self._run_video)
        v.addWidget(self.up_btn)
        v.addStretch()
        self.video_path = ""
        return self._scroll_page(inner)

    def _file_dropped(self, path):
        self.video_path = path
        self.up_name.setText(f"Selected: {os.path.basename(path)}")
        self.fi_name._val_label.setText(os.path.basename(path))
        self.fi_size._val_label.setText(f"{os.path.getsize(path) / (1024 * 1024):.1f} MB")
        self.fi_dur._val_label.setText("Calculating…")
        self.fi_res._val_label.setText("Calculating…")
        self.up_btn.setEnabled(True)

    def _info_tile(self, title, value):
        frame, fl = make_card("V", 12, 4)
        frame.setObjectName("infoCard")
        fl.addWidget(label(title, "metaLabel"))
        val = label(value, "valueLabel")
        val.setWordWrap(True)
        fl.addWidget(val)
        frame._val_label = val
        return frame

    # ---------------------------------------------------------------- #
    #  Page: YouTube Link                                                #
    # ---------------------------------------------------------------- #
    def _page_youtube(self):
        inner = QWidget()
        v = QVBoxLayout(inner)
        v.setContentsMargins(28, 24, 28, 24)
        v.setSpacing(16)

        v.addWidget(label("🎥  YouTube Link", "pageTitle"))
        v.addWidget(label("Paste any YouTube video URL below", "metaLabel"))

        card, cl = make_card()
        cl.addWidget(label("YouTube URL", "sectionTitle"))
        self.yt_url = QLineEdit()
        self.yt_url.setPlaceholderText("https://www.youtube.com/watch?v=...")
        cl.addWidget(self.yt_url)

        mr = QHBoxLayout()
        mr.addWidget(label("Model:", "metaLabel"))
        self.yt_model = QComboBox()
        self.yt_model.addItems(["llama3", "llama3.2:1b", "mistral", "gemma3", "phi3", "qwen3:8b"])
        mr.addWidget(self.yt_model)
        mr.addStretch()
        cl.addLayout(mr)

        lr = QHBoxLayout()
        lr.addWidget(label("Language:", "metaLabel"))
        self.yt_lang = QComboBox()
        self.yt_lang.addItems(["auto — Auto detect", "en — English", "hi — Hindi"])
        lr.addWidget(self.yt_lang)
        lr.addStretch()
        cl.addLayout(lr)

        self.yt_prog   = QProgressBar()
        self.yt_prog.setValue(0)
        self.yt_status = label("", "statusLabel")
        cl.addWidget(self.yt_prog)
        cl.addWidget(self.yt_status)

        yt_go = btn("▶️  Fetch & Analyze", "primaryBtn")
        yt_go.clicked.connect(self._run_youtube)
        cl.addWidget(yt_go)
        v.addWidget(card)

        tip, tl = make_card()
        tl.addWidget(label("💡  Tips", "sectionTitle"))
        tips = [
            "✅  Works best with videos that have subtitles/captions",
            "✅  Educational, news, and tutorial videos work great",
            "⚠️  Music videos or no-speech videos may not work",
            "🔧  If error: run  pip install --upgrade youtube-transcript-api",
        ]
        for t in tips:
            tl.addWidget(label(t, "metaLabel"))
        v.addWidget(tip)
        v.addStretch()
        return self._scroll_page(inner)

    # ---------------------------------------------------------------- #
    #  Page: Summary                                                     #
    # ---------------------------------------------------------------- #
    def _page_summary(self):
        inner = QWidget()
        v = QVBoxLayout(inner)
        v.setContentsMargins(28, 24, 28, 24)
        v.setSpacing(16)

        v.addWidget(label("📄  Summary", "pageTitle"))
        self.sum_title = label("No video analyzed yet.", "sectionTitle")
        self.sum_title.setWordWrap(True)
        v.addWidget(self.sum_title)

        info_row = QHBoxLayout()
        self.yt_thumb = QLabel()
        self.yt_thumb.setFixedSize(160, 100)
        self.yt_thumb.setAlignment(Qt.AlignCenter)
        self.yt_thumb.setStyleSheet(
            "border: 1px solid #2a2a50; border-radius: 8px; color:#6666aa; font-size:11px;"
        )
        self.yt_thumb.setText("No thumbnail")
        info_row.addWidget(self.yt_thumb)

        info, il = make_card("H")
        self.s_name = self._info_tile("📹 File Name", "—")
        self.s_dur  = self._info_tile("⏱ Duration",   "—")
        self.s_size = self._info_tile("💾 Size",       "—")
        self.s_res  = self._info_tile("🖥️ Resolution", "—")
        il.addWidget(self.s_name)
        il.addWidget(self.s_dur)
        il.addWidget(self.s_size)
        il.addWidget(self.s_res)
        info_row.addWidget(info)
        info_row.addStretch()
        v.addLayout(info_row)

        c1, l1 = make_card()
        l1.addWidget(label("✏️  Short Summary", "sectionTitle"))
        self.short_box = QTextEdit()
        self.short_box.setReadOnly(True)
        self.short_box.setMaximumHeight(90)
        l1.addWidget(self.short_box)
        v.addWidget(c1)

        c2, l2 = make_card()
        l2.addWidget(label("📝  Detailed Summary", "sectionTitle"))
        self.detail_box = QTextEdit()
        self.detail_box.setReadOnly(True)
        self.detail_box.setMaximumHeight(150)
        l2.addWidget(self.detail_box)
        v.addWidget(c2)

        row = QHBoxLayout()
        b_copy = btn("📋  Copy",     "secondaryBtn"); b_copy.clicked.connect(self._copy)
        b_txt  = btn("💾  Save TXT", "secondaryBtn"); b_txt.clicked.connect(self._save_txt)
        b_pdf  = btn("📄  Save PDF", "successBtn");   b_pdf.clicked.connect(self._save_pdf)
        row.addWidget(b_copy)
        row.addWidget(b_txt)
        row.addWidget(b_pdf)
        row.addStretch()
        v.addLayout(row)
        v.addStretch()
        return self._scroll_page(inner)

    # ---------------------------------------------------------------- #
    #  Page: Key Points                                                  #
    # ---------------------------------------------------------------- #
    def _page_keypoints(self):
        inner = QWidget()
        v = QVBoxLayout(inner)
        v.setContentsMargins(28, 24, 28, 24)
        v.setSpacing(16)

        v.addWidget(label("📌  Key Points", "pageTitle"))

        c1, l1 = make_card()
        l1.addWidget(label("🎯  Key Points", "sectionTitle"))
        self.kp_box = QTextEdit()
        self.kp_box.setReadOnly(True)
        l1.addWidget(self.kp_box)
        v.addWidget(c1)

        c2, l2 = make_card()
        l2.addWidget(label("🏷️  Topics Covered", "sectionTitle"))
        self.topics_box = QTextEdit()
        self.topics_box.setReadOnly(True)
        self.topics_box.setMaximumHeight(80)
        l2.addWidget(self.topics_box)
        v.addWidget(c2)

        c3, l3 = make_card()
        l3.addWidget(label("✅  Action Items", "sectionTitle"))
        self.action_box = QTextEdit()
        self.action_box.setReadOnly(True)
        self.action_box.setMaximumHeight(100)
        l3.addWidget(self.action_box)
        v.addWidget(c3)

        c4, l4 = make_card()
        l4.addWidget(label("⏱️  Important Timestamps", "sectionTitle"))
        self.ts_box = QTextEdit()
        self.ts_box.setReadOnly(True)
        self.ts_box.setMaximumHeight(130)
        l4.addWidget(self.ts_box)
        v.addWidget(c4)
        v.addStretch()
        return self._scroll_page(inner)

    # ---------------------------------------------------------------- #
    #  Page: Chapters                                                    #
    # ---------------------------------------------------------------- #
    def _page_chapters(self):
        inner = QWidget()
        v = QVBoxLayout(inner)
        v.setContentsMargins(28, 24, 28, 24)
        v.setSpacing(16)

        v.addWidget(label("📖  Chapters", "pageTitle"))
        v.addWidget(label("Chapter-wise breakdown with summaries", "metaLabel"))

        c1, l1 = make_card()
        l1.addWidget(label("📚  Chapter List", "sectionTitle"))
        self.chapters_list = QListWidget()
        self.chapters_list.setMinimumHeight(160)
        l1.addWidget(self.chapters_list)
        v.addWidget(c1)

        c2, l2 = make_card()
        l2.addWidget(label("📝  Chapter Details", "sectionTitle"))
        self.chapter_details = QTextEdit()
        self.chapter_details.setReadOnly(True)
        self.chapter_details.setMinimumHeight(120)
        l2.addWidget(self.chapter_details)
        v.addWidget(c2)

        row = QHBoxLayout()
        b_copy = btn("📋  Copy Chapter", "secondaryBtn")
        b_copy.clicked.connect(self._copy_chapter)
        b_all  = btn("📋  Copy All",     "secondaryBtn")
        b_all.clicked.connect(self._copy_all_chapters)
        row.addWidget(b_copy)
        row.addWidget(b_all)
        row.addStretch()
        v.addLayout(row)
        v.addStretch()
        return self._scroll_page(inner)

    def _copy_chapter(self):
        item = self.chapters_list.currentItem()
        if item:
            QApplication.clipboard().setText(
                item.text() + "\n" + self.chapter_details.toPlainText()
            )
            QMessageBox.information(self, "Copied!", "Chapter copied to clipboard! 📋")

    def _copy_all_chapters(self):
        lines = []
        for i in range(self.chapters_list.count()):
            lines.append(self.chapters_list.item(i).text())
        QApplication.clipboard().setText("\n\n".join(lines))
        QMessageBox.information(self, "Copied!", "All chapters copied! 📋")

    # ---------------------------------------------------------------- #
    #  Page: Quiz                                                        #
    # ---------------------------------------------------------------- #
    def _page_quiz(self):
        inner = QWidget()
        v = QVBoxLayout(inner)
        v.setContentsMargins(28, 24, 28, 24)
        v.setSpacing(16)

        v.addWidget(label("❓  Quiz", "pageTitle"))

        qcard, ql = make_card()
        self.q_progress = label("First analysis video!", "metaLabel")
        ql.addWidget(self.q_progress)
        self.q_text = label("", "sectionTitle")
        self.q_text.setWordWrap(True)
        self.q_text.setStyleSheet("font-size:15px; color:#e0e0f0; padding:4px 0;")
        ql.addWidget(self.q_text)
        v.addWidget(qcard)

        self.opt_btns = []
        for i in range(4):
            b = btn("", "optBtn")
            b.clicked.connect(lambda _, x=i: self._answer(x))
            b.setVisible(False)
            v.addWidget(b)
            self.opt_btns.append(b)

        self.q_feedback = label("", "")
        self.q_feedback.setWordWrap(True)
        self.q_feedback.setStyleSheet("font-size:13px; padding:10px; border-radius:10px;")
        v.addWidget(self.q_feedback)

        row = QHBoxLayout()
        self.btn_next    = btn("Next →",     "primaryBtn")
        self.btn_next.setVisible(False)
        self.btn_restart = btn("🔄 Restart", "secondaryBtn")
        self.btn_restart.setVisible(False)
        self.btn_next.clicked.connect(self._next_q)
        self.btn_restart.clicked.connect(self._start_quiz)
        row.addWidget(self.btn_next)
        row.addWidget(self.btn_restart)
        row.addStretch()
        v.addLayout(row)
        v.addStretch()
        return self._scroll_page(inner)

    # ---------------------------------------------------------------- #
    #  Page: Settings                                                    #
    # ---------------------------------------------------------------- #
    def _page_settings(self):
        inner = QWidget()
        v = QVBoxLayout(inner)
        v.setContentsMargins(28, 24, 28, 24)
        v.setSpacing(16)

        v.addWidget(label("⚙️  Settings", "pageTitle"))

        card, cl = make_card()
        cl.addWidget(label("🤖  Default Ollama Model", "sectionTitle"))
        self.settings_model = QComboBox()
        self.settings_model.addItems(
            ["llama3","mistral", "llama3.2:1b", "gemma3", "phi3", "qwen3:8b"]
        )
        cl.addWidget(self.settings_model)

        cl.addWidget(label("🌐  Default Language", "sectionTitle"))
        self.settings_lang = QComboBox()
        self.settings_lang.addItems(
            ["auto — Auto detect", "en — English", "hi — Hindi"]
        )
        cl.addWidget(self.settings_lang)

        cl.addWidget(label("📊  Quiz Questions", "sectionTitle"))
        qh = QHBoxLayout()
        self.settings_qcount = QSpinBox()
        self.settings_qcount.setMinimum(1)
        self.settings_qcount.setMaximum(10)
        self.settings_qcount.setValue(5)
        qh.addWidget(self.settings_qcount)
        qh.addWidget(label("(1–10 questions)", "metaLabel"))
        qh.addStretch()
        cl.addLayout(qh)
        v.addWidget(card)

        adv, al = make_card()
        al.addWidget(label("🔧  Advanced Options", "sectionTitle"))
        self.extract_chapters  = QCheckBox("Auto-detect chapters")
        self.extract_chapters.setChecked(True)
        self.enable_timestamps = QCheckBox("Include timestamps")
        self.enable_timestamps.setChecked(True)
        self.show_thumbnail    = QCheckBox("Show video thumbnail")
        self.show_thumbnail.setChecked(True)
        al.addWidget(self.extract_chapters)
        al.addWidget(self.enable_timestamps)
        al.addWidget(self.show_thumbnail)
        v.addWidget(adv)

        sc, sl = make_card()
        sl.addWidget(label("📦  Library Status", "sectionTitle"))
        libs = [
            ("faster-whisper (local speech)", WHISPER_OK),
            ("reportlab (PDF export)",        PDF_OK),
            ("yt-dlp (YouTube download)",     YTDLP_OK),
            ("youtube-transcript-api",        YT_OK),
        ]
        for name, ok in libs:
            row = QHBoxLayout()
            row.addWidget(label(("✅  " if ok else "❌  ") + name, "metaLabel"))
            if not ok:
                b_install = btn(f"pip install {name.split()[0]}", "secondaryBtn")
                b_install.setFixedHeight(28)
                row.addWidget(b_install)
            sl.addLayout(row)
        v.addWidget(sc)
        v.addStretch()
        return self._scroll_page(inner)

    # ---------------------------------------------------------------- #
    #  Actions                                                           #
    # ---------------------------------------------------------------- #

    def _pick_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "", "Video (*.mp4 *.mkv *.avi *.mov)"
        )
        if path:
            self._file_dropped(path)

    def _run_video(self):
        if not self.video_path:
            return
        self.up_btn.setEnabled(False)
        model      = self.settings_model.currentText()
        lang       = self.settings_lang.currentText().split(" — ")[0]
        quiz_count = self.settings_qcount.value()

        self.worker = Worker(
            "video", path=self.video_path, model=model,
            lang=lang, quiz_count=quiz_count
        )
        self.worker.progress.connect(
            lambda val, msg: (self.up_prog.setValue(val), self.up_status.setText(msg))
        )
        self.worker.finished.connect(self._done)
        self.worker.error.connect(self._err)
        self.worker.start()

    def _run_youtube(self):
        url = self.yt_url.text().strip()
        if not url:
            QMessageBox.warning(self, "Error", "Please paste a YouTube URL!")
            return
        model      = self.yt_model.currentText()
        lang       = self.yt_lang.currentText().split(" — ")[0]
        quiz_count = self.settings_qcount.value()

        self.worker = Worker(
            "youtube", url=url, model=model,
            lang=lang, quiz_count=quiz_count
        )
        self.worker.progress.connect(
            lambda val, msg: (self.yt_prog.setValue(val), self.yt_status.setText(msg))
        )
        self.worker.finished.connect(self._done)
        self.worker.error.connect(self._err)
        self.worker.start()

    def _done(self, data):
        self.result = data

        self.sum_title.setText(f"🎬  {data.get('title', 'Video Summary')}")
        self.short_box.setText(data.get("short_summary", ""))
        self.detail_box.setText(data.get("detailed_summary", ""))

        thumb = data.get("thumbnail")
        if thumb and self.show_thumbnail.isChecked():
            pix = None
            if isinstance(thumb, str) and os.path.exists(thumb):
                pix = QPixmap(thumb)
            if pix and not pix.isNull():
                scaled = pix.scaled(160, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.yt_thumb.setPixmap(scaled)
                self.yt_thumb.setText("")
                self.thumb_label.setPixmap(
                    pix.scaled(200, 120, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                )

        for tile_set in [
            (self.s_name,  self.s_dur,  self.s_size,  self.s_res),
            (self.fi_name, self.fi_dur, self.fi_size, self.fi_res),
        ]:
            tile_set[0]._val_label.setText(data.get("filename",   "—"))
            tile_set[1]._val_label.setText(data.get("duration",   "—"))
            tile_set[2]._val_label.setText(data.get("size",       "—"))
            tile_set[3]._val_label.setText(data.get("resolution", "—"))

        kp = data.get("key_points", [])
        self.kp_box.setText("\n\n".join(f"•  {p}" for p in kp))
        self.topics_box.setText("   ·   ".join(data.get("topics", [])))
        actions = data.get("action_items", [])
        self.action_box.setText("\n\n".join(f"✓  {a}" for a in actions))
        ts = data.get("timestamps", [])
        self.ts_box.setText("\n".join(f"[{t['time']}]   {t['text']}" for t in ts))

        self.chapters_list.clear()
        try:
            self.chapters_list.itemClicked.disconnect()
        except:
            pass

        chapters = data.get("chapters", [])
        if chapters:
            for ch in chapters:
                item = QListWidgetItem(
                    f"📌 {ch.get('title','Chapter')} ({ch.get('start','0:00')})"
                )
                item.setData(Qt.UserRole, json.dumps(ch))
                self.chapters_list.addItem(item)
            self.chapters_list.itemClicked.connect(self._show_chapter_details)
            self.chapters_list.setCurrentRow(0)
            self._show_chapter_details(self.chapters_list.item(0))
        else:
            self.chapters_list.addItem("No chapters detected.")

        self._start_quiz()
        self.up_btn.setEnabled(True)
        self._goto(2)
        QMessageBox.information(
            self, "✅ Done!",
            "Analysis complete!\nSummary, Key Points, Chapters aur Quiz tabs check karein."
        )

    def _show_chapter_details(self, item):
        try:
            ch   = json.loads(item.data(Qt.UserRole))
            text = (
                f"Title:   {ch.get('title','Chapter')}\n"
                f"Time:    {ch.get('start','0:00')}\n\n"
                f"Summary:\n{ch.get('summary', ch.get('description',''))}"
            )
            self.chapter_details.setText(text)
        except:
            pass

    def _err(self, msg):
        self.up_btn.setEnabled(True)
        QMessageBox.critical(self, "Error", msg)

    # ---------------------------------------------------------------- #
    #  Quiz logic                                                        #
    # ---------------------------------------------------------------- #

    def _start_quiz(self):
        self.cur_q   = 0
        self.answers = []
        self.btn_restart.setVisible(False)
        self._show_q()

    def _show_q(self):
        qs = self.result.get("quiz", [])
        if not qs:
            self.q_text.setText("No quiz yet. Pehle koi video analyze karein!")
            for b in self.opt_btns:
                b.setVisible(False)
            return
        if self.cur_q >= len(qs):
            self._score()
            return
        q = qs[self.cur_q]
        self.q_progress.setText(f"Question {self.cur_q + 1} of {len(qs)}")
        self.q_text.setText(q.get("question", ""))
        self.q_feedback.setText("")
        self.q_feedback.setStyleSheet("")
        letters = ["A", "B", "C", "D"]
        opts    = q.get("options", [])
        for i, b in enumerate(self.opt_btns):
            if i < len(opts):
                clean_opt = re.sub(r"^[ABCD][)\.\s]+", "", opts[i]).strip()
                b.setText(f"  {letters[i]})   {clean_opt}")
                b.setVisible(True)
                b.setEnabled(True)
                b.setStyleSheet("")
            else:
                b.setVisible(False)
        self.btn_next.setVisible(False)

    def _answer(self, idx):
        qs = self.result.get("quiz", [])
        if not qs or self.cur_q >= len(qs):
            return
        q       = qs[self.cur_q]
        correct = q.get("answer", 0)
        self.answers.append(idx == correct)
        for i, b in enumerate(self.opt_btns):
            b.setEnabled(False)
            if i == correct:
                b.setStyleSheet("background:#064e3b; color:#6ee7b7; border:1px solid #059669;")
            elif i == idx and idx != correct:
                b.setStyleSheet("background:#450a0a; color:#fca5a5; border:1px solid #dc2626;")
        if idx == correct:
            self.q_feedback.setStyleSheet(
                "background:#064e3b; color:#6ee7b7; border-radius:10px; padding:10px;"
            )
            self.q_feedback.setText(f"✅  Correct!   {q.get('explanation','')}")
        else:
            self.q_feedback.setStyleSheet(
                "background:#450a0a; color:#fca5a5; border-radius:10px; padding:10px;"
            )
            self.q_feedback.setText(f"❌  Wrong.   {q.get('explanation','')}")
        self.btn_next.setVisible(True)
        self.btn_next.setText(
            "Next Question →" if self.cur_q + 1 < len(qs) else "See Results 🏆"
        )

    def _next_q(self):
        self.cur_q += 1
        self._show_q()

    def _score(self):
        total   = len(self.answers)
        correct = sum(self.answers)
        pct     = round(correct / total * 100) if total else 0
        self.q_progress.setText("Quiz Complete!")
        emoji = "🎉" if pct == 100 else "👍" if pct >= 70 else "📚"
        self.q_text.setText(f"{emoji}  Score: {correct}/{total}  ({pct}%)")
        self.q_feedback.setStyleSheet(
            "background:#1e1b4b; color:#a78bfa; border-radius:10px; padding:10px;"
        )
        self.q_feedback.setText(
            "Excellent! Bahut achha kiya!" if pct >= 70
            else "Practice karo — summary dobara padho!"
        )
        for b in self.opt_btns:
            b.setVisible(False)
        self.btn_next.setVisible(False)
        self.btn_restart.setVisible(True)

    # ---------------------------------------------------------------- #
    #  Export                                                            #
    # ---------------------------------------------------------------- #

    def _summary_text(self):
        d    = self.result
        text = "\n\n".join([
            f"TITLE: {d.get('title','')}",
            f"SHORT SUMMARY:\n{d.get('short_summary','')}",
            f"DETAILED SUMMARY:\n{d.get('detailed_summary','')}",
            "KEY POINTS:\n"   + "\n".join(f"• {p}" for p in d.get("key_points",   [])),
            "ACTION ITEMS:\n" + "\n".join(f"✓ {a}" for a in d.get("action_items", [])),
            "TOPICS:\n"       + ", ".join(d.get("topics", [])),
            "TIMESTAMPS:\n"   + "\n".join(
                f"[{t['time']}] {t['text']}" for t in d.get("timestamps", [])
            ),
        ])
        chapters = d.get("chapters", [])
        if chapters:
            text += "\n\nCHAPTERS:\n" + "\n".join(
                f"• {ch.get('title','Chapter')} ({ch.get('start','0:00')}): "
                f"{ch.get('summary', ch.get('description',''))}"
                for ch in chapters
            )
        return text

    def _copy(self):
        QApplication.clipboard().setText(self._summary_text())
        QMessageBox.information(self, "Copied!", "Summary clipboard mein copy ho gayi! 📋")

    def _save_txt(self):
        p, _ = QFileDialog.getSaveFileName(self, "Save TXT", "summary.txt", "Text (*.txt)")
        if p:
            with open(p, "w", encoding="utf-8") as f:
                f.write(self._summary_text())
            QMessageBox.information(self, "Saved!", f"Saved to:\n{p}")

    def _save_pdf(self):
        if not PDF_OK:
            QMessageBox.warning(self, "Error", "Run: pip install reportlab")
            return
        p, _ = QFileDialog.getSaveFileName(self, "Save PDF", "summary.pdf", "PDF (*.pdf)")
        if not p:
            return

        c    = pdf_canvas.Canvas(p, pagesize=A4)
        W, H = A4
        c.setFont("Helvetica-Bold", 16)
        c.setFillColorRGB(0.47, 0.22, 0.93)
        c.drawString(40, H - 50, self.result.get("title", "Video Summary")[:80])

        c.setFont("Helvetica", 11)
        c.setFillColorRGB(0.1, 0.1, 0.1)
        y = H - 80
        for line in self._summary_text().split("\n"):
            if y < 60:
                c.showPage()
                y = H - 40
                c.setFont("Helvetica", 11)
            c.drawString(40, y, line[:100])
            y -= 16

        c.save()
        QMessageBox.information(self, "Saved!", f"PDF saved:\n{p}")


# ------------------------------------------------------------------ #
#  Entry point                                                         #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = App()
    w.show()
    sys.exit(app.exec_())
