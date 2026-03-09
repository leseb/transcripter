package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"html/template"
	"io"
	"log"
	"mime/multipart"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

const (
	maxChunkSeconds  = 600 // 10 minutes per chunk
	maxFileSizeBytes = 25 * 1024 * 1024
	defaultAddr      = ":8080"
	transcriptsDir   = "transcripts"
	metadataFile     = "transcripts/metadata.json"
)

// Progress tracks transcription progress for a session.
type Progress struct {
	Status       string `json:"status"`
	ProgressPct  int    `json:"progress"`
	Message      string `json:"message"`
	CurrentChunk int    `json:"current_chunk"`
	TotalChunks  int    `json:"total_chunks"`
}

// TranscriptMeta holds metadata for a saved transcript.
type TranscriptMeta struct {
	Filename         string `json:"filename"`
	OriginalFilename string `json:"original_filename"`
	CreatedAt        string `json:"created_at"`
	Duration         string `json:"duration"`
	ProcessingTime   string `json:"processing_time"`
	TranscriptLength int    `json:"transcript_length"`
	FilePath         string `json:"file_path"`
}

// TranscriptView is used for template rendering.
type TranscriptView struct {
	ID               string
	Filename         string
	OriginalFilename string
	CreatedAt        string
	Duration         string
	ProcessingTime   string
	TranscriptLength int
}

// PageData is passed to the HTML template.
type PageData struct {
	Transcript          string
	TranscriptFilename  string
	OriginalFilename    string
	ProcessingTime      string
	AudioDuration       string
	AvailableTranscripts []TranscriptView
	SessionID           string
	TranscriptChars     int
	TranscriptWords     int
}

var (
	progressMu sync.RWMutex
	progressMap = make(map[string]*Progress)
	tmpl        *template.Template
)

func main() {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Fatal("OPENAI_API_KEY environment variable is required")
	}

	// Check ffmpeg/ffprobe availability
	if _, err := exec.LookPath("ffmpeg"); err != nil {
		log.Fatal("ffmpeg is required but not found in PATH")
	}
	if _, err := exec.LookPath("ffprobe"); err != nil {
		log.Fatal("ffprobe is required but not found in PATH")
	}

	// Create transcripts directory
	if err := os.MkdirAll(transcriptsDir, 0755); err != nil {
		log.Fatalf("Failed to create transcripts directory: %v", err)
	}

	// Parse templates
	funcMap := template.FuncMap{
		"wordCount": func(s string) int {
			return len(strings.Fields(s))
		},
		"formatDate": func(s string) string {
			if len(s) >= 19 {
				return strings.Replace(s[:19], "T", " ", 1)
			}
			return s
		},
	}
	var err error
	tmpl, err = template.New("index.html").Funcs(funcMap).ParseFiles("templates/index.html")
	if err != nil {
		log.Fatalf("Failed to parse templates: %v", err)
	}

	addr := os.Getenv("PORT")
	if addr == "" {
		addr = defaultAddr
	} else if !strings.Contains(addr, ":") {
		addr = ":" + addr
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/", handleIndex)
	mux.HandleFunc("/transcribe", handleTranscribe(apiKey))
	mux.HandleFunc("/progress/", handleProgress)
	mux.HandleFunc("/download/", handleDownload)
	mux.HandleFunc("/health", handleHealth(apiKey))

	log.Printf("Transcripter starting on %s", addr)
	log.Fatal(http.ListenAndServe(addr, mux))
}

func handleIndex(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}
	data := PageData{
		AvailableTranscripts: getAvailableTranscripts(),
	}
	renderTemplate(w, data)
}

func handleHealth(apiKey string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"status":            "healthy",
			"timestamp":         time.Now().Format(time.RFC3339),
			"version":           "0.2.0",
			"openai_configured": apiKey != "",
		})
	}
}

func handleProgress(w http.ResponseWriter, r *http.Request) {
	sessionID := strings.TrimPrefix(r.URL.Path, "/progress/")
	if sessionID == "" {
		http.Error(w, "missing session id", http.StatusBadRequest)
		return
	}

	progressMu.RLock()
	p, ok := progressMap[sessionID]
	progressMu.RUnlock()

	w.Header().Set("Content-Type", "application/json")
	if !ok {
		json.NewEncoder(w).Encode(Progress{
			Status:  "waiting",
			Message: "Waiting for transcription to start...",
		})
		return
	}
	json.NewEncoder(w).Encode(p)
}

func handleDownload(w http.ResponseWriter, r *http.Request) {
	filename := strings.TrimPrefix(r.URL.Path, "/download/")

	if !strings.HasPrefix(filename, "transcript_") || !strings.HasSuffix(filename, ".txt") {
		http.Error(w, "Invalid filename", http.StatusBadRequest)
		return
	}

	// Prevent path traversal
	if strings.Contains(filename, "/") || strings.Contains(filename, "..") {
		http.Error(w, "Invalid filename", http.StatusBadRequest)
		return
	}

	filePath := filepath.Join(transcriptsDir, filename)
	if _, err := os.Stat(filePath); os.IsNotExist(err) {
		http.Error(w, "File not found", http.StatusNotFound)
		return
	}

	w.Header().Set("Content-Disposition", fmt.Sprintf("attachment; filename=%s", filename))
	w.Header().Set("Content-Type", "text/plain; charset=utf-8")
	http.ServeFile(w, r, filePath)
}

func handleTranscribe(apiKey string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		// Parse multipart form (max 200MB in memory)
		if err := r.ParseMultipartForm(200 << 20); err != nil {
			http.Error(w, "Failed to parse form", http.StatusBadRequest)
			return
		}

		file, header, err := r.FormFile("file")
		if err != nil {
			http.Error(w, "No file provided", http.StatusBadRequest)
			return
		}
		defer file.Close()

		sessionID := r.FormValue("session_id")
		if sessionID == "" {
			sessionID = fmt.Sprintf("session_%d", time.Now().UnixNano())
		}

		contentType := header.Header.Get("Content-Type")
		if !strings.HasPrefix(contentType, "audio/") {
			setProgress(sessionID, &Progress{Status: "error", Message: "File must be an audio file"})
			http.Error(w, "File must be an audio file", http.StatusBadRequest)
			return
		}

		startTime := time.Now()
		log.Printf("Transcription started: %s (session: %s)", header.Filename, sessionID)

		setProgress(sessionID, &Progress{
			Status:      "uploading",
			ProgressPct: 10,
			Message:     "Saving uploaded file...",
		})

		// Save uploaded file to temp
		ext := filepath.Ext(header.Filename)
		if ext == "" {
			ext = ".mp3"
		}
		tmpFile, err := os.CreateTemp("", "transcripter-*"+ext)
		if err != nil {
			setProgress(sessionID, &Progress{Status: "error", Message: "Failed to create temp file"})
			http.Error(w, "Internal error", http.StatusInternalServerError)
			return
		}
		tmpPath := tmpFile.Name()
		defer os.Remove(tmpPath)

		if _, err := io.Copy(tmpFile, file); err != nil {
			tmpFile.Close()
			setProgress(sessionID, &Progress{Status: "error", Message: "Failed to save file"})
			http.Error(w, "Internal error", http.StatusInternalServerError)
			return
		}
		tmpFile.Close()

		setProgress(sessionID, &Progress{
			Status:      "processing",
			ProgressPct: 20,
			Message:     "Analyzing audio file...",
		})

		// Get audio duration via ffprobe
		duration, err := getAudioDuration(tmpPath)
		if err != nil {
			log.Printf("Error getting duration: %v", err)
			setProgress(sessionID, &Progress{Status: "error", Message: "Failed to analyze audio file"})
			http.Error(w, "Failed to analyze audio", http.StatusInternalServerError)
			return
		}
		log.Printf("Audio duration: %.2f seconds", duration)

		// Calculate chunks
		numChunks := int(duration/float64(maxChunkSeconds)) + 1
		if duration <= float64(maxChunkSeconds) {
			numChunks = 1
		}
		log.Printf("Will process %d chunk(s)", numChunks)

		setProgress(sessionID, &Progress{
			Status:       "transcribing",
			ProgressPct:  30,
			Message:      fmt.Sprintf("Starting transcription of %d chunk(s)...", numChunks),
			TotalChunks:  numChunks,
			CurrentChunk: 0,
		})

		var fullTranscript strings.Builder

		for i := 0; i < numChunks; i++ {
			startSec := float64(i) * float64(maxChunkSeconds)
			chunkDuration := float64(maxChunkSeconds)
			remaining := duration - startSec
			if remaining < chunkDuration {
				chunkDuration = remaining
			}

			progressPct := 30 + int(float64(i)/float64(numChunks)*60)
			setProgress(sessionID, &Progress{
				Status:       "transcribing",
				ProgressPct:  progressPct,
				Message:      fmt.Sprintf("Transcribing chunk %d/%d (%.0fs)...", i+1, numChunks, chunkDuration),
				CurrentChunk: i + 1,
				TotalChunks:  numChunks,
			})

			// Extract and compress chunk with ffmpeg
			chunkFile, err := os.CreateTemp("", "chunk-*.mp3")
			if err != nil {
				setProgress(sessionID, &Progress{Status: "error", Message: "Failed to create chunk file"})
				http.Error(w, "Internal error", http.StatusInternalServerError)
				return
			}
			chunkPath := chunkFile.Name()
			chunkFile.Close()

			cmd := exec.Command("ffmpeg", "-y",
				"-i", tmpPath,
				"-ss", fmt.Sprintf("%.2f", startSec),
				"-t", fmt.Sprintf("%.2f", chunkDuration),
				"-ac", "1",
				"-ar", "16000",
				"-b:a", "64k",
				chunkPath,
			)
			if output, err := cmd.CombinedOutput(); err != nil {
				os.Remove(chunkPath)
				log.Printf("ffmpeg error: %s", string(output))
				setProgress(sessionID, &Progress{Status: "error", Message: fmt.Sprintf("Failed to process chunk %d", i+1)})
				http.Error(w, "Audio processing failed", http.StatusInternalServerError)
				return
			}

			// Check file size
			fi, err := os.Stat(chunkPath)
			if err != nil {
				os.Remove(chunkPath)
				setProgress(sessionID, &Progress{Status: "error", Message: "Failed to read chunk"})
				http.Error(w, "Internal error", http.StatusInternalServerError)
				return
			}
			log.Printf("Chunk %d/%d: %d bytes", i+1, numChunks, fi.Size())

			if fi.Size() > maxFileSizeBytes {
				// Try lower bitrate
				cmd = exec.Command("ffmpeg", "-y",
					"-i", tmpPath,
					"-ss", fmt.Sprintf("%.2f", startSec),
					"-t", fmt.Sprintf("%.2f", chunkDuration),
					"-ac", "1",
					"-ar", "16000",
					"-b:a", "32k",
					chunkPath,
				)
				if output, err := cmd.CombinedOutput(); err != nil {
					os.Remove(chunkPath)
					log.Printf("ffmpeg error (retry): %s", string(output))
					setProgress(sessionID, &Progress{Status: "error", Message: fmt.Sprintf("Chunk %d too large", i+1)})
					http.Error(w, "Audio chunk too large", http.StatusInternalServerError)
					return
				}
			}

			// Send to OpenAI
			text, err := transcribeChunk(apiKey, chunkPath)
			os.Remove(chunkPath)
			if err != nil {
				log.Printf("OpenAI error for chunk %d: %v", i+1, err)
				setProgress(sessionID, &Progress{
					Status:       "error",
					ProgressPct:  progressPct,
					Message:      fmt.Sprintf("Transcription failed for chunk %d: %v", i+1, err),
					CurrentChunk: i + 1,
					TotalChunks:  numChunks,
				})
				http.Error(w, "Transcription failed", http.StatusInternalServerError)
				return
			}

			fullTranscript.WriteString(strings.TrimSpace(text))
			fullTranscript.WriteString("\n")
			log.Printf("Chunk %d/%d transcribed: %d chars", i+1, numChunks, len(text))
		}

		totalTime := time.Since(startTime).Seconds()
		transcriptText := fullTranscript.String()
		log.Printf("Transcription complete in %.2fs, %d characters", totalTime, len(transcriptText))

		setProgress(sessionID, &Progress{
			Status:       "saving",
			ProgressPct:  90,
			Message:      "Saving transcript...",
			CurrentChunk: numChunks,
			TotalChunks:  numChunks,
		})

		// Save transcript
		transcriptFilename := fmt.Sprintf("transcript_%s.txt", time.Now().Format("20060102_150405"))
		transcriptPath := filepath.Join(transcriptsDir, transcriptFilename)

		fileContent := fmt.Sprintf("Transcription of: %s\nGenerated on: %s\nDuration: %.2f seconds\nProcessing time: %.2f seconds\n%s\n\n%s",
			header.Filename,
			time.Now().Format("2006-01-02 15:04:05"),
			duration,
			totalTime,
			strings.Repeat("-", 50),
			transcriptText,
		)

		if err := os.WriteFile(transcriptPath, []byte(fileContent), 0644); err != nil {
			log.Printf("Failed to save transcript: %v", err)
		}

		// Update metadata
		addTranscriptMetadata(transcriptFilename, header.Filename,
			fmt.Sprintf("%.2f", duration),
			fmt.Sprintf("%.2f", totalTime),
			len(transcriptText))

		setProgress(sessionID, &Progress{
			Status:       "completed",
			ProgressPct:  100,
			Message:      fmt.Sprintf("Transcription completed! %d characters transcribed.", len(transcriptText)),
			CurrentChunk: numChunks,
			TotalChunks:  numChunks,
		})

		data := PageData{
			Transcript:           transcriptText,
			TranscriptFilename:   transcriptFilename,
			OriginalFilename:     header.Filename,
			ProcessingTime:       fmt.Sprintf("%.2f", totalTime),
			AudioDuration:        fmt.Sprintf("%.2f", duration),
			AvailableTranscripts: getAvailableTranscripts(),
			SessionID:            sessionID,
			TranscriptChars:      len(transcriptText),
			TranscriptWords:      len(strings.Fields(transcriptText)),
		}
		renderTemplate(w, data)
	}
}

// getAudioDuration returns audio duration in seconds using ffprobe.
func getAudioDuration(path string) (float64, error) {
	cmd := exec.Command("ffprobe",
		"-v", "quiet",
		"-show_entries", "format=duration",
		"-of", "csv=p=0",
		path,
	)
	out, err := cmd.Output()
	if err != nil {
		return 0, fmt.Errorf("ffprobe failed: %w", err)
	}
	s := strings.TrimSpace(string(out))
	return strconv.ParseFloat(s, 64)
}

// transcribeChunk sends an audio file to the OpenAI Whisper API.
func transcribeChunk(apiKey, filePath string) (string, error) {
	f, err := os.Open(filePath)
	if err != nil {
		return "", fmt.Errorf("open chunk: %w", err)
	}
	defer f.Close()

	var body bytes.Buffer
	writer := multipart.NewWriter(&body)

	if err := writer.WriteField("model", "whisper-1"); err != nil {
		return "", err
	}

	part, err := writer.CreateFormFile("file", filepath.Base(filePath))
	if err != nil {
		return "", err
	}
	if _, err := io.Copy(part, f); err != nil {
		return "", err
	}
	writer.Close()

	req, err := http.NewRequest("POST", "https://api.openai.com/v1/audio/transcriptions", &body)
	if err != nil {
		return "", err
	}
	req.Header.Set("Authorization", "Bearer "+apiKey)
	req.Header.Set("Content-Type", writer.FormDataContentType())

	client := &http.Client{Timeout: 5 * time.Minute}
	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("API request failed: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("API error %d: %s", resp.StatusCode, string(respBody))
	}

	var result struct {
		Text string `json:"text"`
	}
	if err := json.Unmarshal(respBody, &result); err != nil {
		return "", fmt.Errorf("parse response: %w", err)
	}
	return result.Text, nil
}

func setProgress(sessionID string, p *Progress) {
	progressMu.Lock()
	progressMap[sessionID] = p
	progressMu.Unlock()
}

func renderTemplate(w http.ResponseWriter, data PageData) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	if err := tmpl.Execute(w, data); err != nil {
		log.Printf("Template error: %v", err)
		http.Error(w, "Internal error", http.StatusInternalServerError)
	}
}

// Metadata helpers

func loadMetadata() map[string]TranscriptMeta {
	data, err := os.ReadFile(metadataFile)
	if err != nil {
		return make(map[string]TranscriptMeta)
	}
	var m map[string]TranscriptMeta
	if err := json.Unmarshal(data, &m); err != nil {
		log.Printf("Error loading metadata: %v", err)
		return make(map[string]TranscriptMeta)
	}
	return m
}

func saveMetadata(m map[string]TranscriptMeta) {
	data, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		log.Printf("Error marshaling metadata: %v", err)
		return
	}
	if err := os.WriteFile(metadataFile, data, 0644); err != nil {
		log.Printf("Error saving metadata: %v", err)
	}
}

func addTranscriptMetadata(filename, originalFilename, duration, processingTime string, transcriptLength int) {
	m := loadMetadata()
	id := strings.TrimSuffix(filename, ".txt")
	m[id] = TranscriptMeta{
		Filename:         filename,
		OriginalFilename: originalFilename,
		CreatedAt:        time.Now().Format(time.RFC3339),
		Duration:         duration,
		ProcessingTime:   processingTime,
		TranscriptLength: transcriptLength,
		FilePath:         filepath.Join(transcriptsDir, filename),
	}
	saveMetadata(m)
}

func getAvailableTranscripts() []TranscriptView {
	m := loadMetadata()
	var views []TranscriptView
	for id, info := range m {
		if _, err := os.Stat(info.FilePath); err == nil {
			views = append(views, TranscriptView{
				ID:               id,
				Filename:         info.Filename,
				OriginalFilename: info.OriginalFilename,
				CreatedAt:        info.CreatedAt,
				Duration:         info.Duration,
				ProcessingTime:   info.ProcessingTime,
				TranscriptLength: info.TranscriptLength,
			})
		}
	}
	sort.Slice(views, func(i, j int) bool {
		return views[i].CreatedAt > views[j].CreatedAt
	})
	return views
}
