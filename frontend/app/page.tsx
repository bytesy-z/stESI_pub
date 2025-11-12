"use client"

import { useState } from "react"
import { FileUploadSection } from "@/components/file-upload-section"
import { ProcessingWindow } from "@/components/processing-window"
import { OutputWindow } from "@/components/output-window"
import { ErrorAlert } from "@/components/error-alert"

type AppState = "upload" | "processing" | "output" | "error"

interface AnalysisResult {
  fileName: string
  uploadTime: string
  processingTime: number
  plotHtml: string
}

export default function Home() {
  const [appState, setAppState] = useState<AppState>("upload")
  const [error, setError] = useState<string | null>(null)
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null)

  const handleFileUpload = async (file: File) => {
    if (!file.name.endsWith(".edf")) {
      setError("Invalid file format. Please upload an EDF file.")
      setAppState("error")
      return
    }

    setAppState("processing")
    setError(null)

    try {
      const formData = new FormData()
      formData.append("file", file)

      const response = await fetch("/api/analyze-eeg", {
        method: "POST",
        body: formData,
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.message || "Failed to process EEG file")
      }

      const data = await response.json()

      setAnalysisResult({
        fileName: file.name,
        uploadTime: new Date().toLocaleString(),
        processingTime: data.processingTime || 0,
        plotHtml: data.plotHtml,
      })

      setAppState("output")
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred during processing")
      setAppState("error")
    }
  }

  const handleRetry = () => {
    setAppState("upload")
    setError(null)
    setAnalysisResult(null)
  }

  return (
    <main className="min-h-screen w-full flex flex-col bg-[#1f2937] text-foreground">
      <div className="max-w-7xl text-center mx-auto py-10">
        {/* Header */}
        <div className="mb-12">
          <h1 className="text-4xl font-bold mb-2 text-chart-4">VESL</h1>
          <p className="text-popover">Visual EEG source localization</p>
        </div>

        {/* Main Content */}
        <div className="space-y-6">
          {appState === "upload" && <FileUploadSection onFileUpload={handleFileUpload} />}
          {appState === "processing" && <ProcessingWindow />}
          {appState === "output" && analysisResult && (
            <OutputWindow result={analysisResult} onNewAnalysis={handleRetry} />
          )}
          {appState === "error" && <ErrorAlert message={error || "An error occurred"} onRetry={handleRetry} />}
        </div>
      </div>
    </main>
  )
}
