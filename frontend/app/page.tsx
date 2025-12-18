"use client"

import { useState, useCallback } from "react"
import { FileUploadSection } from "@/components/file-upload-section"
import { ProcessingWindow } from "@/components/processing-window"
import { OutputWindow } from "@/components/output-window"
import { ErrorAlert } from "@/components/error-alert"

type AppState = "upload" | "processing" | "output" | "error"

interface EvaluationMetrics {
  mean_nmse: number
  mean_auc: number
  mean_localization_error_mm: number
  mean_time_error_ms?: number
  best_window_nmse?: number
  best_window_auc?: number
  best_window_localization_error_mm?: number
  n_windows?: number
}

interface AnalysisResult {
  fileName: string
  fileType: "edf" | "mat"
  uploadTime: string
  processingTime: number
  plotHtml: string
  animationFile?: string
  outputDir?: string
  // MAT-specific fields
  hasGroundTruth?: boolean
  metrics?: EvaluationMetrics
  nWindowsProcessed?: number
  sourceFile?: string
}

export default function Home() {
  const [appState, setAppState] = useState<AppState>("upload")
  const [error, setError] = useState<string | null>(null)
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null)
  const [elapsedTime, setElapsedTime] = useState(0)

  // Poll for job status
  const pollJobStatus = useCallback(async (jobId: string, outputDir: string, isMat: boolean, fileName: string): Promise<void> => {
    const startTime = Date.now()
    console.log(`[FRONTEND] Starting to poll for job ${jobId}`)
    
    let pollCount = 0
    while (true) {
      await new Promise(resolve => setTimeout(resolve, 2000)) // Poll every 2 seconds
      pollCount++
      
      try {
        console.log(`[FRONTEND] Poll #${pollCount} for job ${jobId}`)
        const statusResponse = await fetch(`/api/job-status?jobId=${encodeURIComponent(jobId)}&outputDir=${encodeURIComponent(outputDir)}`)
        
        if (!statusResponse.ok) {
          console.error(`[FRONTEND] Status check failed: ${statusResponse.status} ${statusResponse.statusText}`)
          continue
        }
        
        const statusData = await statusResponse.json()
        console.log(`[FRONTEND] Status response:`, statusData)
        
        const elapsed = Math.round((Date.now() - startTime) / 1000)
        setElapsedTime(elapsed)
        console.log(`[FRONTEND] Elapsed time: ${elapsed}s`)
        
        if (statusData.status === 'completed') {
          console.log(`[FRONTEND] Job completed! Building result object...`)
          const result: AnalysisResult = {
            fileName,
            fileType: isMat ? "mat" : "edf",
            uploadTime: new Date().toLocaleString(),
            processingTime: statusData.processingTime || statusData.result?.processingTime || 0,
            plotHtml: statusData.result?.plotHtml || '',
            animationFile: "animation_data.npz",
            outputDir: statusData.result?.outputDir || outputDir,
            hasGroundTruth: statusData.result?.hasGroundTruth,
            metrics: statusData.result?.metrics,
            nWindowsProcessed: statusData.result?.nWindowsProcessed,
            sourceFile: statusData.result?.sourceFile,
          }
          console.log(`[FRONTEND] Setting analysis result:`, result)
          setAnalysisResult(result)
          console.log(`[FRONTEND] Switching to output state`)
          setAppState("output")
          return
        } else if (statusData.status === 'failed') {
          console.error(`[FRONTEND] Job failed:`, statusData.error)
          throw new Error(statusData.error || 'Processing failed')
        } else {
          console.log(`[FRONTEND] Still processing... (status: ${statusData.status})`)
        }
        // Still processing, continue polling
      } catch (err) {
        console.error(`[FRONTEND] Error during polling:`, err)
        
        // Check if it's a network error (server crashed)
        if (err instanceof TypeError && err.message.includes('fetch')) {
          console.error(`[FRONTEND] Network error - server may have crashed`)
          // Wait longer and retry a few times before giving up
          if (pollCount < 5) {
            console.log(`[FRONTEND] Retrying in 10 seconds... (attempt ${pollCount}/5)`)
            await new Promise(resolve => setTimeout(resolve, 10000))
            continue
          } else {
            throw new Error('Server appears to have crashed. Please restart the development server and try again.')
          }
        }
        
        throw err
      }
    }
  }, [])

  const handleFileUpload = async (file: File) => {
    console.log(`[FRONTEND] Starting file upload for: ${file.name} (size: ${file.size} bytes)`)
    
    const isEdf = file.name.endsWith(".edf")
    const isMat = file.name.endsWith(".mat")
    
    console.log(`[FRONTEND] File type detection: isEdf=${isEdf}, isMat=${isMat}`)
    
    if (!isEdf && !isMat) {
      console.error(`[FRONTEND] Invalid file format: ${file.name}`)
      setError("Invalid file format. Please upload an EDF or MAT file.")
      setAppState("error")
      return
    }

    console.log(`[FRONTEND] Switching to processing state`)
    setAppState("processing")
    setError(null)
    setElapsedTime(0)

    try {
      console.log(`[FRONTEND] Creating form data...`)
      const formData = new FormData()
      formData.append("file", file)

      // Use different API endpoint based on file type
      const apiEndpoint = isMat ? "/api/analyze-mat" : "/api/analyze-eeg"
      console.log(`[FRONTEND] Using API endpoint: ${apiEndpoint}`)
      
      // Submit file and get job ID (returns immediately)
      console.log(`[FRONTEND] Submitting file to API...`)
      const response = await fetch(apiEndpoint, {
        method: "POST",
        body: formData,
      })

      console.log(`[FRONTEND] API response status: ${response.status} ${response.statusText}`)

      if (!response.ok) {
        const errorData = await response.json()
        console.error(`[FRONTEND] API error:`, errorData)
        throw new Error(errorData.message || "Failed to start processing")
      }

      const data = await response.json()
      console.log(`[FRONTEND] API response data:`, data)
      
      // Check if this is the new async format (has jobId)
      if (data.jobId) {
        console.log(`[FRONTEND] Got jobId: ${data.jobId}, starting polling...`)
        // Poll for completion
        await pollJobStatus(data.jobId, data.outputDir, isMat, file.name)
      } else {
        // Legacy sync format (direct result)
        setAnalysisResult({
          fileName: file.name,
          fileType: isMat ? "mat" : "edf",
          uploadTime: new Date().toLocaleString(),
          processingTime: data.processingTime || 0,
          plotHtml: data.plotHtml,
          animationFile: data.animationFile,
          outputDir: data.outputDir,
          hasGroundTruth: data.hasGroundTruth,
          metrics: data.metrics,
          nWindowsProcessed: data.nWindowsProcessed,
          sourceFile: data.sourceFile,
        })
        setAppState("output")
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred during processing")
      setAppState("error")
    }
  }

  const handleRetry = () => {
    setAppState("upload")
    setError(null)
    setAnalysisResult(null)
    setElapsedTime(0)
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
          {appState === "processing" && <ProcessingWindow elapsedTime={elapsedTime} />}
          {appState === "output" && analysisResult && (
            <OutputWindow result={analysisResult} onNewAnalysis={handleRetry} />
          )}
          {appState === "error" && <ErrorAlert message={error || "An error occurred"} onRetry={handleRetry} />}
        </div>
      </div>
    </main>
  )
}
