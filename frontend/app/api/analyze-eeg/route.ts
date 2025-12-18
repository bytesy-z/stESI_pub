import { type NextRequest, NextResponse } from "next/server"
import { writeFile, readFile, mkdir } from "fs/promises"
import path from "path"
import { spawn } from "child_process"

export const maxDuration = 300
export const dynamic = 'force-dynamic'

const test = false

const jobs = new Map<string, {
  status: 'processing' | 'completed' | 'failed'
  startTime: number
  error?: string
  result?: {
    plotHtml: string
    outputDir: string
    processingTime: number
  }
}>()

export { jobs }

export async function POST(request: NextRequest) {
  const startTime = Date.now()
  console.log("[EEG API] Received request")

  try {
    console.log("[EEG API] Parsing form data...")
    const formData = await request.formData()
    const file = formData.get("file") as File

    if (!file) {
      console.error("[EEG API] No file provided")
      return NextResponse.json({ message: "No file provided" }, { status: 400 })
    }

    console.log(`[EEG API] Received file: ${file.name} (${file.size} bytes)`)

    if (!file.name.endsWith(".edf")) {
      console.error(`[EEG API] Invalid file format: ${file.name}`)
      return NextResponse.json({ message: "Invalid file format. Please upload an EDF file." }, { status: 400 })
    }

    console.log("[EEG API] Setting up file paths...")
    const repoRoot = path.join(process.cwd(), "..")
    const uploadsDir = path.join(repoRoot, "uploads")
    await mkdir(uploadsDir, { recursive: true })

    console.log("[EEG API] Converting file to buffer...")
    const buffer = Buffer.from(await file.arrayBuffer())
    const timestamp = Date.now()
    const fileName = timestamp + "_" + file.name
    const filePath = path.join(uploadsDir, fileName)
    console.log(`[EEG API] Writing file to: ${filePath}`)
    await writeFile(filePath, buffer)

    const outDir = path.join(repoRoot, "results", "edf_inference", path.parse(fileName).name)
    console.log(`[EEG API] Creating output directory: ${outDir}`)
    await mkdir(outDir, { recursive: true })

    const scriptPath = path.join(repoRoot, "inverse_problem", "run_edf_inference.py")

    const jobId = "eeg_" + timestamp + "_" + Math.random().toString(36).substr(2, 9)
    console.log(`[EEG API] Generated jobId: ${jobId}`)
    
    console.log("[EEG API] Storing job in memory...")
    jobs.set(jobId, {
      status: 'processing',
      startTime,
    })
    
    const jobInfoPath = path.join(outDir, "_job_info.json")
    console.log(`[EEG API] Writing job info to: ${jobInfoPath}`)
    await writeFile(jobInfoPath, JSON.stringify({
      jobId,
      status: 'processing',
      startTime,
      fileName: file.name,
    }))

    var args: string[]

    if (test) {
      args = [
        scriptPath,
        filePath,
        "--output_dir",
        outDir,
        "--overlap_fraction",
        "0.5",
        "--use_global_norm",
  "--smoothing_alpha",
  "0.6",
        "--max_windows",
        "100"
      ]
    }
    else {
      args = [
      scriptPath,
      filePath,
      "--output_dir",
      outDir,
      "--overlap_fraction",
      "0.5",
      "--use_global_norm",
  "--smoothing_alpha",
  "0.6"
    ]
  }

    console.log("[EEG API] Starting background analysis for " + file.name + " (job: " + jobId + ")...")
    console.log("[EEG API] Python command:", ["conda", "run", "-n", "inv_solver", "python3", ...args])
    
    const proc = spawn("conda", ["run", "-n", "inv_solver", "python3", ...args], { 
      cwd: repoRoot,
      detached: true,
      stdio: ['ignore', 'pipe', 'pipe']
    })
    
    console.log(`[EEG API] Process spawned with PID: ${proc.pid}`)
    
    let stderr = ""
    
    proc.stdout?.on("data", (data) => {
      const output = data.toString()
      if (output.includes("Loaded EDF") || output.includes("Generating animation") || output.includes("Saved") || output.includes("window")) {
        console.log("[EEG Job " + jobId + "] " + output.trim())
      }
    })
    
    proc.stderr?.on("data", (data) => {
      stderr += data.toString()
    })
    
    proc.on("close", async (code) => {
      const processingTime = (Date.now() - startTime) / 1000
      
      if (code === 0) {
        console.log("[EEG Job " + jobId + "] Completed successfully in " + processingTime.toFixed(1) + "s")
        
        try {
          const summaryPath = path.join(outDir, "best_window_summary.json")
          const summaryContent = await readFile(summaryPath, "utf8")
          const summary = JSON.parse(summaryContent)

          const plotPath = path.join(outDir, summary.interactive_plot)
          const plotHtml = await readFile(plotPath, "utf8")

          const bodyMatch = plotHtml.match(/<body[^>]*>([\s\S]*?)<\/body>/i)
          const plotContent = bodyMatch ? bodyMatch[1] : plotHtml
          const styledPlotHtml = "<div style='width: 100%; height: 100%; display: flex; justify-content: center; align-items: center;'>" + plotContent + "</div>"

          const outputDirRelative = path.relative(repoRoot, outDir)

          jobs.set(jobId, {
            status: 'completed',
            startTime,
            result: {
              plotHtml: styledPlotHtml,
              outputDir: outputDirRelative,
              processingTime,
            }
          })
          
          await writeFile(jobInfoPath, JSON.stringify({
            jobId,
            status: 'completed',
            startTime,
            processingTime,
            outputDir: outputDirRelative,
          }))
          
        } catch (err) {
          console.error("[EEG Job " + jobId + "] Error reading results:", err)
          jobs.set(jobId, {
            status: 'failed',
            startTime,
            error: 'Failed to read results after processing',
          })
        }
      } else {
        console.error("[EEG Job " + jobId + "] Failed with code " + code)
        console.error("[EEG Job " + jobId + "] stderr: " + stderr)
        
        jobs.set(jobId, {
          status: 'failed',
          startTime,
          error: stderr || "Process exited with code " + code,
        })
        
        await writeFile(jobInfoPath, JSON.stringify({
          jobId,
          status: 'failed',
          startTime,
          error: stderr,
        }))
      }
    })
    
    proc.on("error", async (err) => {
      console.error("[EEG Job " + jobId + "] Error:", err)
      jobs.set(jobId, {
        status: 'failed',
        startTime,
        error: err.message,
      })
    })
    
    proc.unref()

    const responseData = {
      success: true,
      jobId,
      outputDir: path.relative(repoRoot, outDir),
      message: "Processing started. Poll /api/job-status for completion.",
    }
    console.log("[EEG API] Returning response:", responseData)
    return NextResponse.json(responseData)
    
  } catch (error) {
    console.error("Error starting EEG processing:", error)
    return NextResponse.json({ message: error instanceof Error ? error.message : "Failed to start EEG processing" }, { status: 500 })
  }
}
