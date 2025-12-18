import { type NextRequest, NextResponse } from "next/server"
import { writeFile, readFile, mkdir } from "fs/promises"
import path from "path"
import { spawn } from "child_process"

export const maxDuration = 300
export const dynamic = 'force-dynamic'

const jobs = new Map<string, {
  status: 'processing' | 'completed' | 'failed'
  startTime: number
  error?: string
  result?: {
    plotHtml: string
    outputDir: string
    processingTime: number
    hasGroundTruth?: boolean
    metrics?: unknown
    bestWindow?: unknown
    nWindowsProcessed?: number
    sourceFile?: string
  }
}>()

export { jobs }

export async function POST(request: NextRequest) {
  const startTime = Date.now()

  try {
    const formData = await request.formData()
    const file = formData.get("file") as File

    if (!file) {
      return NextResponse.json({ message: "No file provided" }, { status: 400 })
    }

    if (!file.name.endsWith(".mat")) {
      return NextResponse.json({ message: "Invalid file format. Please upload a MAT file." }, { status: 400 })
    }

    const repoRoot = path.join(process.cwd(), "..")
    const uploadsDir = path.join(repoRoot, "uploads")
    await mkdir(uploadsDir, { recursive: true })

    const buffer = Buffer.from(await file.arrayBuffer())
    const timestamp = Date.now()
    const fileName = timestamp + "_" + file.name
    const filePath = path.join(uploadsDir, fileName)
    await writeFile(filePath, buffer)

    const outDir = path.join(repoRoot, "results", "mat_inference", path.parse(fileName).name)
    await mkdir(outDir, { recursive: true })

    const scriptPath = path.join(repoRoot, "inverse_problem", "run_mat_inference.py")

    const jobId = "mat_" + timestamp + "_" + Math.random().toString(36).substr(2, 9)
    
    jobs.set(jobId, {
      status: 'processing',
      startTime,
    })
    
    const jobInfoPath = path.join(outDir, "_job_info.json")
    await writeFile(jobInfoPath, JSON.stringify({
      jobId,
      status: 'processing',
      startTime,
      fileName: file.name,
    }))

    const args = [
      scriptPath,
      filePath,
      "--output_dir",
      outDir,
      "--overlap_fraction",
      "0.5",
      "--use_global_norm",
      "--smoothing_alpha",
      "0.3",
    ]
    
    console.log("[MAT Processing] Starting background analysis for " + file.name + " (job: " + jobId + ")...")
    
    const proc = spawn("conda", ["run", "-n", "inv_solver", "python3", ...args], { 
      cwd: repoRoot,
      detached: true,
      stdio: ['ignore', 'pipe', 'pipe']
    })
    
    let stderr = ""
    
    proc.stdout?.on("data", (data) => {
      const output = data.toString()
      if (output.includes("Loaded") || output.includes("Generating animation") || output.includes("Saved") || output.includes("window")) {
        console.log("[MAT Job " + jobId + "] " + output.trim())
      }
    })
    
    proc.stderr?.on("data", (data) => {
      stderr += data.toString()
    })
    
    proc.on("close", async (code) => {
      const processingTime = (Date.now() - startTime) / 1000
      
      if (code === 0) {
        console.log("[MAT Job " + jobId + "] Completed successfully in " + processingTime.toFixed(1) + "s")
        
        try {
          const summaryPath = path.join(outDir, "inference_summary.json")
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
              hasGroundTruth: summary.has_ground_truth || false,
              metrics: summary.metrics || null,
              bestWindow: summary.best_window || null,
              nWindowsProcessed: summary.n_windows_processed || 0,
              sourceFile: summary.source_file || null,
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
          console.error("[MAT Job " + jobId + "] Error reading results:", err)
          jobs.set(jobId, {
            status: 'failed',
            startTime,
            error: 'Failed to read results after processing',
          })
        }
      } else {
        console.error("[MAT Job " + jobId + "] Failed with code " + code)
        console.error("[MAT Job " + jobId + "] stderr: " + stderr)
        
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
      console.error("[MAT Job " + jobId + "] Error:", err)
      jobs.set(jobId, {
        status: 'failed',
        startTime,
        error: err.message,
      })
    })
    
    proc.unref()

    return NextResponse.json({
      success: true,
      jobId,
      outputDir: path.relative(repoRoot, outDir),
      message: "Processing started. Poll /api/job-status for completion.",
    })
    
  } catch (error) {
    console.error("Error starting MAT processing:", error)
    return NextResponse.json({ message: error instanceof Error ? error.message : "Failed to start MAT processing" }, { status: 500 })
  }
}
