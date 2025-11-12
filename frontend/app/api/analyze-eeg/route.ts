import { type NextRequest, NextResponse } from "next/server"
import { writeFile, readFile, mkdir } from "fs/promises"
import path from "path"
import { spawn } from "child_process"

export async function POST(request: NextRequest) {
  const startTime = Date.now()

  try {
    const formData = await request.formData()
    const file = formData.get("file") as File

    if (!file) {
      return NextResponse.json({ message: "No file provided" }, { status: 400 })
    }

    // Validate file extension
    if (!file.name.endsWith(".edf")) {
      return NextResponse.json({ message: "Invalid file format. Please upload an EDF file." }, { status: 400 })
    }

    const repoRoot = path.join(process.cwd(), "..")
    const uploadsDir = path.join(repoRoot, "uploads")
    await mkdir(uploadsDir, { recursive: true })

    const buffer = Buffer.from(await file.arrayBuffer())
    const fileName = `${Date.now()}_${file.name}`
    const filePath = path.join(uploadsDir, fileName)
    await writeFile(filePath, buffer)

    const outDir = path.join(repoRoot, "results", "edf_inference", path.parse(fileName).name)
    await mkdir(outDir, { recursive: true })

    const scriptPath = path.join(repoRoot, "inverse_problem", "run_edf_inference.py")

    // Run the Python script in inv_solver conda environment
    const args = [scriptPath, filePath, "--output_dir", outDir]
    await new Promise<void>((resolve, reject) => {
      const proc = spawn("conda", ["run", "-n", "inv_solver", "python3", ...args], { cwd: repoRoot })
      let stderr = ""
      proc.stderr.on("data", (data) => {
        stderr += data.toString()
      })
      proc.on("close", (code) => {
        if (code === 0) {
          resolve()
        } else {
          reject(new Error(`Python script failed: ${stderr}`))
        }
      })
      proc.on("error", reject)
    })

    // Read the summary to get the plot path
    const summaryPath = path.join(outDir, "best_window_summary.json")
    const summaryContent = await readFile(summaryPath, "utf8")
    const summary = JSON.parse(summaryContent)

    const plotPath = path.join(outDir, summary.interactive_plot)
    const plotHtml = await readFile(plotPath, "utf8")

    // Extract the body content (assuming the plot is in <body>)
    const bodyMatch = plotHtml.match(/<body[^>]*>([\s\S]*?)<\/body>/i)
    const plotContent = bodyMatch ? bodyMatch[1] : plotHtml

    const styledPlotHtml = `<div style='width: 100%; height: 100%; display: flex; justify-content: center; align-items: center;'>${plotContent}</div>`

    const processingTime = (Date.now() - startTime) / 1000

    return NextResponse.json({
      success: true,
      processingTime,
      plotHtml: styledPlotHtml,
      message: "EEG analysis completed successfully",
    })
  } catch (error) {
    console.error("Error processing EEG file:", error)
    return NextResponse.json({ message: error instanceof Error ? error.message : "Failed to process EEG file" }, { status: 500 })
  }
}
