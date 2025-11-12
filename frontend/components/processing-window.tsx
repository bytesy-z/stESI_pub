"use client"

import { useEffect, useState } from "react"
import { Card } from "@/components/ui/card"
import { Loader2 } from "lucide-react"

export function ProcessingWindow() {
  const [progress, setProgress] = useState(0)

  useEffect(() => {
    const interval = setInterval(() => {
      setProgress((prev) => {
        if (prev >= 90) return 90
        return Math.min(90, prev + Math.random() * 20)
      })
    }, 500)

    return () => clearInterval(interval)
  }, [])

  const steps = [
    { name: "Validating EDF Format", completed: progress > 10 },
    { name: "Loading Signal Data", completed: progress > 30 },
    { name: "Preprocessing Signals", completed: progress > 50 },
    { name: "Computing Source Localization", completed: progress > 70 },
  { name: "Generating 3D Brain Maps", completed: progress >= 90 },
  ]

  return (
    <Card className="p-8 bg-card border border-border">
      <div className="max-w-2xl mx-auto">
        <div className="flex items-center justify-center mb-8">
          <Loader2 className="w-8 h-8 text-primary animate-spin mr-3" />
          <h2 className="text-2xl font-semibold text-foreground">Processing EEG Data</h2>
        </div>

        {/* Progress Bar */}
        <div className="mb-8">
          <div className="flex justify-between items-center mb-2">
            <span className="text-sm font-medium text-foreground">Overall Progress</span>
            <span className="text-sm text-muted-foreground">{Math.round(progress)}%</span>
          </div>
          <div className="w-full h-2 bg-muted rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-primary to-accent transition-all duration-300"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>

        {/* Processing Steps */}
        <div className="space-y-3">
          {steps.map((step, index) => (
            <div key={index} className="flex items-center gap-3">
              <div
                className={`w-5 h-5 rounded-full flex items-center justify-center text-xs font-bold transition-all ${
                  step.completed ? "bg-primary text-primary-foreground" : "bg-muted text-muted-foreground"
                }`}
              >
                {step.completed ? "✓" : index + 1}
              </div>
              <span className={`text-sm ${step.completed ? "text-foreground font-medium" : "text-muted-foreground"}`}>
                {step.name}
              </span>
            </div>
          ))}
        </div>

        <div className="mt-8 p-4 bg-muted/30 rounded-lg border border-border">
          <p className="text-xs text-muted-foreground text-center">
            This may take a few minutes depending on the file size and complexity
          </p>
        </div>
      </div>
    </Card>
  )
}
