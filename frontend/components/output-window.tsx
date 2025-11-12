"use client"

import { useState } from "react"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { RotateCcw } from "lucide-react"
import { BrainVisualization } from "./brain-visualization"

interface OutputWindowProps {
  result: {
    fileName: string
    uploadTime: string
    processingTime: number
    plotHtml: string
  }
  onNewAnalysis: () => void
}

export function OutputWindow({ result, onNewAnalysis }: OutputWindowProps) {
  const [activeTab, setActiveTab] = useState<"visualization" | "details">("visualization")

  return (
    <div className="space-y-6">
      {/* Header with Actions */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h2 className="text-2xl font-semibold text-white">Analysis Complete</h2>
          <p className="text-sm text-slate-300 mt-1">
            File: <span className="font-medium text-white">{result.fileName}</span>
          </p>
        </div>
        <div className="flex gap-3 w-full sm:w-auto">
          <Button onClick={onNewAnalysis} variant="outline" className="flex-1 sm:flex-none bg-transparent border-border/60 text-emerald-300 hover:text-white hover:bg-emerald-500/10">
            <RotateCcw className="w-4 h-4 mr-2" />
            New Analysis
          </Button>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex gap-2 border-b border-border">
        <button
          onClick={() => setActiveTab("visualization")}
          className={`px-4 py-2 font-medium text-sm transition-colors ${
            activeTab === "visualization"
              ? "text-emerald-300 border-b-2 border-emerald-300"
              : "text-slate-400 hover:text-slate-200"
          }`}
        >
          3D Brain Map
        </button>
        <button
          onClick={() => setActiveTab("details")}
          className={`px-4 py-2 font-medium text-sm transition-colors ${
            activeTab === "details"
              ? "text-emerald-300 border-b-2 border-emerald-300"
              : "text-slate-400 hover:text-slate-200"
          }`}
        >
          Analysis Details
        </button>
      </div>

      {/* Content */}
      {activeTab === "visualization" && (
        <Card className="bg-card border border-border">
          <BrainVisualization plotHtml={result.plotHtml} />
        </Card>
      )}

      {activeTab === "details" && (
        <Card className="p-6 bg-background/40 border border-border/60">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Input Information */}
            <div>
              <h3 className="text-lg font-semibold text-white mb-4">Input Information</h3>
              <div className="space-y-3">
                <div>
                  <p className="text-xs font-medium text-slate-400 uppercase tracking-wide">File Name</p>
                  <p className="text-sm text-slate-200 mt-1">{result.fileName}</p>
                </div>
                <div>
                  <p className="text-xs font-medium text-slate-400 uppercase tracking-wide">Upload Time</p>
                  <p className="text-sm text-slate-200 mt-1">{result.uploadTime}</p>
                </div>
              </div>
            </div>

            {/* Processing Information */}
            <div>
              <h3 className="text-lg font-semibold text-white mb-4">Processing Information</h3>
              <div className="space-y-3">
                <div>
                  <p className="text-xs font-medium text-slate-400 uppercase tracking-wide">Processing Time</p>
                  <p className="text-sm text-slate-200 mt-1">{result.processingTime.toFixed(2)}s</p>
                </div>
                <div>
                  <p className="text-xs font-medium text-slate-400 uppercase tracking-wide">Status</p>
                  <p className="text-sm text-emerald-300 font-medium mt-1">✓ Completed Successfully</p>
                </div>
              </div>
            </div>
          </div>
        </Card>
      )}
    </div>
  )
}
