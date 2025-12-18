"use client"

import type React from "react"

import { useState, useRef } from "react"
import { Upload } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"

interface FileUploadSectionProps {
  onFileUpload: (file: File) => void
}

export function FileUploadSection({ onFileUpload }: FileUploadSectionProps) {
  const [isDragActive, setIsDragActive] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragActive(e.type === "dragenter" || e.type === "dragover")
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragActive(false)

    const files = e.dataTransfer.files
    if (files && files.length > 0) {
      onFileUpload(files[0])
    }
  }

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (files && files.length > 0) {
      onFileUpload(files[0])
    }
  }

  return (
    <div className="flex w-full flex-row rounded-xl items-center justify-center h-max text-foreground bg-card-foreground px-0 py-0">
      <Card
        className={`w-full max-w-md p-8 border-2 border-dashed transition-all cursor-pointer bg-sidebar-accent ${
          isDragActive
            ? "border-[#52c4a0] bg-slate-700"
            : "border-slate-600 hover:border-[#52c4a0] hover:bg-slate-700/50"
        }`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        <div className="flex flex-col items-center justify-center text-center py-12">
          <div className="mb-4 p-3 bg-[#52c4a0]/10 rounded-lg bg-muted opacity-85">
            <Upload className="w-8 h-8 text-[#52c4a0] text-secondary" />
          </div>
          <h3 className="text-lg font-semibold text-slate-100 mb-2">Upload EEG File</h3>
          <p className="text-sm text-slate-400 mb-4">Drag and drop your EDF or MAT file here or click to browse</p>

          <Button
            onClick={() => fileInputRef.current?.click()}
            className="bg-[#52c4a0] hover:bg-[#52c4a0]/90 text-slate-900 font-semibold"
          >
            Select EEG File
          </Button>
          <input ref={fileInputRef} type="file" accept=".edf,.mat" onChange={handleFileSelect} className="hidden" />
          <p className="text-xs text-slate-500 mt-3">Supported formats: EDF (real recordings), MAT (simulations)</p>
        </div>
      </Card>
    </div>
  )
}
