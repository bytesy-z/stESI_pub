"use client"

import { AlertCircle, RotateCcw } from "lucide-react"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"

interface ErrorAlertProps {
  message: string
  onRetry: () => void
}

export function ErrorAlert({ message, onRetry }: ErrorAlertProps) {
  return (
    <Card className="p-6 bg-destructive/5 border border-destructive/20">
      <div className="flex gap-4">
        <div className="flex-shrink-0">
          <AlertCircle className="w-6 h-6 text-destructive" />
        </div>
        <div className="flex-1">
          <h3 className="font-semibold text-foreground mb-2">Processing Error</h3>
          <p className="text-sm text-muted-foreground mb-4">{message}</p>
          <div className="flex gap-3">
            <Button onClick={onRetry} className="bg-primary hover:bg-primary/90 text-primary-foreground">
              <RotateCcw className="w-4 h-4 mr-2" />
              Try Again
            </Button>
            <Button variant="outline">View Documentation</Button>
          </div>
        </div>
      </div>
    </Card>
  )
}
