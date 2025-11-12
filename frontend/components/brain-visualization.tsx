"use client"

import { useEffect, useRef } from "react"

interface BrainVisualizationProps {
  plotHtml: string
}

export function BrainVisualization({ plotHtml }: BrainVisualizationProps) {
  const containerRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    let cancelled = false

    const ensurePlotDimensions = () => {
      const plotDiv = container.querySelector<HTMLElement>(".plotly-graph-div")
      if (plotDiv) {
        plotDiv.style.width = "100%"
        plotDiv.style.height = "100%"
        plotDiv.style.maxWidth = "100%"
      }

      const svg = container.querySelector<SVGElement>("svg")
      if (svg) {
        svg.setAttribute("preserveAspectRatio", "xMidYMid meet")
        svg.style.width = "100%"
        svg.style.height = "100%"
      }
    }

    const triggerPlotResize = () => {
      const plotDiv = container.querySelector<HTMLElement>(".plotly-graph-div")
      const plotly = (window as typeof window & { Plotly?: any }).Plotly

      if (plotDiv && plotly && typeof plotly.Plots?.resize === "function") {
        plotly.Plots.resize(plotDiv)
      }
    }

    const handleResize = () => {
      ensurePlotDimensions()
      triggerPlotResize()
    }

    container.style.display = "flex"
    container.style.width = "100%"
    container.style.height = "100%"
    container.style.alignItems = "stretch"

    const run = async () => {
      container.innerHTML = ""

      if (!plotHtml || cancelled) return

      const wrapper = document.createElement("div")
      wrapper.innerHTML = plotHtml

      const scripts = Array.from(wrapper.querySelectorAll("script"))
      scripts.forEach((script) => {
        script.parentNode?.removeChild(script)
      })

      while (wrapper.firstChild) {
        container.appendChild(wrapper.firstChild)
      }

      const loadScript = (oldScript: HTMLScriptElement) =>
        new Promise<void>((resolve, reject) => {
          const newScript = document.createElement("script")
          Array.from(oldScript.attributes).forEach((attr) => {
            newScript.setAttribute(attr.name, attr.value)
          })

          if (oldScript.src) {
            newScript.onload = () => resolve()
            newScript.onerror = () => reject(new Error(`Failed to load script: ${oldScript.src}`))
            newScript.src = oldScript.src
            container.appendChild(newScript)
          } else {
            newScript.textContent = oldScript.textContent
            container.appendChild(newScript)
            resolve()
          }
        })

      const scriptQueue = scripts.reduce<Promise<void>>(
        (chain, script) => chain.then(() => (!cancelled ? loadScript(script) : Promise.resolve())),
        Promise.resolve()
      )

      try {
        await scriptQueue
        ensurePlotDimensions()
        triggerPlotResize()
      } catch (error) {
        console.error(error)
      }
    }

    run()

    window.addEventListener("resize", handleResize)

    return () => {
      cancelled = true
      window.removeEventListener("resize", handleResize)
      container.innerHTML = ""
    }
  }, [plotHtml])

  return (
    <div className="w-full h-full">
      <div className="h-full min-h-[420px] rounded-xl bg-gradient-to-b from-muted/10 to-muted/30 overflow-hidden">
        <div ref={containerRef} className="w-full h-full" />
      </div>
    </div>
  )
}
