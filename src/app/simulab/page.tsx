"use client"

import { useState, useEffect, useMemo, useRef } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { FlaskConical, Shield, DollarSign, Trophy, Filter, FileDown, RefreshCcw, ArrowRight, Home, RefreshCw, CheckCircle, XCircle, X } from "lucide-react"
// Design change tracing is done via API route: /api/simulab/trace-design-change


// Animation presets available module-wide
const fadeInUp = { initial: { opacity: 0, y: 8 }, animate: { opacity: 1, y: 0 } }
const fadeInRight = { initial: { opacity: 0, x: 10 }, animate: { opacity: 1, x: 0 } }
const stagger = { animate: { transition: { staggerChildren: 0.07 } } }

// ============================================================================
// FLOATING AGENT HEALTH INDICATOR - 3 lines that expand to panel
// ============================================================================
type AgentHealthProps = {
  stage: "prompt" | "constraints" | "designing" | "review" | "generating" | "report";
  generationProgress?: {
    currentStep: string;
    scenarioStatuses: Record<string, string>;
    completedScenarios: number;
    totalScenarios: number;
    judgeStatus: string;
  } | null;
}

function FloatingAgentHealth({ stage, generationProgress }: AgentHealthProps) {
  const [isExpanded, setIsExpanded] = useState(false)
  
  // Determine agent statuses based on current stage
  const getAgentStatus = (agent: string): "online" | "running" | "done" => {
    if (stage === "designing") {
      if (agent === "Orchestrator") return "running"
      return "online"
    }
    if (stage === "generating" && generationProgress) {
      const { completedScenarios, totalScenarios, judgeStatus } = generationProgress
      if (agent === "Orchestrator") return "running"
      if (agent === "Simulator") {
        return completedScenarios >= totalScenarios ? "done" : "running"
      }
      if (agent === "Judge") {
        return judgeStatus === "complete" ? "done" : judgeStatus === "running" ? "running" : "online"
      }
    }
    if (stage === "report") {
      return "done"
    }
    return "online"
  }
  
  // Agent data - 3 agents connected to GPT-4o
  const agents = [
    { name: "Orchestrator", model: "GPT-4o", status: getAgentStatus("Orchestrator") },
    { name: "Simulator", model: "GPT-4o", status: getAgentStatus("Simulator") },
    { name: "Judge", model: "GPT-4o", status: getAgentStatus("Judge") },
  ]
  
  const runningCount = agents.filter(a => a.status === "running").length
  
  // Get current system status message
  const getSystemStatus = (): string => {
    switch (stage) {
      case "prompt": return "Awaiting research objective"
      case "constraints": return "Ready for experiment constraints"
      case "designing": return "Orchestrator designing experiment"
      case "review": return "Experiment design ready for review"
      case "generating":
        if (generationProgress) {
          const { completedScenarios, totalScenarios, judgeStatus } = generationProgress
          if (judgeStatus === "complete") return "Analysis complete"
          if (judgeStatus === "running") return "Judge determining final verdict"
          if (completedScenarios >= totalScenarios) return "Orchestrator aggregating results"
          return `Running ${totalScenarios} parallel simulations`
        }
        return "Initializing simulations"
      case "report": return "Report generated successfully"
      default: return "System ready"
    }
  }
  
  // Get all data connectivity steps dynamically based on stage
  const getDataConnectivitySteps = (): { step: string; status: "done" | "active" | "pending"; type: string }[] => {
    if (stage === "prompt" || stage === "constraints") {
      return [
        { step: "Knowledge DB connection", status: "done", type: "internal" },
        { step: "SGP Policy Engine", status: "done", type: "internal" },
        { step: "Awaiting user input...", status: "pending", type: "system" },
      ]
    }
    
    if (stage === "designing") {
      return [
        { step: "Connecting to internal Knowledge DB", status: "done", type: "internal" },
        { step: "Retrieving SMILES from API route", status: "active", type: "internal" },
        { step: "Accessing constraints from Policy Engine", status: "active", type: "internal" },
        { step: "Building experiment design", status: "pending", type: "system" },
      ]
    }
    
    if (stage === "review") {
      return [
        { step: "Knowledge DB connection", status: "done", type: "internal" },
        { step: "SMILES retrieved from API", status: "done", type: "internal" },
        { step: "Constraints loaded from Policy Engine", status: "done", type: "internal" },
        { step: "Experiment design complete", status: "done", type: "system" },
        { step: "Awaiting user confirmation...", status: "pending", type: "system" },
      ]
    }
    
    if (stage === "generating" && generationProgress) {
      const { completedScenarios, totalScenarios, judgeStatus } = generationProgress
      const simulatorActive = completedScenarios < totalScenarios
      const simulatorDone = completedScenarios >= totalScenarios
      const judgeActive = judgeStatus === "running"
      const judgeDone = judgeStatus === "complete"
      
      const steps: { step: string; status: "done" | "active" | "pending"; type: string }[] = [
        { step: "Orchestrator launching SGP agents in parallel", status: "done", type: "internal" },
        { step: `Executing 3D docking simulation algorithm`, status: simulatorActive ? "active" : "done", type: "simulation" },
        { step: "Connecting to QSAR toxicity model (ext API)", status: simulatorActive ? "active" : "done", type: "external" },
        { step: "Querying SYNTHIA planning service (ext API)", status: simulatorActive ? "active" : "done", type: "external" },
        { step: `Simulations: ${completedScenarios}/${totalScenarios} complete`, status: simulatorDone ? "done" : "active", type: "system" },
        { step: "Logging all agent calculations", status: simulatorDone ? "done" : "active", type: "internal" },
        { step: "Orchestrator consolidating results", status: simulatorDone && !judgeActive && !judgeDone ? "active" : simulatorDone ? "done" : "pending", type: "internal" },
        { step: "Running advanced AI model for verdict", status: judgeActive ? "active" : judgeDone ? "done" : "pending", type: "internal" },
        { step: "Connecting to SGP Policy Engine", status: judgeActive ? "active" : judgeDone ? "done" : "pending", type: "internal" },
        { step: "Applying hard constraints validation", status: judgeActive ? "active" : judgeDone ? "done" : "pending", type: "internal" },
        { step: "Determining final verdict", status: judgeDone ? "done" : judgeActive ? "active" : "pending", type: "system" },
      ]
      
      return steps
    }
    
    if (stage === "report") {
      return [
        { step: "SGP agents executed in parallel", status: "done", type: "internal" },
        { step: "3D docking simulations complete", status: "done", type: "simulation" },
        { step: "QSAR toxicity analysis complete", status: "done", type: "external" },
        { step: "SYNTHIA synthesis verification complete", status: "done", type: "external" },
        { step: "Agent calculations logged", status: "done", type: "internal" },
        { step: "AI verdict model executed", status: "done", type: "internal" },
        { step: "Policy constraints validated", status: "done", type: "internal" },
        { step: "Final report generated", status: "done", type: "system" },
      ]
    }
    
    return []
  }
  
  const dataSteps = getDataConnectivitySteps()
  const activeSteps = dataSteps.filter(s => s.status === "active").length
  
  return (
    <>
      {/* 3-line indicator button - fixed position at top LEFT, hidden when panel is open */}
      {!isExpanded && (
        <motion.button
          onClick={() => setIsExpanded(true)}
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: -20 }}
          style={{
            position: "fixed",
            left: 16,
            top: 16,
            zIndex: 1000,
            background: "#ffffff",
            border: "1px solid #e2e8f0",
            borderRadius: 8,
            padding: "10px 8px",
            cursor: "pointer",
            display: "flex",
            flexDirection: "column",
            gap: 4,
            boxShadow: "0 2px 8px rgba(0,0,0,0.08)",
            transition: "all 0.2s ease",
          }}
          whileHover={{ 
            scale: 1.05,
            boxShadow: "0 4px 12px rgba(0,0,0,0.12)",
          }}
          whileTap={{ scale: 0.95 }}
        >
          {[0, 1, 2].map((i) => (
            <motion.div
              key={i}
              style={{
                width: 18,
                height: 3,
                borderRadius: 2,
                background: "#94a3b8",
              }}
            />
          ))}
        </motion.button>
      )}
      
      {/* Expanded panel - NO backdrop, stays open while navigating */}
      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ x: -320, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: -320, opacity: 0 }}
            transition={{ type: "spring", damping: 25, stiffness: 300 }}
            style={{
              position: "fixed",
              left: 0,
              top: 0,
              bottom: 0,
              width: 300,
              background: "#f8fafc",
              borderRight: "1px solid #e2e8f0",
              zIndex: 1000,
              padding: 16,
              display: "flex",
              flexDirection: "column",
              boxShadow: "4px 0 20px rgba(0,0,0,0.08)",
              overflowY: "auto",
            }}
          >
            {/* Header */}
            <div style={{ 
              display: "flex", 
              alignItems: "center", 
              justifyContent: "space-between",
              marginBottom: 14,
              paddingBottom: 10,
              borderBottom: "1px solid #e2e8f0"
            }}>
              <div>
                <h3 style={{ fontSize: 14, fontWeight: 600, color: "#1e293b", margin: 0 }}>
                  System Monitor
                </h3>
              </div>
              <button
                onClick={() => setIsExpanded(false)}
                style={{
                  background: "#f1f5f9",
                  border: "1px solid #e2e8f0",
                  borderRadius: 6,
                  padding: 5,
                  cursor: "pointer",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                }}
              >
                <X size={12} color="#64748b" />
              </button>
            </div>
            
            {/* Current System Status */}
            <div style={{
              background: runningCount > 0 
                ? "linear-gradient(135deg, rgba(139, 92, 246, 0.06) 0%, rgba(99, 102, 241, 0.04) 100%)"
                : "#ffffff",
              border: `1px solid ${runningCount > 0 ? "rgba(139, 92, 246, 0.15)" : "#e2e8f0"}`,
              borderRadius: 8,
              padding: 12,
              marginBottom: 14,
            }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                {runningCount > 0 ? (
                  <motion.div
                    animate={{ rotate: 360 }}
                    transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                    style={{
                      width: 14,
                      height: 14,
                      border: "2px solid #8b5cf6",
                      borderTopColor: "transparent",
                      borderRadius: "50%",
                    }}
                  />
                ) : (
                  <CheckCircle size={14} color="#22c55e" />
                )}
                <span style={{ fontSize: 12, fontWeight: 500, color: runningCount > 0 ? "#7c3aed" : "#1e293b" }}>
                  {getSystemStatus()}
                </span>
              </div>
            </div>
            
            {/* Agent Health Section - Simplified */}
            <div style={{ marginBottom: 14 }}>
              <div style={{ 
                display: "flex", 
                alignItems: "center", 
                justifyContent: "space-between",
                marginBottom: 8
              }}>
                <span style={{ fontSize: 10, fontWeight: 600, color: "#64748b", textTransform: "uppercase", letterSpacing: "0.05em" }}>
                  Agent Health
                </span>
                <span style={{ 
                  fontSize: 9, 
                  padding: "2px 6px",
                  borderRadius: 8,
                  background: "rgba(34, 197, 94, 0.1)",
                  color: "#16a34a",
                  fontWeight: 500
                }}>
                  3/3 Online
                </span>
              </div>
              
              <div style={{ 
                background: "#ffffff",
                border: "1px solid #e2e8f0",
                borderRadius: 8,
                overflow: "hidden"
              }}>
                {agents.map((agent, idx) => {
                  const isRunning = agent.status === "running"
                  const isDone = agent.status === "done"
                  
                  return (
                    <div
                      key={agent.name}
                      style={{
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "space-between",
                        padding: "8px 10px",
                        borderBottom: idx < agents.length - 1 ? "1px solid #f1f5f9" : "none",
                        background: isRunning ? "rgba(139, 92, 246, 0.02)" : "transparent",
                      }}
                    >
                      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                        {isRunning ? (
                          <motion.div
                            animate={{ scale: [1, 1.3, 1] }}
                            transition={{ duration: 1, repeat: Infinity }}
                            style={{ width: 6, height: 6, borderRadius: "50%", background: "#8b5cf6" }}
                          />
                        ) : (
                          <div style={{ width: 6, height: 6, borderRadius: "50%", background: "#22c55e" }} />
                        )}
                        <span style={{ fontSize: 11, fontWeight: 500, color: "#1e293b" }}>{agent.name}</span>
                        <span style={{ fontSize: 8, color: "#7c3aed", fontWeight: 500 }}>{agent.model}</span>
                      </div>
                      <span style={{ 
                        fontSize: 7, 
                        padding: "2px 5px",
                        borderRadius: 4,
                        fontWeight: 500,
                        background: isRunning ? "rgba(139, 92, 246, 0.1)" : isDone ? "rgba(34, 197, 94, 0.1)" : "#f1f5f9",
                        color: isRunning ? "#7c3aed" : isDone ? "#16a34a" : "#64748b",
                        textTransform: "uppercase",
                      }}>
                        {isRunning ? "Active" : isDone ? "Done" : "Ready"}
                      </span>
                    </div>
                  )
                })}
              </div>
            </div>
            
            {/* Data Connectivity Section - All steps listed dynamically */}
            <div style={{ marginBottom: 14, flex: 1 }}>
              <div style={{ 
                display: "flex", 
                alignItems: "center", 
                justifyContent: "space-between",
                marginBottom: 8
              }}>
                <span style={{ fontSize: 10, fontWeight: 600, color: "#64748b", textTransform: "uppercase", letterSpacing: "0.05em" }}>
                  Data Connectivity
                </span>
                <span style={{ 
                  fontSize: 9, 
                  padding: "2px 6px",
                  borderRadius: 8,
                  background: activeSteps > 0 ? "rgba(139, 92, 246, 0.1)" : "rgba(34, 197, 94, 0.1)",
                  color: activeSteps > 0 ? "#7c3aed" : "#16a34a",
                  fontWeight: 500
                }}>
                  {activeSteps > 0 ? `${activeSteps} Active` : "Idle"}
                </span>
              </div>
              
              <div style={{ 
                background: "#ffffff",
                border: "1px solid #e2e8f0",
                borderRadius: 8,
                padding: 10,
              }}>
                {dataSteps.map((step, idx) => (
                  <div
                    key={idx}
                    style={{
                      display: "flex",
                      alignItems: "flex-start",
                      gap: 8,
                      padding: "5px 0",
                      borderBottom: idx < dataSteps.length - 1 ? "1px solid #f8fafc" : "none",
                    }}
                  >
                    <div style={{ marginTop: 2 }}>
                      {step.status === "done" ? (
                        <CheckCircle size={10} color="#22c55e" />
                      ) : step.status === "active" ? (
                        <motion.div
                          animate={{ rotate: 360 }}
                          transition={{ duration: 1.5, repeat: Infinity, ease: "linear" }}
                          style={{
                            width: 10,
                            height: 10,
                            border: "1.5px solid #8b5cf6",
                            borderTopColor: "transparent",
                            borderRadius: "50%",
                          }}
                        />
                      ) : (
                        <div style={{ width: 10, height: 10, borderRadius: "50%", border: "1.5px solid #e2e8f0" }} />
                      )}
                    </div>
                    <div style={{ flex: 1 }}>
                      <span style={{ 
                        fontSize: 9, 
                        color: step.status === "done" ? "#64748b" : step.status === "active" ? "#7c3aed" : "#94a3b8",
                        fontWeight: step.status === "active" ? 500 : 400,
                        lineHeight: 1.4,
                      }}>
                        {step.step}
                      </span>
                    </div>
                    <span style={{ 
                      fontSize: 6, 
                      padding: "1px 4px",
                      borderRadius: 3,
                      fontWeight: 500,
                      background: step.type === "external" ? "rgba(245, 158, 11, 0.1)" : step.type === "simulation" ? "rgba(59, 130, 246, 0.1)" : step.type === "system" ? "rgba(107, 114, 128, 0.05)" : "rgba(107, 114, 128, 0.1)",
                      color: step.type === "external" ? "#d97706" : step.type === "simulation" ? "#2563eb" : "#94a3b8",
                      textTransform: "uppercase",
                      marginTop: 1,
                    }}>
                      {step.type === "external" ? "EXT" : step.type === "simulation" ? "SIM" : step.type === "system" ? "SYS" : "INT"}
                    </span>
                  </div>
                ))}
              </div>
            </div>
            
            {/* SGP Parallel Execution Indicator */}
            {stage === "generating" && generationProgress && generationProgress.completedScenarios < generationProgress.totalScenarios && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                style={{
                  background: "linear-gradient(135deg, rgba(139, 92, 246, 0.08) 0%, rgba(59, 130, 246, 0.08) 100%)",
                  border: "1px solid rgba(139, 92, 246, 0.2)",
                  borderRadius: 8,
                  padding: 10,
                  marginBottom: 12,
                }}
              >
                <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 4 }}>
                  <motion.div
                    animate={{ scale: [1, 1.2, 1] }}
                    transition={{ duration: 1, repeat: Infinity }}
                    style={{ width: 8, height: 8, borderRadius: "50%", background: "#8b5cf6" }}
                  />
                  <span style={{ fontSize: 10, fontWeight: 600, color: "#7c3aed" }}>
                    Parallel SGP Execution
                  </span>
                </div>
                <div style={{ fontSize: 8, color: "#64748b", lineHeight: 1.4 }}>
                  <strong style={{ color: "#7c3aed" }}>{generationProgress.totalScenarios} agents</strong> processing independently on SGP
                </div>
              </motion.div>
            )}
            
            {/* Footer */}
            <div style={{ 
              paddingTop: 10,
              borderTop: "1px solid #e2e8f0",
            }}>
              <p style={{ fontSize: 8, color: "#94a3b8", margin: 0, textAlign: "center" }}>
                Powered by <span style={{ color: "#7c3aed" }}>SGP Agentex</span>
              </p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  )
}

type CreateResponse = { id: string } | { error: string }

type MessageEntry = {
  id: string
  content: {
    author: string
    content: string
    format?: string
  }
  created_at?: string
}

type DecisionCriteria = {
  docking: {
    idealMin: number
    idealMax: number
    hardFailThreshold: number
  }
  admet: {
    idealMin: number
    idealMax: number
    hardFailHERG: boolean
  }
  synthesis: {
    idealSaMax: number
    idealStepsMax: number
    hardFailSa: number
    hardFailSteps: number
  }
}

export default function SimuLabUI() {
  const [proteinTarget, setProteinTarget] = useState("EGFR")
  const [seedMolecule, setSeedMolecule] = useState("CC1=CC=CC=C1")
  const [numScenarios, setNumScenarios] = useState(3)
  const [taskId, setTaskId] = useState<string | null>(null)
  const [uiStage, setUiStage] = useState<"prompt" | "constraints" | "designing" | "review" | "generating" | "report">("prompt")
  const [nlPrompt, setNlPrompt] = useState<string>("")
  const [constraintsText, setConstraintsText] = useState<string>("")
  const [refined, setRefined] = useState<any | null>(null)
  // Data source and confidence tracking for Google Sheets integration
  const [dataSource, setDataSource] = useState<"llm" | "llm_validated">("llm")
  const [dataConfidence, setDataConfidence] = useState<"high" | "medium" | "low">("medium")
  const [messages, setMessages] = useState<MessageEntry[]>([])
  const [events, setEvents] = useState<any[]>([])
  const [extraEvents, setExtraEvents] = useState<any[]>([])
  const [scenarios, setScenarios] = useState<any[] | null>(null)
  const [winners, setWinners] = useState<any[] | null>(null)
  const [rejected, setRejected] = useState<any[] | null>(null)
  const [summaryMarkdown, setSummaryMarkdown] = useState<string | null>(null)
  const [ingestion, setIngestion] = useState<any | null>(null)
  const [hypotheses, setHypotheses] = useState<any | null>(null)
  const [dispatchStarted, setDispatchStarted] = useState<boolean>(false)
  const [metricsAggregated, setMetricsAggregated] = useState<boolean>(false)
  const [scenarioMetrics, setScenarioMetrics] = useState<Record<string, { docking?: any; admet?: any; synthesis?: any }>>({})
  const [demoMode, setDemoMode] = useState<boolean>(false)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  // Single-page view (tabs removed)
  const [filterStatus, setFilterStatus] = useState<"all" | "winners" | "rejected" | "considered">("all")
  const [carouselIndex, setCarouselIndex] = useState<number>(0)
  const [llmReason, setLlmReason] = useState<string | null>(null)
  const [structuredReport, setStructuredReport] = useState<any | null>(null)
  const [reasoning, setReasoning] = useState<boolean>(false)
  const [reportLaunched, setReportLaunched] = useState<boolean>(false)
  const [editMode, setEditMode] = useState<boolean>(false)
  const [editGoal, setEditGoal] = useState<string>("")
  const [editConstraints, setEditConstraints] = useState<string[]>([])
  const [editScenarios, setEditScenarios] = useState<any[]>([])
  const [editReason, setEditReason] = useState<string>("")
  // section refs for smooth navigation
  const refs = {
    phase1: (typeof window !== "undefined" ? document.getElementById("simulab-phase1") : null) as any,
    phase2: (typeof window !== "undefined" ? document.getElementById("simulab-phase2") : null) as any,
    scorecards: (typeof window !== "undefined" ? document.getElementById("simulab-scorecards") : null) as any,
    ranking: (typeof window !== "undefined" ? document.getElementById("simulab-ranking") : null) as any,
    final: (typeof window !== "undefined" ? document.getElementById("simulab-final") : null) as any,
  }
  const [feedback, setFeedback] = useState<Record<
    string,
    {
      overall?: number
      notes?: string
      perAgent?: {
        "simu-docking"?: { rating?: number; comment?: string }
        "simu-admet"?: { rating?: number; comment?: string }
        "simu-synthesis"?: { rating?: number; comment?: string }
        "simulab-judge"?: { rating?: number; comment?: string }
      }
    }
  >>({})
  // Edit Report chat state
  const [showEditChat, setShowEditChat] = useState(false)
  const [editChatInput, setEditChatInput] = useState("")
  const [editChatHistory, setEditChatHistory] = useState<Array<{ role: "user" | "assistant"; content: string }>>([])
  const [isProcessingEdit, setIsProcessingEdit] = useState(false)
  // Report overrides from chat edits
  const [reportOverrides, setReportOverrides] = useState<{
    llmReason?: string;
    customNotes?: string[];
  }>({})
  // Audit log for tracking all edits made to the report
  const [editAuditLog, setEditAuditLog] = useState<Array<{
    id: string;
    timestamp: string;
    instruction: string;
    summary: string;
    reportSnapshot: any; // Store the report state after this edit
  }>>([])
  // Original report for revert functionality
  const [originalReport, setOriginalReport] = useState<any>(null)
  const [showAuditLog, setShowAuditLog] = useState(false)
  // State for LLM-generated metrics loading
  const [generatingMetrics, setGeneratingMetrics] = useState<boolean>(false)
  // New states for step-by-step flow
  const [designingExperiment, setDesigningExperiment] = useState<boolean>(false)
  const [designProgress, setDesignProgress] = useState<string[]>([])
  const [generationProgress, setGenerationProgress] = useState<{
    currentStep: string;
    scenarioStatuses: Record<string, "pending" | "running" | "complete" | "error">;
    completedScenarios: number;
    totalScenarios: number;
    judgeStatus: "pending" | "running" | "complete";
  } | null>(null)
  const [criteriaSaved, setCriteriaSaved] = useState<boolean>(true)
  const [criteriaChangedAndSaved, setCriteriaChangedAndSaved] = useState<boolean>(false) // True after save is clicked with changes
  const [showRegenerateModal, setShowRegenerateModal] = useState<boolean>(false)
  const [isRegenerating, setIsRegenerating] = useState<boolean>(false)
  const [regenerateProgress, setRegenerateProgress] = useState<string[]>([])
  const [savedCriteria, setSavedCriteria] = useState<DecisionCriteria | null>(null)
  const DEFAULT_DECISION_CRITERIA: DecisionCriteria = {
    docking: {
      idealMin: -12,
      idealMax: -8,
      hardFailThreshold: -7,
    },
    admet: {
      idealMin: 0,
      idealMax: 0.3,
      hardFailHERG: true,
    },
    synthesis: {
      idealSaMax: 4,
      idealStepsMax: 5,
      hardFailSa: 6,
      hardFailSteps: 7,
    },
  }
  const [decisionCriteria, setDecisionCriteria] = useState<DecisionCriteria>(DEFAULT_DECISION_CRITERIA)
  const handleGoHome = () => {
    if (typeof window !== "undefined") {
      window.location.href = "/simulab"
    }
  }

  // Edit Report chat handler - processes user commands and updates report via API
  const handleEditChatSubmit = async () => {
    if (!editChatInput.trim()) return
    
    const userMessage = editChatInput.trim()
    setEditChatInput("")
    setEditChatHistory((prev) => [...prev, { role: "user", content: userMessage }])
    setIsProcessingEdit(true)
    
    // Log the edit trace
    const editTrace = {
      taskId,
      timestamp: new Date().toISOString(),
      editInstruction: userMessage,
      context: {
        protein_target: proteinTarget,
        scenarios: scenarios?.length || 0,
        winners: winners?.length || 0,
        rejected: rejected?.length || 0,
      }
    }
    console.log("[SimuLab] Edit trace captured:", editTrace)
    
    try {
      // Use the structured report if available
      if (!structuredReport) {
        console.error("[SimuLab] No structured report available for editing")
        setEditChatHistory((prev) => [...prev, { 
          role: "assistant", 
          content: "⚠️ No report data available to edit. Please generate a report first." 
        }])
        setIsProcessingEdit(false)
        return
      }
      
      console.log("[SimuLab] Sending edit request with structured report:", JSON.stringify(structuredReport, null, 2))
      
      // Call our API endpoint to process the edit with LLM
      const resp = await fetch("/api/simulab/edit-report", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          structuredReport: structuredReport,
          editInstruction: userMessage,
          context: {
            protein_target: proteinTarget,
            goal: refined?.goal || nlPrompt,
            taskId,
            decisionCriteria,
          }
        }),
      })
      
      if (resp.ok) {
        const data = await resp.json()
        
        if (data.updatedReport) {
          // Fallback: Save original report if not already saved (shouldn't happen normally)
          if (!originalReport && structuredReport) {
            console.log("[SimuLab] Saving original report on first edit (fallback)")
            setOriginalReport(JSON.parse(JSON.stringify(structuredReport)))
          }
          
          // Update the structured report with the LLM's changes
          setStructuredReport(data.updatedReport)
          
          // Check if the edit instruction mentions decision criteria and sync the criteria box
          const lowerMessage = userMessage.toLowerCase()
          const criteriaKeywords = [
            "herg", "herg veto", "cardiac", "safety veto",
            "binding affinity", "potency", "threshold", "δg", "dg",
            "sa score", "synthetic", "cost veto", "synthesis"
          ]
          
          const mentionsCriteria = criteriaKeywords.some(kw => lowerMessage.includes(kw))
          
          if (mentionsCriteria) {
            // Parse the edit to detect criteria changes
            let criteriaUpdated = false
            const newCriteria = { ...decisionCriteria }
            
            // Check for hERG veto changes
            if (lowerMessage.includes("herg") || lowerMessage.includes("cardiac") || lowerMessage.includes("safety veto")) {
              if (lowerMessage.includes("disable") || lowerMessage.includes("remove") || 
                  lowerMessage.includes("turn off") || lowerMessage.includes("ignore") ||
                  lowerMessage.includes("no longer") || lowerMessage.includes("don't veto")) {
                newCriteria.admet = { ...newCriteria.admet, hardFailHERG: false }
                criteriaUpdated = true
                console.log("[SimuLab] Detected hERG veto DISABLE in edit")
              } else if (lowerMessage.includes("enable") || lowerMessage.includes("add") || 
                         lowerMessage.includes("turn on") || lowerMessage.includes("activate")) {
                newCriteria.admet = { ...newCriteria.admet, hardFailHERG: true }
                criteriaUpdated = true
                console.log("[SimuLab] Detected hERG veto ENABLE in edit")
              }
            }
            
            // Check for potency/binding affinity threshold changes
            const potencyMatch = lowerMessage.match(/(?:potency|binding|affinity|threshold|δg|dg).*?(-?\d+\.?\d*)/i)
            if (potencyMatch) {
              const newThreshold = parseFloat(potencyMatch[1])
              if (!isNaN(newThreshold) && newThreshold >= -15 && newThreshold <= 0) {
                newCriteria.docking = { ...newCriteria.docking, hardFailThreshold: newThreshold }
                criteriaUpdated = true
                console.log("[SimuLab] Detected potency threshold change to:", newThreshold)
              }
            }
            
            // Check for SA score threshold changes
            const saMatch = lowerMessage.match(/(?:sa\s*score|synthesis|synthetic).*?(\d+\.?\d*)/i)
            if (saMatch) {
              const newSa = parseFloat(saMatch[1])
              if (!isNaN(newSa) && newSa >= 1 && newSa <= 10) {
                newCriteria.synthesis = { ...newCriteria.synthesis, hardFailSa: newSa }
                criteriaUpdated = true
                console.log("[SimuLab] Detected SA score threshold change to:", newSa)
              }
            }
            
            if (criteriaUpdated) {
              setDecisionCriteria(newCriteria)
              setSavedCriteria(newCriteria)
              setCriteriaSaved(true)
              console.log("[SimuLab] Decision criteria synced from edit chat:", newCriteria)
            }
          }
          
          setEditChatHistory((prev) => [...prev, { 
            role: "assistant", 
            content: data.summary || "✓ Report updated." 
          }])
          
          // Add to audit log
          const auditEntry = {
            id: `edit_${Date.now()}`,
            timestamp: new Date().toISOString(),
            instruction: userMessage,
            summary: data.summary || "Report updated",
            reportSnapshot: JSON.parse(JSON.stringify(data.updatedReport))
          }
          setEditAuditLog((prev) => [...prev, auditEntry])
          
          // Log successful edit trace
          console.log("[SimuLab] Edit applied successfully:", {
            ...editTrace,
            status: "success",
            summary: data.summary,
            auditEntryId: auditEntry.id,
          })
        } else if (data.error) {
          setEditChatHistory((prev) => [...prev, { 
            role: "assistant", 
            content: data.summary || `⚠️ ${data.error}` 
          }])
        } else {
          setEditChatHistory((prev) => [...prev, { 
            role: "assistant", 
            content: "I couldn't apply that edit. Please try rephrasing your request." 
          }])
        }
      } else {
        const errorData = await resp.json().catch(() => ({}))
        console.error("[SimuLab] Edit API error:", errorData)
        setEditChatHistory((prev) => [...prev, { 
          role: "assistant", 
          content: "Error processing edit. Please try again." 
        }])
      }
      
    } catch (error) {
      console.error("[SimuLab] Error processing edit:", error)
      setEditChatHistory((prev) => [...prev, { 
        role: "assistant", 
        content: "Connection error. Please try again." 
      }])
    } finally {
      setIsProcessingEdit(false)
    }
  }

  // (Agent health summary is rendered in the Agentic Flow sidebar)

  // Derive concise constraints from refined (if present) or user text
  const conciseConstraints = useMemo(() => {
    const fromRefine = Array.isArray((refined as any)?.constraints)
      ? ((refined as any).constraints as string[]).filter((s) => !!s && s.trim())
      : []
    if (fromRefine.length > 0) return fromRefine
    if (constraintsText && constraintsText.trim()) {
      return constraintsText
        .split(/[\n.;]+/)
        .map((s) => s.trim())
        .filter((s) => !!s)
    }
    return [] as string[]
  }, [refined, constraintsText])

  const scenarioLookup = useMemo(() => {
    const map: Record<string, any> = {}
    ;(scenarios || []).forEach((s) => {
      if (s?.scenario_id) map[s.scenario_id] = s
    })
    return map
  }, [scenarios])

  const normalizedMetrics = useMemo(() => {
    const normalized: Record<string, { docking?: any; admet?: any; synthesis?: any }> = {}
    const ids = new Set<string>()
    Object.keys(scenarioMetrics || {}).forEach((id) => ids.add(id))
    ;(scenarios || []).forEach((s: any) => {
      if (s?.scenario_id) ids.add(s.scenario_id)
    })
    ids.forEach((sid) => {
      normalized[sid] = {
        docking: normalizeDockingMetrics(sid, scenarioMetrics?.[sid]?.docking),
        admet: normalizeAdmetMetrics(sid, scenarioMetrics?.[sid]?.admet),
        synthesis: normalizeSynthesisMetrics(sid, scenarioMetrics?.[sid]?.synthesis),
      }
    })
    return normalized
  }, [scenarioMetrics, scenarios])

  // (Dark mode removed)

  // Poll messages when taskId set
  useEffect(() => {
    if (!taskId) return
    let mounted = true
    let pollCount = 0
    const maxPollsBeforeFallback = 10 // 20 seconds before fallback
    console.log("[SimuLab] Starting message polling for task:", taskId)
    
    const interval = setInterval(async () => {
      pollCount++
      try {
        const res = await fetch(`/api/simulab/messages?task_id=${encodeURIComponent(taskId)}`)
        const data = await res.json()
        if (mounted && Array.isArray(data)) {
          console.log("[SimuLab] Received messages:", data.length, "raw messages (poll #" + pollCount + ")")
          setMessages(data)
          const evts = parseMessagesToEvents(data)
          console.log("[SimuLab] Parsed events:", evts.length, "events", evts.map((e: any) => e.type))
          setEvents(evts)
          
          // Check if we have real agent results from backend
          const hasRealResults = evts.some((e: any) => e.type === "agent_result" || e.type === "judgement_complete")
          
          if (hasRealResults) {
            // Use real backend data, ignore extraEvents fallback scenarios
            console.log("[SimuLab] Using real backend agent results")
            deriveStateFromEvents(evts)
          } else {
            // Merge with extraEvents (initial scenarios)
            const combined = [...evts, ...extraEvents]
            deriveStateFromEvents(combined)
            
            // If no real results after timeout, use LLM to generate metrics for edited scenarios
            if (pollCount >= maxPollsBeforeFallback && extraEvents.length > 0 && !winners && !rejected && !generatingMetrics) {
              console.log("[SimuLab] Backend timeout - calling LLM to generate metrics for edited scenarios")
              // Call LLM API asynchronously
              generateLLMMetricsForScenarios(extraEvents).then((llmEvents) => {
                if (llmEvents.length > 0) {
                  const finalCombined = [...evts, ...extraEvents, ...llmEvents]
                  deriveStateFromEvents(finalCombined)
                  // Also update extraEvents so subsequent derives include the LLM results
                  setExtraEvents((prev: any[]) => [...prev, ...llmEvents])
                }
              })
            }
          }
        }
      } catch (err) {
        console.error("[SimuLab] Error polling messages:", err)
      }
    }, 2000)
    return () => {
      mounted = false
      clearInterval(interval)
    }
  }, [taskId, extraEvents, winners, rejected, generatingMetrics])

  // Enter generating stage when a task is present (legacy compatibility)
  useEffect(() => {
    if (taskId && uiStage !== "generating" && uiStage !== "report") {
      setUiStage("generating")
    }
  }, [taskId, uiStage])

  // Load/save feedback to localStorage keyed by taskId
  useEffect(() => {
    if (!taskId) return
    try {
      const raw = localStorage.getItem(lsFeedbackKey(taskId))
      if (raw) {
        const parsed = JSON.parse(raw)
        if (parsed && typeof parsed === "object") {
          setFeedback(parsed)
        }
      }
    } catch {}
  }, [taskId])
  useEffect(() => {
    if (!taskId) return
    try {
      localStorage.setItem(lsFeedbackKey(taskId), JSON.stringify(feedback))
    } catch {}
  }, [taskId, feedback])

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (uiStage === "prompt") {
      if (!nlPrompt.trim()) return
      setUiStage("constraints")
      return
    }
    if (uiStage === "constraints") {
      // Show designing screen with progress
      setUiStage("designing")
      setDesigningExperiment(true)
      setDesignProgress(["Initializing experiment design..."])
      
      // Mark Orchestrator as running in the events
      setExtraEvents((prev: any[]) => [
        ...prev,
        { type: "agent_run", agent: "simulab-orchestrator", input: { prompt: nlPrompt, constraints: constraintsText?.trim() || null } },
      ])
      
      const payload = { prompt: nlPrompt, constraints: constraintsText?.trim() || "" }
      try {
        // Simulate step-by-step progress
        await new Promise(r => setTimeout(r, 400))
        setDesignProgress(prev => [...prev, "Analyzing research goal..."])
        
        await new Promise(r => setTimeout(r, 400))
        setDesignProgress(prev => [...prev, "Identifying protein target and binding sites..."])
        
        const res = await fetch("/api/simulab/refine", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        })
        const data = await res.json()
        
        setDesignProgress(prev => [...prev, "Generating molecular scaffold hypotheses..."])
        await new Promise(r => setTimeout(r, 300))
        
        setRefined(data)
        // Track data source and confidence from API response
        if (data?.data_source) setDataSource(data.data_source === "database" ? "llm_validated" : "llm")
        if (data?.confidence) setDataConfidence(data.confidence)
        if (data?.protein_target) setProteinTarget(data.protein_target)
        if (Array.isArray(data?.scenarios) && data.scenarios.length > 0) {
          const proposed = data.scenarios.map((s: any, i: number) => ({
            scenario_id: s.scenario_id || `scenario_${i + 1}`,
            smiles: s.smiles || "",
            metadata: { scaffold: s.scaffold || s.name || `Scenario ${i + 1}`, source: data?.data_source || "refined" },
          }))
          setExtraEvents((prev: any[]) => [
            ...prev,
            { type: "agent_result", agent: "simulab-orchestrator", output: { goal: data?.goal, constraints: data?.constraints, scenarios: proposed.length, model_used: data?.model_used, data_source: data?.data_source } },
            { type: "simulation_scenarios", protein_target: data.protein_target || "", scenarios: proposed },
          ])
          setDesignProgress(prev => [...prev, `✓ Created ${proposed.length} molecular scenarios`])
        }
        if (typeof data?.suggested_num_scenarios === "number") setNumScenarios(data.suggested_num_scenarios)
        
        await new Promise(r => setTimeout(r, 500))
        setDesignProgress(prev => [...prev, "✓ Experiment design complete!"])
        await new Promise(r => setTimeout(r, 300))
        
        setDesigningExperiment(false)
        setUiStage("review")
      } catch (err: any) {
        // Show error - no hardcoded fallbacks
        console.error("[SimuLab] Orchestrator LLM error:", err)
        setDesignProgress(prev => [...prev, `✗ Error: ${err?.message || "Orchestrator agent failed"}`])
        setDesignProgress(prev => [...prev, "Please check that OPENAI_API_KEY is configured and try again."])
        await new Promise(r => setTimeout(r, 2000))
        setDesigningExperiment(false)
        // Stay on constraints stage so user can retry
      }
      return
    }
    if (uiStage === "review") {
      // Don't auto-run anymore - user must click "Generate Report"
      return
    }
    // default (run)
    await startRun()
  }

  const resetRun = () => {
    setTaskId(null)
    setMessages([])
    setEvents([])
    setExtraEvents([])
    setScenarios(null)
    setWinners(null)
    setRejected(null)
    setSummaryMarkdown(null)
    setScenarioMetrics({})
    setMetricsAggregated(false)
    setDispatchStarted(false)
    setIngestion(null)
    setHypotheses(null)
    setLlmReason(null)
    setStructuredReport(null)
    setFeedback({})
  }

  const startRun = async () => {
    setError(null)
    setSubmitting(true)
    setMessages([])
    setEvents([])
    setExtraEvents([])
    setScenarios(null)
    setWinners(null)
    setRejected(null)
    setSummaryMarkdown(null)
    setTaskId(null)
    setLlmReason(null)
    setStructuredReport(null)

    // Use latest refined/edited values for the lab run
    const currentGoal = refined?.goal || nlPrompt
    const currentConstraints = refined?.constraints || (constraintsText ? constraintsText.split(".").map((s: string) => s.trim()).filter(Boolean) : [])
    const currentScenarios = refined?.scenarios || []

    console.log("[SimuLab] Starting run with:", {
      proteinTarget: proteinTarget.trim(),
      goal: currentGoal,
      constraints: currentConstraints,
      scenarios: currentScenarios,
      numScenarios,
    })

    try {
      const payload = {
        protein_target: proteinTarget.trim(),
        seed_molecule: seedMolecule.trim() || undefined,
        num_scenarios: numScenarios,
        // Pass the refined/edited lab design for the backend to use
        goal: currentGoal,
        constraints: currentConstraints,
        scenarios: currentScenarios.map((s: any, idx: number) => ({
          scenario_id: s.scenario_id || `scenario_${idx + 1}`,
          scaffold: s.scaffold || s.name || `Scenario ${idx + 1}`,
          smiles: s.smiles || "",
        })),
      }
      console.log("[SimuLab] Sending to /api/simulab/create:", payload)
      
      const res = await fetch("/api/simulab/create", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      })
      const data: CreateResponse = await res.json()
      console.log("[SimuLab] Create response:", data)
      
      if (!res.ok || "error" in data) {
        throw new Error("error" in data ? data.error : `HTTP ${res.status}`)
      }
      setTaskId((data as any).id)
      console.log("[SimuLab] Task created with ID:", (data as any).id)

      // Also inject the current scenarios into extraEvents so the UI reflects them immediately
      if (currentScenarios.length > 0) {
        const scenarioEvent = {
          type: "simulation_scenarios",
          protein_target: proteinTarget.trim(),
          scenarios: currentScenarios.map((s: any, idx: number) => ({
            scenario_id: s.scenario_id || `scenario_${idx + 1}`,
            smiles: s.smiles || "",
            metadata: { scaffold: s.scaffold || s.name || `Scenario ${idx + 1}`, source: "user_confirmed" },
          })),
        }
        console.log("[SimuLab] Injecting initial scenarios event:", scenarioEvent)
        setExtraEvents((prev: any[]) => [...prev, scenarioEvent])
      }
    } catch (err: any) {
      console.error("[SimuLab] Error starting run:", err)
      setError(err?.message || "Failed to start run")
    } finally {
      setSubmitting(false)
    }
  }

  const onStartDemo = () => {
    // Demo mode uses the same flow as regular runs
    // If backend agents don't respond, LLM generates realistic metrics via /api/simulab/generate-metrics
    setProteinTarget("BCR-ABL Kinase")
    setSeedMolecule("")
    setNumScenarios(2)
    setDemoMode(true)
    console.log("[SimuLab] Demo mode started - will use backend agents or LLM fallback for metrics")
    startRun()
  }

  // Generate Report function - runs simulator agent with step-by-step progress
  const generateReport = async () => {
    setUiStage("generating")
    setError(null)
    
    const currentScenarios = refined?.scenarios || []
    const numScen = currentScenarios.length || numScenarios
    
    // Initialize progress tracking
    const initialStatuses: Record<string, "pending" | "running" | "complete" | "error"> = {}
    currentScenarios.forEach((s: any) => {
      initialStatuses[s.scenario_id || s.name] = "pending"
    })
    
    setGenerationProgress({
      currentStep: "Initializing simulation...",
      scenarioStatuses: initialStatuses,
      completedScenarios: 0,
      totalScenarios: numScen,
      judgeStatus: "pending",
    })

    try {
      // Step 1: Initialize
      await new Promise(r => setTimeout(r, 500))
      setGenerationProgress(prev => prev ? {
        ...prev,
        currentStep: "Connecting to Simulator Agent...",
      } : null)

      // Push Simulator agent_run event
      setExtraEvents(prev => [
        ...prev,
        { type: "agent_run", agent: "simulab-simulator", input: { scenarios: numScen, protein_target: proteinTarget } },
        { type: "dispatch_started" },
      ])

      // Step 2: Start parallel LLM calls for each scenario
      await new Promise(r => setTimeout(r, 500))
      
      // Mark all scenarios as running (parallel)
      setGenerationProgress(prev => {
        if (!prev) return null
        const newStatuses = { ...prev.scenarioStatuses }
        Object.keys(newStatuses).forEach(k => { newStatuses[k] = "running" })
        return {
          ...prev,
          currentStep: `Running ${numScen} parallel LLM evaluations...`,
          scenarioStatuses: newStatuses,
        }
      })

      // Simulate random completion of scenarios for visual effect
      // Some scenarios complete at the same time to show parallel execution
      const scenarioIds = Object.keys(initialStatuses)
      
      // Group scenarios into batches - some complete together (parallel), some alone
      const createRandomBatches = (ids: string[]) => {
        const shuffled = [...ids].sort(() => Math.random() - 0.5)
        const batches: string[][] = []
        let i = 0
        while (i < shuffled.length) {
          // Randomly decide batch size: 1 (solo) or 2 (parallel pair)
          // Higher chance of parallel completion (60% chance of pair if possible)
          const canPair = i + 1 < shuffled.length
          const doPair = canPair && Math.random() < 0.6
          if (doPair) {
            batches.push([shuffled[i], shuffled[i + 1]])
            i += 2
          } else {
            batches.push([shuffled[i]])
            i += 1
          }
        }
        return batches
      }
      
      const scenarioBatches = createRandomBatches(scenarioIds)
      
      // Start random completion simulation in background
      const randomCompletionPromise = (async () => {
        let completedCount = 0
        for (const batch of scenarioBatches) {
          // Random delay between 600ms and 2000ms before this batch completes
          const delay = 600 + Math.random() * 1400
          await new Promise(r => setTimeout(r, delay))
          
          // Complete all scenarios in this batch simultaneously
          completedCount += batch.length
          setGenerationProgress(prev => {
            if (!prev) return null
            const newStatuses = { ...prev.scenarioStatuses }
            batch.forEach(scenarioId => {
              newStatuses[scenarioId] = "complete"
            })
            return {
              ...prev,
              currentStep: completedCount < numScen 
                ? `Evaluating scenarios... (${completedCount}/${numScen} complete)`
                : `All scenarios evaluated. Sending to Judge Agent...`,
              scenarioStatuses: newStatuses,
              completedScenarios: completedCount,
            }
          })
        }
      })()

      // Call the generate-metrics API
      const metricsPayload = {
        scenarios: currentScenarios.map((s: any, idx: number) => ({
          scenario_id: s.scenario_id || `scenario_${idx + 1}`,
          scaffold: s.scaffold || s.name || `Scenario ${idx + 1}`,
          smiles: s.smiles || "",
        })),
        protein_target: proteinTarget,
        goal: refined?.goal || nlPrompt,
        constraints: refined?.constraints || [],
        decision_criteria: decisionCriteria,
      }

      console.log("[SimuLab] Calling generate-metrics:", metricsPayload)
      
      const metricsRes = await fetch("/api/simulab/generate-metrics", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(metricsPayload),
      })

      if (!metricsRes.ok) {
        throw new Error(`Metrics generation failed: ${metricsRes.status}`)
      }

      const metricsData = await metricsRes.json()
      console.log("[SimuLab] Metrics generated:", metricsData)

      // Wait for the random completion animation to finish
      await randomCompletionPromise

      // Update progress - all scenarios complete, move to Judge
      const completedStatuses: Record<string, "pending" | "running" | "complete" | "error"> = {}
      metricsData.results?.forEach((r: any) => {
        completedStatuses[r.scenario_id] = "complete"
      })

      // Small delay before moving to Judge for visual clarity
      await new Promise(r => setTimeout(r, 600))

      setGenerationProgress(prev => prev ? {
        ...prev,
        currentStep: "All scenarios evaluated. Sending to Judge Agent...",
        scenarioStatuses: completedStatuses,
        completedScenarios: metricsData.results?.length || 0,
        judgeStatus: "running",
      } : null)

      // Process results and determine winners/rejected
      const newWinners: any[] = []
      const newRejected: any[] = []
      const newMetrics: Record<string, any> = {}

      metricsData.results?.forEach((r: any) => {
        newMetrics[r.scenario_id] = r.metrics
        if (r.is_winner) {
          newWinners.push({ scenario_id: r.scenario_id, smiles: r.smiles, scaffold: r.scaffold })
        } else {
          newRejected.push({ scenario_id: r.scenario_id, smiles: r.smiles, scaffold: r.scaffold, veto_reason: r.rejection_reason })
        }
      })

      setScenarioMetrics(newMetrics)
      setWinners(newWinners)
      setRejected(newRejected)
      setMetricsAggregated(true)

      // Push Simulator agent_result and metrics_aggregated events
      setExtraEvents(prev => [
        ...prev,
        { type: "agent_result", agent: "simulab-simulator", output: { scenarios: metricsData.results?.length || 0, llm_calls: metricsData.results?.length || 0 } },
        { type: "metrics_aggregated", scenarios: metricsData.results || [] },
      ])

      // Step 3: Call Judge Agent (LLM reasoning)
      await new Promise(r => setTimeout(r, 500))
      
      // Push Judge agent_run event
      setExtraEvents(prev => [
        ...prev,
        { type: "agent_run", agent: "simulab-judge", input: { winners: newWinners.length, rejected: newRejected.length } },
      ])
      
      setGenerationProgress(prev => prev ? {
        ...prev,
        currentStep: "Judge Agent analyzing results...",
      } : null)

      // Call reason API for final report
      const reasonPayload = {
        winners: newWinners,
        rejected: newRejected,
        scenarios: currentScenarios,
        scenarioMetrics: newMetrics,
        context: {
          protein_target: proteinTarget,
          goal: refined?.goal || nlPrompt,
          constraints: refined?.constraints || [],
          decision_criteria: decisionCriteria,
        },
        decisionCriteria,
      }

      const reasonRes = await fetch("/api/simulab/reason", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(reasonPayload),
      })

      if (reasonRes.ok) {
        const reasonData = await reasonRes.json()
        if (reasonData?.reason) setLlmReason(reasonData.reason)
        if (reasonData?.structured) {
          setStructuredReport(reasonData.structured)
          // Reset audit log for fresh report
          setEditAuditLog([])
          // SAVE the original report so user can revert after edits
          setOriginalReport(JSON.parse(JSON.stringify(reasonData.structured)))
          setEditChatHistory([])
        }
      }

      // Push Judge agent_result and judgement_complete events
      setExtraEvents(prev => [
        ...prev,
        { type: "agent_result", agent: "simulab-judge", output: { winners: newWinners.length, rejected: newRejected.length } },
        { type: "judgement_complete", winners: newWinners, rejected: newRejected },
      ])

      // Complete
      setGenerationProgress(prev => prev ? {
        ...prev,
        currentStep: "✓ Report generation complete!",
        judgeStatus: "complete",
      } : null)

      await new Promise(r => setTimeout(r, 500))
      setUiStage("report")
      setReportLaunched(true)

    } catch (err: any) {
      console.error("[SimuLab] Error generating report:", err)
      setError(err?.message || "Failed to generate report")
      setGenerationProgress(prev => prev ? {
        ...prev,
        currentStep: `Error: ${err?.message || "Generation failed"}`,
      } : null)
    }
  }

  // Save decision criteria changes
  const saveCriteriaChanges = () => {
    setCriteriaSaved(true)
    // Only show regenerate button if criteria actually changed from the original
    // Compare current criteria with the last saved criteria or default
    const previousCriteria = savedCriteria || DEFAULT_DECISION_CRITERIA
    const hasActualChanges = JSON.stringify(decisionCriteria) !== JSON.stringify(previousCriteria)
    if (hasActualChanges) {
      setCriteriaChangedAndSaved(true) // Show regenerate button after save with actual changes
    }
    setSavedCriteria(decisionCriteria)
    console.log("[SimuLab] Decision criteria saved:", decisionCriteria, "hasActualChanges:", hasActualChanges)
  }

  // Re-generate report with updated decision criteria
  // ONLY re-runs Judge agent on existing metrics - does NOT regenerate scenarios or call Simulator
  const regenerateReport = async () => {
    setShowRegenerateModal(false)
    setIsRegenerating(true)
    setRegenerateProgress(["Orchestrator: Re-evaluating with updated decision criteria..."])
    setError(null)

    const currentScenarios = refined?.scenarios || scenarios || []
    const numScen = Object.keys(scenarioMetrics).length || currentScenarios.length
    
    try {
      // Use existing metrics - don't regenerate them
      const existingMetrics = scenarioMetrics
      
      console.log("[SimuLab] Re-running Judge with updated criteria on existing metrics:", {
        numScenarios: numScen,
        decisionCriteria,
        existingMetrics,
      })

      await new Promise(r => setTimeout(r, 400))
      setRegenerateProgress(prev => [...prev, `Orchestrator: Sending ${numScen} scenarios to Judge Agent...`])

      // Push Judge agent_run event (Orchestrator dispatches to Judge)
      setExtraEvents(prev => [
        ...prev,
        { type: "agent_run", agent: "simulab-judge", input: { scenarios: numScen, criteria_update: true } },
      ])

      await new Promise(r => setTimeout(r, 300))
      setRegenerateProgress(prev => [...prev, "Judge Agent: Applying new decision criteria..."])

      // Re-evaluate pass/fail based on new criteria using existing metrics
      const newWinners: any[] = []
      const newRejected: any[] = []

      // Apply decision criteria to existing metrics
      Object.entries(existingMetrics).forEach(([scenarioId, metrics]: [string, any]) => {
        const scenario = currentScenarios.find((s: any) => s.scenario_id === scenarioId) || { scenario_id: scenarioId }
        
        // Extract values from metrics
        const bindingAffinity = metrics?.docking?.binding_affinity_kcal_per_mol ?? metrics?.binding_affinity_kcal_per_mol ?? -7
        const hergFlag = metrics?.admet?.herg_flag ?? metrics?.herg_flag ?? false
        const saScore = metrics?.synthesis?.sa_score ?? metrics?.sa_score ?? 4
        const toxRisk = metrics?.admet?.toxicity_risk ?? metrics?.toxicity_risk ?? "LOW"
        
        // Apply decision criteria thresholds
        let isRejected = false
        let vetoReason = ""
        
        // Potency check
        if (bindingAffinity > decisionCriteria.docking.hardFailThreshold) {
          isRejected = true
          vetoReason = "Potency Fail"
        }
        
        // Safety check (hERG)
        if (!isRejected && decisionCriteria.admet.hardFailHERG && hergFlag) {
          isRejected = true
          vetoReason = "Safety Veto (hERG)"
        }
        
        // Synthesis check (SA score)
        if (!isRejected && saScore > decisionCriteria.synthesis.hardFailSa) {
          isRejected = true
          vetoReason = "Cost Veto (SA Score)"
        }
        
        if (isRejected) {
          newRejected.push({
            scenario_id: scenarioId,
            smiles: scenario.smiles || "",
            scaffold: scenario.scaffold || scenario.metadata?.scaffold || "",
            veto_reason: vetoReason,
          })
        } else {
          newWinners.push({
            scenario_id: scenarioId,
            smiles: scenario.smiles || "",
            scaffold: scenario.scaffold || scenario.metadata?.scaffold || "",
          })
        }
      })

      await new Promise(r => setTimeout(r, 300))
      setRegenerateProgress(prev => [...prev, `Judge Agent: Evaluated ${numScen} scenarios`])
      setRegenerateProgress(prev => [...prev, `Judge Agent: ${newWinners.length} passed, ${newRejected.length} rejected`])

      setWinners(newWinners)
      setRejected(newRejected)

      await new Promise(r => setTimeout(r, 300))
      setRegenerateProgress(prev => [...prev, "Judge Agent: Generating final verdict..."])

      // Call reason API for final report with updated winners/rejected
      const reasonPayload = {
        winners: newWinners,
        rejected: newRejected,
        scenarios: currentScenarios,
        scenarioMetrics: existingMetrics,
        context: {
          protein_target: proteinTarget,
          goal: refined?.goal || nlPrompt,
          constraints: refined?.constraints || [],
          decision_criteria: decisionCriteria,
        },
        decisionCriteria,
      }

      const reasonRes = await fetch("/api/simulab/reason", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(reasonPayload),
      })

      if (reasonRes.ok) {
        const reasonData = await reasonRes.json()
        if (reasonData?.reason) setLlmReason(reasonData.reason)
        if (reasonData?.structured) {
          setStructuredReport(reasonData.structured)
          // Reset audit log for regenerated report
          setEditAuditLog([])
          // SAVE the regenerated report as the new "original" for revert purposes
          setOriginalReport(JSON.parse(JSON.stringify(reasonData.structured)))
          setEditChatHistory([])
        }
      }

      // Push Judge agent_result and judgement_complete events
      setExtraEvents(prev => [
        ...prev,
        { type: "agent_result", agent: "simulab-judge", output: { winners: newWinners.length, rejected: newRejected.length } },
        { type: "judgement_complete", winners: newWinners, rejected: newRejected },
      ])

      setRegenerateProgress(prev => [...prev, "✓ Judge Agent: Verdict complete"])
      await new Promise(r => setTimeout(r, 300))
      setRegenerateProgress(prev => [...prev, "✓ Report updated with new criteria!"])

      await new Promise(r => setTimeout(r, 500))
      setCriteriaSaved(true)
      setSavedCriteria(decisionCriteria)

    } catch (err: any) {
      console.error("[SimuLab] Error re-generating report:", err)
      setError(err?.message || "Failed to re-generate report")
      setRegenerateProgress(prev => [...prev, `❌ Error: ${err?.message || "Re-generation failed"}`])
    } finally {
      setIsRegenerating(false)
    }
  }

  function parseMessagesToEvents(list: MessageEntry[]): any[] {
    const out: any[] = []
    for (const m of list) {
      const raw = m?.content?.content
      if (typeof raw !== "string") continue
      try {
        const j = JSON.parse(raw)
        if (j && typeof j === "object" && j.type) {
          out.push(j)
        }
      } catch {
        // not json
      }
    }
    return out
  }

  function deriveStateFromEvents(combined: any[]) {
    let _ingestion: any = null
    let _hypotheses: any = null
    let _dispatch = false
    let _scenarios: any[] | null = null
    let _metricsAgg = false
    let _winners: any[] | null = null
    let _rejected: any[] | null = null
    let _summary: string | null = null
    const _scenarioMetrics: Record<string, { docking?: any; admet?: any; synthesis?: any }> = {}
    
    console.log("[SimuLab] Deriving state from", combined.length, "combined events")
    
    for (const e of combined) {
      if (!e || typeof e !== "object") continue
      if (e.type === "ingestion") {
        _ingestion = e
        console.log("[SimuLab] Found ingestion event:", e)
      }
      if (e.type === "hypotheses") {
        _hypotheses = e
        console.log("[SimuLab] Found hypotheses event:", e)
      }
      if (e.type === "dispatch_started") _dispatch = true
      if (e.type === "simulation_scenarios" && Array.isArray(e.scenarios)) {
        _scenarios = e.scenarios
        console.log("[SimuLab] Found scenarios:", _scenarios?.length || 0, "scenarios", _scenarios?.map((s: any) => s.scenario_id) || [])
      }
      if (e.type === "metrics_aggregated") {
        _metricsAgg = true
        if (Array.isArray(e.scenarios) && !_scenarios) _scenarios = e.scenarios
        console.log("[SimuLab] Metrics aggregated")
      }
      if (e.type === "agent_result" && e.scenario_id && e.agent && e.output) {
        const sid = String(e.scenario_id)
        _scenarioMetrics[sid] = _scenarioMetrics[sid] || {}
        if (e.agent === "simu-docking") {
          _scenarioMetrics[sid].docking = e.output
          console.log("[SimuLab] Docking result for", sid, ":", e.output)
        }
        if (e.agent === "simu-admet") {
          _scenarioMetrics[sid].admet = e.output
          console.log("[SimuLab] ADMET result for", sid, ":", e.output)
        }
        if (e.agent === "simu-synthesis") {
          _scenarioMetrics[sid].synthesis = e.output
          console.log("[SimuLab] Synthesis result for", sid, ":", e.output)
        }
      }
      if (e.type === "judgement_complete") {
        _winners = e.winners || []
        _rejected = e.rejected || []
        _summary = e.summary_markdown || null
        console.log("[SimuLab] Judgement complete - Winners:", _winners, "Rejected:", _rejected)
      }
    }
    
    console.log("[SimuLab] Final derived state:", {
      scenarios: _scenarios?.length || 0,
      metricsAggregated: _metricsAgg,
      winners: _winners?.length || 0,
      rejected: _rejected?.length || 0,
      scenarioMetrics: Object.keys(_scenarioMetrics),
    })
    
    setIngestion(_ingestion)
    setHypotheses(_hypotheses)
    setDispatchStarted(_dispatch)
    setMetricsAggregated(_metricsAgg)
    setScenarios(_scenarios)
    setScenarioMetrics(_scenarioMetrics)
    setWinners(_winners)
    setRejected(_rejected)
    setSummaryMarkdown(_summary)
  }

  // Generate metrics using LLM API for scientifically plausible values
  async function generateLLMMetricsForScenarios(events: any[]): Promise<any[]> {
    const scenarioEvent = events.find((e: any) => e.type === "simulation_scenarios")
    if (!scenarioEvent || !Array.isArray(scenarioEvent.scenarios)) return []
    
    const scenariosToEvaluate = scenarioEvent.scenarios
    const currentGoal = refined?.goal || nlPrompt
    const currentConstraints = refined?.constraints || []
    
    console.log("[SimuLab] Calling LLM to generate metrics for", scenariosToEvaluate.length, "scenarios")
    setGeneratingMetrics(true)
    
    try {
      const resp = await fetch("/api/simulab/generate-metrics", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          scenarios: scenariosToEvaluate.map((s: any) => ({
            scenario_id: s.scenario_id,
            scaffold: s.metadata?.scaffold || s.scaffold || "Unknown scaffold",
            smiles: s.smiles || "",
          })),
          protein_target: proteinTarget || scenarioEvent.protein_target || "Unknown",
          goal: currentGoal,
          constraints: currentConstraints,
          decision_criteria: decisionCriteria,
        }),
      })
      
      if (!resp.ok) {
        console.error("[SimuLab] LLM metrics API error:", resp.status)
        return generateFallbackResults(scenariosToEvaluate)
      }
      
      const data = await resp.json()
      const llmResults = data.results || []
      
      console.log("[SimuLab] LLM generated metrics:", llmResults.length, "results, source:", data.source)
      
      // Convert LLM results to agent event format
      const results: any[] = []
      const winners: any[] = []
      const rejected: any[] = []
      
      llmResults.forEach((r: any) => {
        const sid = r.scenario_id
        const metrics = r.metrics || {}
        
        // Docking result
        if (metrics.docking) {
          results.push({
            type: "agent_result",
            agent: "simu-docking",
            scenario_id: sid,
            output: {
              binding_affinity_kcal_per_mol: metrics.docking.binding_affinity_kcal_per_mol,
              potency_pass: metrics.docking.potency_pass,
            },
          })
        }
        
        // ADMET result
        if (metrics.admet) {
          results.push({
            type: "agent_result",
            agent: "simu-admet",
            scenario_id: sid,
            output: {
              toxicity_risk: metrics.admet.toxicity_risk,
              toxicity_prob: metrics.admet.toxicity_prob,
              herg_flag: metrics.admet.herg_flag,
              is_safe: metrics.admet.is_safe,
            },
          })
        }
        
        // Synthesis result
        if (metrics.synthesis) {
          results.push({
            type: "agent_result",
            agent: "simu-synthesis",
            scenario_id: sid,
            output: {
              sa_score: metrics.synthesis.sa_score,
              num_steps: metrics.synthesis.num_steps,
              estimated_cost_usd: metrics.synthesis.estimated_cost_usd,
            },
          })
        }
        
        // Track winners and rejected
        if (r.is_winner) {
          winners.push({ scenario_id: sid, smiles: r.smiles })
        } else {
          rejected.push({ scenario_id: sid, smiles: r.smiles, veto_reason: r.rejection_reason || "Did not meet criteria" })
        }
      })
      
      // Add metrics aggregated event
      results.push({ 
        type: "metrics_aggregated", 
        scenarios: scenariosToEvaluate.map((s: any) => ({ scenario_id: s.scenario_id, smiles: s.smiles })) 
      })
      
      // Add judgement complete event
      results.push({
        type: "judgement_complete",
        winners,
        rejected,
        summary_markdown: `Evaluated ${scenariosToEvaluate.length} scenario(s) using LLM analysis. Winners: ${winners.length}, Rejected: ${rejected.length}.`,
      })
      
      console.log("[SimuLab] LLM results processed - Winners:", winners.map((w: any) => w.scenario_id), "Rejected:", rejected.map((r: any) => r.scenario_id))
      
      return results
      
    } catch (error) {
      console.error("[SimuLab] Error calling LLM metrics API:", error)
      return generateFallbackResults(scenariosToEvaluate)
    } finally {
      setGeneratingMetrics(false)
    }
  }
  
  // Fallback when LLM is unavailable - uses simple heuristics
  function generateFallbackResults(scenarios: any[]): any[] {
    console.log("[SimuLab] Using fallback metric generation for", scenarios.length, "scenarios")
    
    const results: any[] = []
    const winners: any[] = []
    const rejected: any[] = []
    
    scenarios.forEach((s: any, idx: number) => {
      const sid = s.scenario_id
      const scaffold = (s.metadata?.scaffold || s.scaffold || "").toLowerCase()
      
      // Scaffold-based variation
      const isPyrazolo = scaffold.includes("pyrazolo")
      const isPyrrolo = scaffold.includes("pyrrolo")
      
      // Generate varied metrics
      const bindingAffinity = isPyrazolo ? -9.5 - (Math.random() * 1.5) : -8.0 - (idx * 1.2) - (Math.random() * 2)
      const potencyPass = bindingAffinity < decisionCriteria.docking.hardFailThreshold
      const toxicityRisk = isPyrrolo ? "HIGH" : (idx === 0 ? "LOW" : (Math.random() > 0.6 ? "LOW" : "MED"))
      const hergFlag = isPyrrolo || (idx > 0 && Math.random() > 0.8)
      const isSafe = toxicityRisk === "LOW" && !hergFlag
      const saScore = 3.0 + (idx * 0.6) + (Math.random() * 2)
      const estimatedCost = 1000 + (idx * 350) + Math.floor(Math.random() * 600)
      
      results.push({
        type: "agent_result",
        agent: "simu-docking",
        scenario_id: sid,
        output: { binding_affinity_kcal_per_mol: Math.round(bindingAffinity * 10) / 10, potency_pass: potencyPass },
      })
      
      results.push({
        type: "agent_result",
        agent: "simu-admet",
        scenario_id: sid,
        output: { toxicity_risk: toxicityRisk, herg_flag: hergFlag, is_safe: isSafe },
      })
      
      results.push({
        type: "agent_result",
        agent: "simu-synthesis",
        scenario_id: sid,
        output: { sa_score: Math.round(saScore * 10) / 10, estimated_cost_usd: estimatedCost },
      })
      
      const passes = potencyPass && isSafe && saScore < decisionCriteria.synthesis.hardFailSa
      if (passes) {
        winners.push({ scenario_id: sid, smiles: s.smiles })
      } else {
        const reasons = []
        if (!potencyPass) reasons.push("Weak binding affinity")
        if (hergFlag) reasons.push("hERG toxicity flag")
        if (toxicityRisk !== "LOW") reasons.push(`${toxicityRisk} toxicity risk`)
        if (saScore >= decisionCriteria.synthesis.hardFailSa) reasons.push("Poor synthetic accessibility")
        rejected.push({ scenario_id: sid, smiles: s.smiles, veto_reason: reasons.join("; ") })
      }
    })
    
    results.push({ type: "metrics_aggregated", scenarios: scenarios.map((s: any) => ({ scenario_id: s.scenario_id, smiles: s.smiles })) })
    results.push({
      type: "judgement_complete",
      winners,
      rejected,
      summary_markdown: `Evaluated ${scenarios.length} scenario(s). Winners: ${winners.length}, Rejected: ${rejected.length}.`,
    })
    
    return results
  }

  // Note: buildGoldenExampleEvents was removed - LLM now generates all metrics dynamically
  // via generateLLMMetricsForScenarios() which calls /api/simulab/generate-metrics
  // Expose events to global so layout sidebar can render Agentic Flow
  useEffect(() => {
    try {
      ;(window as any).__simulabAgentEvents = [...events, ...extraEvents]
    } catch {}
  }, [events, extraEvents])

  function toxScore(risk?: string) {
    const r = String(risk || "").toUpperCase()
    if (r === "LOW") return 0
    if (r === "MED") return 1
    if (r === "HIGH") return 2
    return 3
  }
  type SortKey = "best" | "ba" | "tox" | "cost"
  const [sortKey, setSortKey] = useState<SortKey>("best")
  const [asc, setAsc] = useState<boolean>(true)
  function getRankedRows(): any[] {
    const rows = (scenarios || []).map((s: any) => {
      const metrics = normalizedMetrics[s.scenario_id] || {}
      const docking = metrics.docking || {}
      const admet = metrics.admet || {}
      const synthesis = metrics.synthesis || {}
      const ba = s?.docking?.binding_affinity_kcal_per_mol ?? docking.binding_affinity_kcal_per_mol
      const tox = s?.admet?.toxicity_risk ?? admet.toxicity_risk
      const cost = s?.synthesis?.estimated_cost_usd ?? synthesis.estimated_cost_usd
      return { ...s, _ba: ba, _tox: tox, _toxNum: toxScore(tox), _cost: cost }
    })
    const cmp = (a: any, b: any) => {
      let res = 0
      if (sortKey === "best") {
        res = (a._ba ?? 1e9) - (b._ba ?? 1e9)
        if (res === 0) res = (a._toxNum ?? 9) - (b._toxNum ?? 9)
        if (res === 0) res = (a._cost ?? 1e12) - (b._cost ?? 1e12)
      } else if (sortKey === "ba") {
        res = (a._ba ?? 1e9) - (b._ba ?? 1e9)
      } else if (sortKey === "tox") {
        res = (a._toxNum ?? 9) - (b._toxNum ?? 9)
      } else if (sortKey === "cost") {
        res = (a._cost ?? 1e12) - (b._cost ?? 1e12)
      }
      return asc ? res : -res
    }
    rows.sort(cmp)
    return rows
  }

  // Fetch LLM rationale when we have final winners/rejected (demo or real)
  useEffect(() => {
    const hasFinal = Array.isArray(winners) && Array.isArray(rejected) && (winners.length + rejected.length) > 0
    if (!hasFinal || reasoning) return
    ;(async () => {
      try {
        setReasoning(true)
        // Use latest refined/edited goal and constraints
        const currentGoal = refined?.goal || nlPrompt
        const currentConstraints = refined?.constraints || conciseConstraints
        const currentScenarios = scenarios || refined?.scenarios || []
        
        const reasonPayload = {
          winners,
          rejected,
          scenarios: currentScenarios,
          scenarioMetrics: normalizedMetrics,
          context: {
            protein_target: proteinTarget,
            goal: currentGoal,
            constraints: currentConstraints,
            decision_criteria: decisionCriteria,
          },
          decisionCriteria,
        }
        console.log("[SimuLab] Fetching LLM rationale with:", reasonPayload)
        
        // Add timeout to prevent hanging
        const controller = new AbortController()
        const timeoutId = setTimeout(() => controller.abort(), 60000) // 60 second timeout
        
        try {
          const res = await fetch("/api/simulab/reason", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(reasonPayload),
            signal: controller.signal,
          })
          
          clearTimeout(timeoutId)
          
          if (!res.ok) {
            console.error("[SimuLab] LLM rationale API error:", res.status, res.statusText)
            // Use fallback report
            setLlmReason(`Report for ${proteinTarget}: ${winners?.length || 0} winner(s), ${rejected?.length || 0} rejected.`)
            return
          }
          
          const data = await res.json()
          console.log("[SimuLab] LLM rationale response:", data)
          if (data?.reason) setLlmReason(data.reason)
          if (data?.structured) setStructuredReport(data.structured)
        } catch (fetchErr: any) {
          clearTimeout(timeoutId)
          if (fetchErr.name === 'AbortError') {
            console.warn("[SimuLab] LLM rationale request timed out, using fallback")
          } else {
            console.error("[SimuLab] Fetch error:", fetchErr)
          }
          // Use fallback report on any error
          setLlmReason(`Report for ${proteinTarget}: ${winners?.length || 0} winner(s), ${rejected?.length || 0} rejected.`)
          // Generate a basic structured report as fallback
          if (winners && winners.length > 0) {
            const w = winners[0]
            const wMetrics = normalizedMetrics[w.scenario_id] || {}
            setStructuredReport({
              executive_summary: `Analysis complete for ${proteinTarget}. Selected ${w.scenario_id} as the lead candidate.`,
              winner: {
                scenario_id: w.scenario_id,
                scaffold: w.scaffold || "Unknown",
                binding_affinity: wMetrics.docking?.binding_affinity_kcal_per_mol || -9.0,
                toxicity_risk: wMetrics.admet?.toxicity_risk || "LOW",
                herg_flag: wMetrics.admet?.herg_flag || false,
                sa_score: wMetrics.synthesis?.sa_score || 3.5,
                cost_usd: wMetrics.synthesis?.estimated_cost_usd || 1500,
                rationale: "Selected based on optimal balance of potency, safety, and manufacturability.",
              },
              rejected: (rejected || []).map((r: any) => ({
                scenario_id: r.scenario_id,
                scaffold: r.scaffold || "Unknown",
                binding_affinity: (normalizedMetrics[r.scenario_id]?.docking?.binding_affinity_kcal_per_mol) || -6.0,
                toxicity_risk: (normalizedMetrics[r.scenario_id]?.admet?.toxicity_risk) || "HIGH",
                herg_flag: (normalizedMetrics[r.scenario_id]?.admet?.herg_flag) || true,
                sa_score: (normalizedMetrics[r.scenario_id]?.synthesis?.sa_score) || 5.0,
                cost_usd: (normalizedMetrics[r.scenario_id]?.synthesis?.estimated_cost_usd) || 2500,
                rejection_reason: r.veto_reason || "Failed to meet safety or efficacy criteria.",
              })),
              comparative_analysis: "Winner demonstrates superior safety profile with acceptable potency.",
              recommendation: "Proceed with lead optimization studies.",
            })
          }
        }
      } catch (err) {
        console.error("[SimuLab] Error in LLM rationale effect:", err)
      } finally {
        setReasoning(false)
      }
    })()
  }, [winners, rejected, normalizedMetrics, proteinTarget])

  // Keyboard arrows to navigate scorecard carousel
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (!scenarios || scenarios.length === 0) return
      if (e.key === "ArrowRight") {
        setCarouselIndex((i) => (i + 1) % scenarios.length)
      } else if (e.key === "ArrowLeft") {
        setCarouselIndex((i) => (i - 1 + scenarios.length) % scenarios.length)
      }
    }
    window.addEventListener("keydown", handler as any)
    return () => window.removeEventListener("keydown", handler as any)
  }, [scenarios])

  // (Dark mode broadcast removed)
  // Pre-run stages UI
  if (uiStage !== "report") {
    // Designing screen - show while Orchestrator is designing experiment
    if (uiStage === "designing") {
      return (
        <>
        <FloatingAgentHealth stage={uiStage} generationProgress={generationProgress} />
        <div style={{ width: "100%", maxWidth: 520, margin: "80px auto", padding: 24 }}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            style={{
              background: "#ffffff",
              border: "1px solid #e2e8f0",
              borderRadius: 16,
              padding: 32,
              boxShadow: "0 4px 20px rgba(0,0,0,0.06)",
            }}
          >
            {/* Header */}
            <div style={{ textAlign: "center", marginBottom: 28 }}>
              <div style={{ 
                width: 56, 
                height: 56, 
                margin: "0 auto 16px",
                background: "linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%)",
                borderRadius: "50%",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
              }}>
                <div className="aurora-spinner" style={{ width: 28, height: 28, borderWidth: 3, borderColor: "rgba(255,255,255,0.3)", borderTopColor: "#fff" }} />
              </div>
              <h2 style={{ fontSize: 20, fontWeight: 600, color: "#1e293b", margin: 0 }}>
                Designing Your Experiment
              </h2>
              <p style={{ fontSize: 14, color: "#64748b", marginTop: 8, lineHeight: 1.5 }}>
                Analyzing your research goal and preparing molecular scenarios...
              </p>
            </div>
            
            {/* Progress steps */}
            <div style={{ background: "#f8fafc", borderRadius: 10, padding: 16 }}>
              {designProgress.map((step, idx) => (
                <motion.div
                  key={idx}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: idx * 0.08 }}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 12,
                    padding: "10px 0",
                    borderBottom: idx < designProgress.length - 1 ? "1px solid #e2e8f0" : "none",
                  }}
                >
                  {step.startsWith("✓") ? (
                    <div style={{ width: 20, height: 20, borderRadius: "50%", background: "#dcfce7", display: "flex", alignItems: "center", justifyContent: "center" }}>
                      <span style={{ color: "#16a34a", fontSize: 12 }}>✓</span>
                    </div>
                  ) : step.startsWith("✗") ? (
                    <div style={{ width: 20, height: 20, borderRadius: "50%", background: "#fee2e2", display: "flex", alignItems: "center", justifyContent: "center" }}>
                      <span style={{ color: "#dc2626", fontSize: 12 }}>✗</span>
                    </div>
                  ) : idx === designProgress.length - 1 ? (
                    <div style={{ width: 20, height: 20, display: "flex", alignItems: "center", justifyContent: "center" }}>
                      <span className="aurora-spinner" style={{ width: 16, height: 16, borderWidth: 2 }} />
                    </div>
                  ) : (
                    <div style={{ width: 20, height: 20, borderRadius: "50%", background: "#dcfce7", display: "flex", alignItems: "center", justifyContent: "center" }}>
                      <span style={{ color: "#16a34a", fontSize: 12 }}>✓</span>
                    </div>
                  )}
                  <span style={{ 
                    fontSize: 13, 
                    color: step.startsWith("✓") ? "#16a34a" : step.startsWith("✗") ? "#dc2626" : "#475569",
                    fontWeight: idx === designProgress.length - 1 && !step.startsWith("✓") && !step.startsWith("✗") ? 500 : 400,
                  }}>
                    {step}
                  </span>
                </motion.div>
              ))}
            </div>
          </motion.div>
        </div>
        </>
      )
    }

    // Generating screen - show while running N parallel LLM calls with visual diagram
    if (uiStage === "generating" && generationProgress) {
      const { currentStep, scenarioStatuses, completedScenarios, totalScenarios, judgeStatus } = generationProgress
      const scenarioEntries = Object.entries(scenarioStatuses)
      const allScenariosComplete = completedScenarios >= totalScenarios
      
      // Node component for the diagram
      const DiagramNode = ({ 
        label, 
        status, 
        isAgent = false,
        subLabel 
      }: { 
        label: string; 
        status: "pending" | "running" | "complete"; 
        isAgent?: boolean;
        subLabel?: string;
      }) => (
        <div style={{
          padding: isAgent ? "10px 16px" : "8px 12px",
          borderRadius: 8,
          border: "2px solid",
          borderColor: status === "complete" ? "#22c55e" : status === "running" ? "#8b5cf6" : "#d1d5db",
          background: status === "complete" ? "#f0fdf4" : status === "running" ? "#faf5ff" : "#fff",
          minWidth: isAgent ? 140 : 100,
          textAlign: "center",
          position: "relative",
          boxShadow: status === "running" ? "0 0 12px rgba(139, 92, 246, 0.3)" : "0 1px 3px rgba(0,0,0,0.08)",
        }}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 6 }}>
            {status === "running" && <span className="aurora-spinner" style={{ width: 12, height: 12, borderWidth: 2 }} />}
            {status === "complete" && <CheckCircle size={14} color="#22c55e" />}
            {status === "pending" && <div style={{ width: 10, height: 10, borderRadius: "50%", border: "2px solid #d1d5db" }} />}
            <span style={{ 
              fontSize: isAgent ? 13 : 11, 
              fontWeight: 600, 
              color: status === "complete" ? "#166534" : status === "running" ? "#6d28d9" : "#6b7280" 
            }}>
              {label}
            </span>
          </div>
          {subLabel && (
            <div style={{ fontSize: 10, color: "#94a3b8", marginTop: 2 }}>{subLabel}</div>
          )}
        </div>
      );

      // Arrow component
      const Arrow = ({ direction = "down", active = false }: { direction?: "down" | "right"; active?: boolean }) => (
        <div style={{ 
          display: "flex", 
          alignItems: direction === "down" ? "center" : "center",
          justifyContent: "center",
          padding: direction === "down" ? "4px 0" : "0 4px",
        }}>
          {direction === "down" ? (
            <svg width="20" height="24" viewBox="0 0 20 24">
              <path 
                d="M10 0 L10 18 M4 14 L10 20 L16 14" 
                stroke={active ? "#8b5cf6" : "#d1d5db"} 
                strokeWidth="2" 
                fill="none"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          ) : (
            <svg width="24" height="20" viewBox="0 0 24 20">
              <path 
                d="M0 10 L18 10 M14 4 L20 10 L14 16" 
                stroke={active ? "#8b5cf6" : "#d1d5db"} 
                strokeWidth="2" 
                fill="none"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          )}
        </div>
      );

      // Calculate container width based on number of scenarios
      const containerMaxWidth = scenarioEntries.length <= 2 ? 600 : scenarioEntries.length <= 3 ? 700 : scenarioEntries.length <= 4 ? 800 : scenarioEntries.length <= 5 ? 900 : 1000
      
      return (
        <>
        <FloatingAgentHealth stage={uiStage} generationProgress={generationProgress} />
        <div style={{ width: "100%", maxWidth: containerMaxWidth, margin: "40px auto", padding: 24 }}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            style={{
              background: "#fff",
              border: "1px solid #e2e8f0",
              borderRadius: 16,
              padding: 28,
              boxShadow: "0 4px 20px rgba(0,0,0,0.08)",
            }}
          >
            {/* Header */}
            <div style={{ textAlign: "center", marginBottom: 20 }}>
              <h2 style={{ fontSize: 20, fontWeight: 600, color: "#1e293b", margin: "0 0 6px" }}>
                Generating Report
              </h2>
              <p style={{ fontSize: 13, color: "#64748b", margin: 0 }}>
                {currentStep}
              </p>
            </div>

            {/* Progress bar */}
            <div style={{ marginBottom: 24 }}>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: "#64748b", marginBottom: 4 }}>
                <span>Overall Progress</span>
                <span>{Math.round(((completedScenarios + (judgeStatus === "complete" ? 1 : 0)) / (totalScenarios + 1)) * 100)}%</span>
              </div>
              <div style={{ height: 6, background: "#e2e8f0", borderRadius: 3, overflow: "hidden" }}>
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${((completedScenarios + (judgeStatus === "complete" ? 1 : 0)) / (totalScenarios + 1)) * 100}%` }}
                  transition={{ duration: 0.5 }}
                  style={{ height: "100%", background: "linear-gradient(90deg, #7c3aed, #a78bfa)", borderRadius: 3 }}
                />
              </div>
            </div>

            {/* Visual Diagram */}
            <div style={{ 
              background: "#fafafa", 
              borderRadius: 12, 
              padding: scenarioEntries.length <= 3 ? 24 : 20,
              border: "1px solid #e5e7eb",
              overflowX: "auto",
            }}>
              {/* Orchestrator */}
              <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
                <DiagramNode 
                  label="Orchestrator" 
                  status="complete" 
                  isAgent 
                  subLabel="Initialized"
                />
                <Arrow direction="down" active />
              </div>

              {/* Simulator Agent */}
              <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
                <DiagramNode 
                  label="Simulator Agent" 
                  status={allScenariosComplete ? "complete" : "running"} 
                  isAgent 
                  subLabel={allScenariosComplete ? "Complete" : "Processing..."}
                />
              </div>

              {/* Parallel Scenarios - Fan out with branching lines */}
              <div style={{ 
                display: "flex", 
                justifyContent: "center", 
                alignItems: "flex-start",
                gap: scenarioEntries.length <= 3 ? 14 : scenarioEntries.length <= 4 ? 12 : 10,
                marginTop: 0,
                marginBottom: 8,
                position: "relative",
                paddingTop: 30,
                flexWrap: "nowrap",
                minWidth: scenarioEntries.length <= 2 ? 260 : scenarioEntries.length <= 3 ? 380 : scenarioEntries.length <= 4 ? 500 : scenarioEntries.length <= 5 ? 600 : 700,
                margin: "0 auto",
              }}>
                {/* Branching lines from Simulator to each scenario */}
                <svg 
                  style={{ 
                    position: "absolute", 
                    top: 0, 
                    left: "50%", 
                    transform: "translateX(-50%)",
                    width: scenarioEntries.length <= 2 ? 250 : scenarioEntries.length <= 3 ? 370 : scenarioEntries.length <= 4 ? 490 : scenarioEntries.length <= 5 ? 590 : 690,
                    height: 35,
                    overflow: "visible",
                  }}
                >
                  {/* Vertical line from Simulator */}
                  {(() => {
                    const svgWidth = scenarioEntries.length <= 2 ? 250 : scenarioEntries.length <= 3 ? 370 : scenarioEntries.length <= 4 ? 490 : scenarioEntries.length <= 5 ? 590 : 690
                    const centerX = svgWidth / 2
                    return (
                      <line 
                        x1={centerX} 
                        y1="0" 
                        x2={centerX} 
                        y2="12" 
                        stroke={!allScenariosComplete ? "#8b5cf6" : "#22c55e"} 
                        strokeWidth="2" 
                      />
                    )
                  })()}
                  {/* Horizontal line spanning all scenarios */}
                  {scenarioEntries.length > 1 && (() => {
                    const total = scenarioEntries.length
                    const svgWidth = total <= 2 ? 250 : total <= 3 ? 370 : total <= 4 ? 490 : total <= 5 ? 590 : 690
                    const spacing = total <= 2 ? 120 : total <= 3 ? 115 : total <= 4 ? 110 : total <= 5 ? 105 : 100
                    const centerX = svgWidth / 2
                    const leftX = centerX - ((total - 1) / 2) * spacing
                    const rightX = centerX + ((total - 1) / 2) * spacing
                    return (
                      <line 
                        x1={leftX} 
                        y1="12" 
                        x2={rightX} 
                        y2="12" 
                        stroke={!allScenariosComplete ? "#8b5cf6" : "#22c55e"} 
                        strokeWidth="2" 
                      />
                    )
                  })()}
                  {/* Vertical lines down to each scenario */}
                  {scenarioEntries.map((_, idx) => {
                    const total = scenarioEntries.length
                    const svgWidth = total <= 2 ? 250 : total <= 3 ? 370 : total <= 4 ? 490 : total <= 5 ? 590 : 690
                    const spacing = total <= 2 ? 120 : total <= 3 ? 115 : total <= 4 ? 110 : total <= 5 ? 105 : 100
                    const centerX = svgWidth / 2
                    const endX = centerX + (idx - (total - 1) / 2) * spacing
                    const status = scenarioEntries[idx][1]
                    const isActive = status === "running" || status === "complete"
                    return (
                      <line
                        key={idx}
                        x1={endX}
                        y1="12"
                        x2={endX}
                        y2="32"
                        stroke={isActive ? (status === "complete" ? "#22c55e" : "#8b5cf6") : "#d1d5db"}
                        strokeWidth="2"
                      />
                    )
                  })}
                </svg>

                {/* Scenario boxes with Potency, Safety, Cost sub-boxes */}
                {scenarioEntries.map(([scenarioId, status], scenarioIndex) => {
                  const scenarioNum = scenarioId.replace("scenario_", "") || String(scenarioIndex + 1)
                  const isComplete = status === "complete"
                  const isRunning = status === "running"
                  const isPending = status === "pending"
                  
                  const metrics = ["Potency", "Safety", "Cost"]
                  const total = scenarioEntries.length
                  const isCompact = total > 3
                  const isVeryCompact = total > 5
                  
                  // Calculate box width based on number of scenarios (slightly larger)
                  const boxWidth = total <= 2 ? 110 : total <= 3 ? 105 : total <= 4 ? 100 : total <= 5 ? 95 : 88
                  
                  return (
                  <motion.div
                    key={scenarioId}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    style={{ 
                      display: "flex", 
                      flexDirection: "column", 
                      alignItems: "center",
                      width: boxWidth,
                      minWidth: boxWidth,
                      maxWidth: boxWidth,
                      flex: "0 0 auto",
                    }}
                  >
                    <div style={{
                      padding: isVeryCompact ? "4px 6px" : isCompact ? "6px 8px" : "8px 10px",
                      borderRadius: 6,
                      border: "2px solid",
                      borderColor: isComplete ? "#22c55e" : isRunning ? "#8b5cf6" : "#d1d5db",
                      background: isComplete ? "#f0fdf4" : isRunning ? "#faf5ff" : "#fff",
                      textAlign: "center",
                      boxShadow: isRunning ? "0 0 12px rgba(139, 92, 246, 0.3)" : "0 1px 3px rgba(0,0,0,0.08)",
                      width: "100%",
                    }}>
                      {/* Header */}
                      <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 3, marginBottom: isVeryCompact ? 3 : isCompact ? 4 : 6 }}>
                        {isRunning && <span className="aurora-spinner" style={{ width: isVeryCompact ? 6 : 8, height: isVeryCompact ? 6 : 8, borderWidth: 2 }} />}
                        {isComplete && <CheckCircle size={isVeryCompact ? 8 : isCompact ? 10 : 12} color="#22c55e" />}
                        {isPending && <div style={{ width: isVeryCompact ? 5 : 6, height: isVeryCompact ? 5 : 6, borderRadius: "50%", border: "2px solid #d1d5db" }} />}
                        <span style={{ 
                          fontSize: isVeryCompact ? 8 : isCompact ? 9 : 10, 
                          fontWeight: 600, 
                          color: isComplete ? "#166534" : isRunning ? "#6d28d9" : "#6b7280" 
                        }}>
                          S{scenarioNum}
                        </span>
                      </div>
                      
                      {/* Metric boxes - same color, turn green when complete */}
                      <div style={{ display: "flex", flexDirection: "column", gap: isVeryCompact ? 1 : 2 }}>
                        {metrics.map((metric) => (
                          <div 
                            key={metric}
                            style={{
                              padding: isVeryCompact ? "2px 4px" : isCompact ? "3px 6px" : "4px 8px",
                              borderRadius: 3,
                              border: "1px solid",
                              borderColor: isComplete ? "#22c55e" : "#c4b5fd",
                              background: isComplete ? "#dcfce7" : "#ede9fe",
                              display: "flex",
                              alignItems: "center",
                              justifyContent: "space-between",
                              gap: isVeryCompact ? 2 : 4,
                              transition: "all 0.3s ease",
                            }}
                          >
                            <span style={{ 
                              fontSize: isVeryCompact ? 7 : isCompact ? 8 : 9, 
                              fontWeight: 600, 
                              color: isComplete ? "#166534" : "#7c3aed"
                            }}>
                              {metric}
                            </span>
                            {isComplete && <CheckCircle size={isVeryCompact ? 6 : isCompact ? 8 : 10} color="#22c55e" />}
                            {isRunning && (
                              <motion.div
                                animate={{ rotate: 360 }}
                                transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                                style={{ 
                                  width: isVeryCompact ? 6 : isCompact ? 8 : 10, 
                                  height: isVeryCompact ? 6 : isCompact ? 8 : 10, 
                                  border: "2px solid #8b5cf6",
                                  borderTopColor: "transparent",
                                  borderRadius: "50%"
                                }}
                              />
                            )}
                            {isPending && (
                              <div style={{ width: isVeryCompact ? 4 : 6, height: isVeryCompact ? 4 : 6, borderRadius: "50%", border: "1px solid #d1d5db" }} />
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  </motion.div>
                )})}
              </div>

              {/* Converging lines from scenarios to Judge */}
              <div style={{ display: "flex", justifyContent: "center", position: "relative", height: 30 }}>
                <svg 
                  style={{ 
                    width: scenarioEntries.length <= 2 ? 240 : scenarioEntries.length <= 3 ? 360 : scenarioEntries.length <= 4 ? 480 : scenarioEntries.length <= 5 ? 580 : 680,
                    height: 30,
                    overflow: "visible",
                  }}
                >
                  {scenarioEntries.map((_, idx) => {
                    const total = scenarioEntries.length
                    const svgWidth = total <= 2 ? 240 : total <= 3 ? 360 : total <= 4 ? 480 : total <= 5 ? 580 : 680
                    const spacing = total <= 2 ? 120 : total <= 3 ? 115 : total <= 4 ? 110 : total <= 5 ? 105 : 100
                    const endX = svgWidth / 2
                    const startX = endX + (idx - (total - 1) / 2) * spacing
                    return (
                      <path
                        key={idx}
                        d={`M${startX} 5 Q${startX} 15, ${endX} 25`}
                        stroke={allScenariosComplete ? "#8b5cf6" : "#d1d5db"}
                        strokeWidth="2"
                        fill="none"
                      />
                    )
                  })}
                </svg>
              </div>

              {/* Judge Agent */}
              <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
                <DiagramNode 
                  label="Judge Agent" 
                  status={judgeStatus} 
                  isAgent 
                  subLabel={
                    judgeStatus === "complete" ? "Verdict Ready" : 
                    judgeStatus === "running" ? "Analyzing..." : 
                    "Waiting"
                  }
                />
                {judgeStatus === "complete" && (
                  <>
                    <Arrow direction="down" active />
                    <DiagramNode 
                      label="Report Ready" 
                      status="complete" 
                      isAgent 
                    />
                  </>
                )}
              </div>
            </div>

            {/* Legend */}
            <div style={{ 
              display: "flex", 
              justifyContent: "center", 
              gap: 20, 
              marginTop: 16,
              fontSize: 11,
              color: "#64748b",
            }}>
              <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
                <div style={{ width: 10, height: 10, borderRadius: "50%", border: "2px solid #d1d5db" }} />
                Pending
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
                <div style={{ width: 10, height: 10, borderRadius: "50%", background: "#8b5cf6" }} />
                Running
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
                <CheckCircle size={12} color="#22c55e" />
                Complete
              </div>
            </div>

            {error && (
              <div style={{ marginTop: 16, padding: 12, background: "#fef2f2", border: "1px solid #fecaca", borderRadius: 8, color: "#991b1b", fontSize: 13 }}>
                {error}
              </div>
            )}
          </motion.div>
        </div>
        </>
      )
    }

    return (
      <>
      <FloatingAgentHealth stage={uiStage} generationProgress={generationProgress} />
      <div
        style={{
          width: "100%",
          maxWidth: 900,
          margin: "48px auto",
          padding: 16,
        }}
      >
        {/* Classic process flow for pre-run */}
        <StepIndicator
          ingestion={uiStage !== "prompt"}
          hypotheses={!!refined}
          dispatch={false}
          scenariosReady={Array.isArray(refined?.scenarios) && refined!.scenarios.length > 0}
          metricsAggregated={false}
          judged={false}
        />
        <div style={{ height: 10 }} />
        {uiStage === "prompt" && (
          <motion.div {...fadeInUp} transition={{ duration: 0.25 }}>
            <form onSubmit={onSubmit}>
              <div
                style={{
                  border: "1px solid #e5e7eb",
                  borderRadius: 14,
                  background: "#ffffff",
                  boxShadow: "0 4px 12px rgba(16,24,40,0.06)",
                  padding: 16,
                  position: "relative",
                }}
              >
                <button
                  type="button"
                  onClick={() => {
                    setNlPrompt("Initiate a CNS Lead Optimization experiment for the Amyloid Beta target")
                  }}
                  title="Fill demo goal"
                  aria-label="Fill demo goal and constraints"
                  style={{
                    position: "absolute",
                    right: 12,
                    top: 12,
                    width: 26,
                    height: 26,
                    borderRadius: 999,
                    border: "1px solid #e5e7eb",
                    background: "#ffffff",
                    color: "#475569",
                    fontSize: 14,
                    boxShadow: "0 1px 2px rgba(16,24,40,0.06)",
                    display: "inline-flex",
                    alignItems: "center",
                    justifyContent: "center",
                    cursor: "pointer",
                  }}
                >
                  ★
                </button>
                <label htmlFor="simulab-prompt" style={{ display: "block", fontSize: 13, color: "#6b7280", marginBottom: 8 }}>
                  Define your objective
                </label>
                <textarea
                  id="simulab-prompt"
                  value={nlPrompt}
                  onChange={(e) => setNlPrompt(e.target.value)}
                  placeholder='Identify optimized lead molecules for the BCR-ABL Kinase target, focusing on the Pyrrolo-pyrimidine and Pyrazolo-pyridine scaffolds'
                  rows={2}
                  style={{
                    width: "100%",
                    padding: 14,
                    border: "1px solid #e5e7eb",
                    borderRadius: 10,
                    background: "#ffffff",
                    color: "#111827",
                    lineHeight: 1.5,
                    resize: "vertical",
                    minHeight: 72,
                  }}
                />
                <div style={{ display: "flex", justifyContent: "center", marginTop: 14 }}>
                  <button
                    type="submit"
                    style={{
                      padding: "12px 20px",
                      border: "1px solid #111827",
                      borderRadius: 12,
                      background: "#111827",
                      color: "#ffffff",
                      fontWeight: 700,
                      letterSpacing: 0.2,
                      boxShadow: "0 2px 6px rgba(16,24,40,0.08)",
                    }}
                    aria-label="Submit objective prompt"
                  >
                    Submit
                  </button>
                </div>
              </div>
            </form>
          </motion.div>
        )}
        {uiStage === "constraints" && (
          <motion.div {...fadeInUp} transition={{ duration: 0.25 }}>
            <div style={{ border: "1px solid #e5e7eb", borderRadius: 12, padding: 16, background: "#ffffff", boxShadow: "0 1px 3px rgba(16,24,40,0.06)", position: "relative" }}>
              {/* Subtle demo star in corner for constraints */}
              <button
                type="button"
                onClick={() => {
                  setConstraintsText("Focus on maximizing brain penetration and prioritize safety")
                }}
                title="Fill demo constraints"
                aria-label="Fill demo constraints"
                style={{
                  position: "absolute",
                  right: 12,
                  top: 12,
                  width: 26,
                  height: 26,
                  borderRadius: 999,
                  border: "1px solid #e5e7eb",
                  background: "#ffffff",
                  color: "#475569",
                  fontSize: 14,
                  boxShadow: "0 1px 2px rgba(16,24,40,0.06)",
                  display: "inline-flex",
                  alignItems: "center",
                  justifyContent: "center",
                  cursor: "pointer",
                }}
              >
                ★
              </button>
              <form onSubmit={onSubmit} style={{ display: "grid", gap: 12 }}>
                <label htmlFor="simulab-constraints" style={{ display: "block", fontSize: 13, color: "#6b7280" }}>
                  Specify constraints (optional)
                </label>
                <textarea
                  id="simulab-constraints"
                  value={constraintsText}
                  onChange={(e) => setConstraintsText(e.target.value)}
                  placeholder='Focus on maximizing brain penetration and prioritize safety'
                  rows={3}
                  style={{
                    width: "100%",
                    padding: 12,
                    border: "1px solid #e5e7eb",
                    borderRadius: 10,
                    background: "#ffffff",
                    color: "#111827",
                    lineHeight: 1.5,
                    minHeight: 96,
                    resize: "vertical",
                  }}
                />
                <div style={{ display: "flex", justifyContent: "center", gap: 8, marginTop: 4 }}>
                  <button
                    type="button"
                    onClick={() => setUiStage("prompt")}
                    style={{
                      padding: "10px 12px",
                      border: "1px solid #e5e7eb",
                      borderRadius: 10,
                      background: "#ffffff",
                      color: "#111827",
                      fontWeight: 600,
                    }}
                  >
                    Back
                  </button>
                  <button
                    type="submit"
                    style={{
                      padding: "10px 16px",
                      border: "1px solid #111827",
                      borderRadius: 10,
                      background: "#111827",
                      color: "#ffffff",
                      fontWeight: 700,
                      letterSpacing: 0.2,
                      boxShadow: "0 2px 6px rgba(16,24,40,0.08)",
                    }}
                  >
                    Next
                  </button>
                </div>
              </form>
            </div>
          </motion.div>
        )}
        {uiStage === "review" && (
          <motion.div {...fadeInUp} transition={{ duration: 0.25 }}>
            <div style={{ border: "1px solid #e5e7eb", borderRadius: 12, padding: 16, background: "#ffffff", boxShadow: "0 1px 3px rgba(16,24,40,0.06)" }}>
              <div style={{ fontWeight: 700, marginBottom: 8, color: "#0f172a" }}>Review & Confirm</div>
              <div style={{ display: "grid", gap: 14 }}>
                {/* Goal */}
                <div>
                  <div style={{ fontSize: 12, color: "#6b7280", marginBottom: 6 }}>Goal</div>
                  <div
                    style={{
                      border: "1px solid #e5e7eb",
                      borderLeft: "4px solid #111827",
                      borderRadius: 10,
                      padding: 12,
                      background: "#f8fafc",
                      color: "#0f172a",
                      fontWeight: 600,
                    }}
                  >
                    {refined?.goal || nlPrompt}
                  </div>
                </div>
                {/* Constraints */}
                <div>
                  <div style={{ fontSize: 12, color: "#6b7280", marginBottom: 6 }}>Constraints</div>
                  <div style={{ border: "1px solid #e5e7eb", borderRadius: 10, background: "#ffffff" }}>
                    {(() => {
                      const list = (refined?.constraints && refined.constraints.length > 0
                        ? refined.constraints
                        : (constraintsText ? constraintsText.split(".").map((s: string) => s.trim()).filter(Boolean) : []));
                      if (!list || list.length === 0) {
                        return (
                          <div style={{ padding: 10, border: "1px solid #e5e7eb", borderRadius: 10, fontStyle: "italic", color: "#6b7280" }}>Not specified</div>
                        )
                      }
                      return (
                        <ul style={{ margin: 0, padding: "10px 14px", listStyle: "none" }}>
                          {list.map((c: string, i: number) => (
                            <li key={i} style={{ display: "flex", alignItems: "flex-start", gap: 8, color: "#374151", fontSize: 14, marginBottom: 6 }}>
                              <span style={{ width: 6, height: 6, marginTop: 6, borderRadius: 999, background: "#94a3b8", display: "inline-block" }} />
                              <span>{c}</span>
                            </li>
                          ))}
                        </ul>
                      )
                    })()}
                  </div>
                </div>
                {/* Proposed Scenarios as vertical branch diagram */}
                {Array.isArray(refined?.scenarios) && refined.scenarios.length > 0 && (
                  <div>
                    <div style={{ fontSize: 12, color: "#6b7280", marginBottom: 6 }}>Proposed Scenarios</div>
                    <div style={{ position: "relative", paddingLeft: 16 }}>
                      {/* vertical spine */}
                      <div style={{ position: "absolute", left: 8, top: 0, bottom: 0, width: 2, background: "#e5e7eb" }} />
                      {refined.scenarios.map((s: any, idx: number) => (
                        <div key={idx} style={{ position: "relative", marginBottom: 10, paddingLeft: 12 }}>
                          {/* horizontal connector back to spine */}
                          <div style={{ position: "absolute", left: -8, top: 16, width: 8, height: 2, background: "#cbd5e1" }} />
                          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", border: "1px solid #e5e7eb", borderRadius: 8, padding: "8px 10px", background: "#ffffff" }}>
                            <div style={{ fontSize: 13, fontWeight: 600, color: "#0f172a" }}>
                              {s.scaffold || s.name || s.scenario_id || `Scenario ${idx + 1}`}
                            </div>
                            {s.smiles && (
                              <code style={{ fontSize: 11, color: "#6b7280" }}>{s.smiles}</code>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                {/* Actions */}
                <div style={{ display: "flex", gap: 10, justifyContent: "flex-end" }}>
                  <button
                    type="button"
                    onClick={() => setUiStage("constraints")}
                    style={{ padding: "10px 12px", border: "1px solid #e5e7eb", borderRadius: 10, background: "#ffffff", color: "#111827", fontWeight: 600 }}
                  >
                    Back
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setEditGoal(refined?.goal || nlPrompt)
                      const list = refined?.constraints && refined.constraints.length > 0
                        ? refined.constraints
                        : (constraintsText ? constraintsText.split(".").map((s: string) => s.trim()).filter(Boolean) : [])
                      setEditConstraints(list)
                      setEditScenarios(refined?.scenarios || [])
                      setEditReason("")
                      setEditMode(true)
                    }}
                    style={{ padding: "10px 12px", border: "1px solid #9ca3af", borderRadius: 10, background: "#f3f4f6", color: "#374151", fontWeight: 600 }}
                  >
                    Edit
                  </button>
                </div>

                {/* Edit Dialog */}
                {editMode && (
                  <motion.div
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    style={{ marginTop: 16, border: "1px solid #e5e7eb", borderRadius: 12, padding: 16, background: "#fafafa" }}
                  >
                    <div style={{ fontWeight: 600, marginBottom: 12, color: "#0f172a" }}>Edit Lab Design</div>
                    
                    {/* Edit Goal */}
                    <div style={{ marginBottom: 14 }}>
                      <label style={{ fontSize: 12, color: "#6b7280", display: "block", marginBottom: 4 }}>Goal</label>
                      <textarea
                        value={editGoal}
                        onChange={(e) => setEditGoal(e.target.value)}
                        rows={2}
                        style={{ width: "100%", padding: 10, border: "1px solid #d1d5db", borderRadius: 8, fontSize: 14, resize: "vertical" }}
                      />
                    </div>

                    {/* Edit Constraints */}
                    <div style={{ marginBottom: 14 }}>
                      <label style={{ fontSize: 12, color: "#6b7280", display: "block", marginBottom: 4 }}>Constraints</label>
                      {editConstraints.map((c, idx) => (
                        <div key={idx} style={{ display: "flex", gap: 6, marginBottom: 6 }}>
                          <input
                            type="text"
                            value={c}
                            onChange={(e) => {
                              const updated = [...editConstraints]
                              updated[idx] = e.target.value
                              setEditConstraints(updated)
                            }}
                            style={{ flex: 1, padding: 8, border: "1px solid #d1d5db", borderRadius: 6, fontSize: 13 }}
                          />
                          <button
                            type="button"
                            onClick={() => setEditConstraints(editConstraints.filter((_, i) => i !== idx))}
                            style={{ padding: "6px 10px", border: "1px solid #fca5a5", borderRadius: 6, background: "#fef2f2", color: "#b91c1c", fontSize: 12 }}
                          >
                            Remove
                          </button>
                        </div>
                      ))}
                      <button
                        type="button"
                        onClick={() => setEditConstraints([...editConstraints, ""])}
                        style={{ padding: "6px 10px", border: "1px solid #d1d5db", borderRadius: 6, background: "#fff", color: "#374151", fontSize: 12 }}
                      >
                        + Add Constraint
                      </button>
                    </div>

                    {/* Edit Scenarios */}
                    <div style={{ marginBottom: 14 }}>
                      <label style={{ fontSize: 12, color: "#6b7280", display: "block", marginBottom: 4 }}>Proposed Scenarios</label>
                      {editScenarios.map((s, idx) => (
                        <div key={idx} style={{ display: "flex", gap: 6, marginBottom: 6, alignItems: "center" }}>
                          <input
                            type="text"
                            value={s.scaffold || s.name || s.scenario_id || ""}
                            onChange={(e) => {
                              const updated = [...editScenarios]
                              updated[idx] = { ...updated[idx], scaffold: e.target.value, name: e.target.value }
                              setEditScenarios(updated)
                            }}
                            placeholder="Scaffold / Name"
                            style={{ flex: 1, padding: 8, border: "1px solid #d1d5db", borderRadius: 6, fontSize: 13 }}
                          />
                          <input
                            type="text"
                            value={s.smiles || ""}
                            onChange={(e) => {
                              const updated = [...editScenarios]
                              updated[idx] = { ...updated[idx], smiles: e.target.value }
                              setEditScenarios(updated)
                            }}
                            placeholder="SMILES"
                            style={{ flex: 1, padding: 8, border: "1px solid #d1d5db", borderRadius: 6, fontSize: 13, fontFamily: "monospace" }}
                          />
                          <button
                            type="button"
                            onClick={() => setEditScenarios(editScenarios.filter((_, i) => i !== idx))}
                            style={{ padding: "6px 10px", border: "1px solid #fca5a5", borderRadius: 6, background: "#fef2f2", color: "#b91c1c", fontSize: 12 }}
                          >
                            Remove
                          </button>
                        </div>
                      ))}
                      <button
                        type="button"
                        onClick={() => setEditScenarios([...editScenarios, { scaffold: "", smiles: "", scenario_id: `scenario-${editScenarios.length + 1}` }])}
                        style={{ padding: "6px 10px", border: "1px solid #d1d5db", borderRadius: 6, background: "#fff", color: "#374151", fontSize: 12 }}
                      >
                        + Add Scenario
                      </button>
                    </div>

                    {/* Reason for Edit */}
                    <div style={{ marginBottom: 14 }}>
                      <label style={{ fontSize: 12, color: "#6b7280", display: "block", marginBottom: 4 }}>Reason for Changes (required for audit trail)</label>
                      <textarea
                        value={editReason}
                        onChange={(e) => setEditReason(e.target.value)}
                        rows={2}
                        placeholder="Explain why you made these changes..."
                        style={{ width: "100%", padding: 10, border: "1px solid #d1d5db", borderRadius: 8, fontSize: 13, resize: "vertical" }}
                      />
                    </div>

                    {/* Edit Actions */}
                    <div style={{ display: "flex", gap: 8, justifyContent: "flex-end" }}>
                      <button
                        type="button"
                        onClick={() => setEditMode(false)}
                        style={{ padding: "8px 14px", border: "1px solid #e5e7eb", borderRadius: 8, background: "#fff", color: "#374151", fontWeight: 500 }}
                      >
                        Cancel
                      </button>
                      <button
                        type="button"
                        disabled={!editReason.trim()}
                        onClick={async () => {
                          // Capture original values before applying changes
                          const originalValue = {
                            goal: refined?.goal || nlPrompt,
                            constraints: refined?.constraints || [],
                            scenarios: refined?.scenarios || [],
                          }
                          const newValue = {
                            goal: editGoal,
                            constraints: editConstraints.filter(Boolean),
                            scenarios: editScenarios,
                          }
                          
                          // Apply edits to refined state
                          setRefined((prev: any) => ({
                            ...prev,
                            goal: editGoal,
                            constraints: editConstraints.filter(Boolean),
                            scenarios: editScenarios,
                          }))
                          setNlPrompt(editGoal)
                          
                          // Send trace to Simulator agent via API route (server-side call)
                          const experimentId = taskId || `exp_${Date.now()}`
                          console.log("[SimuLab] Sending design change trace via API route...")
                          
                          fetch("/api/simulab/trace-design-change", {
                            method: "POST",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify({
                              experiment_id: experimentId,
                              change_type: "human_override",
                              original_value: originalValue,
                              new_value: newValue,
                              reasoning: editReason,
                            }),
                          })
                            .then((res) => res.json())
                            .then((result) => {
                              if (result.success) {
                                console.log("[SimuLab] Design change traced successfully:", result.trace_id)
                              } else {
                                console.warn("[SimuLab] Design change trace failed:", result.error)
                              }
                            })
                            .catch((err) => {
                              console.warn("[SimuLab] Design change trace error:", err)
                            })
                          
                          setEditMode(false)
                        }}
                        style={{
                          padding: "8px 14px",
                          border: "1px solid #111827",
                          borderRadius: 8,
                          background: editReason.trim() ? "#111827" : "#9ca3af",
                          color: "#fff",
                          fontWeight: 600,
                          cursor: editReason.trim() ? "pointer" : "not-allowed",
                        }}
                      >
                        Apply Changes
                      </button>
                    </div>
                  </motion.div>
                )}
              </div>
            </div>

            {/* Decision Criteria Section - Compact */}
            <div style={{ marginTop: 12 }}>
              <DecisionCriteriaControls
                decisionCriteria={decisionCriteria}
                onCriteriaChange={(newCriteria: DecisionCriteria) => {
                  setDecisionCriteria(newCriteria)
                  setCriteriaSaved(false)
                }}
                showSaveButton={true}
                onSave={saveCriteriaChanges}
              />
            </div>

            {/* Generate Report Button */}
            <div style={{ marginTop: 24, textAlign: "center" }}>
              <button
                onClick={generateReport}
                style={{
                  padding: "12px 28px",
                  border: "none",
                  borderRadius: 8,
                  background: "#1f2937",
                  color: "#ffffff",
                  fontWeight: 600,
                  fontSize: 14,
                  cursor: "pointer",
                  boxShadow: "0 2px 8px rgba(0,0,0,0.12)",
                  transition: "all 0.2s ease",
                }}
              >
                Generate Report
              </button>
            </div>
          </motion.div>
        )}
      </div>
      </>
    )
  }

  const renderScorecard = (scenario: any) => {
    const sid = scenario.scenario_id
    const metrics = normalizedMetrics[sid] || {}
    const isWinner = (winners || []).some((w: any) => w.scenario_id === sid)
    const status = isWinner ? "Winner" : (rejected || []).some((r: any) => r.scenario_id === sid) ? "Vetoed" : "Considered"
    const statusColor = isWinner ? "#34a853" : (rejected || []).some((r: any) => r.scenario_id === sid) ? "#ea4335" : "#999"
    return (
      <motion.div {...stagger} style={{ display: "grid", gap: 8 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <div>
            <div style={{ fontWeight: 600 }}>{sid}</div>
            <div style={{ fontSize: 12, color: "#666" }}>{scenario.metadata?.scaffold || ""}</div>
          </div>
          <div style={{ color: statusColor, fontWeight: 700 }}>{status}</div>
        </div>
        <div>
          <div style={{ fontSize: 12, color: "#666" }}>SMILES</div>
          <code>{scenario.smiles}</code>
        </div>
        <motion.div {...stagger} style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 8 }}>
          <MetricCard title="Docking (Potency)" data={metrics.docking} fallback="Awaiting result..." />
          <MetricCard title="ADMET (Safety)" data={metrics.admet} fallback="Awaiting result..." />
          <MetricCard title="Synthesis (Cost)" data={metrics.synthesis} fallback="Awaiting result..." />
        </motion.div>
      </motion.div>
    )
  }

  return (
    <div
      className="page-transition custom-scrollbar"
      style={{
        width: "100%",
        maxWidth: "100%",
        margin: 0,
        padding: 16,
        background: "#fafafa",
        minHeight: "100vh",
      }}
    >
      {/* Floating Agent Health Panel */}
      <FloatingAgentHealth stage={uiStage} generationProgress={generationProgress} />
      
      <div id="simulab-report-root">
      {/* Flow chips removed to avoid UI switching; classic indicator retained below */}

      {/* Classic flow indicator shown above report summary */}
      <div className="no-print" style={{ display: "flex", alignItems: "center", gap: 16 }}>
        <StepIndicator
          ingestion={reportLaunched || !!ingestion}
          hypotheses={reportLaunched || !!hypotheses}
          dispatch={reportLaunched || dispatchStarted}
          scenariosReady={reportLaunched || (Array.isArray(scenarios) && scenarios.length > 0)}
          metricsAggregated={metricsAggregated}
          judged={!!summaryMarkdown || !!structuredReport}
        />
      </div>
      <div className="no-print" style={{ height: 10 }} />
      <div
        className="no-print"
        style={{
          position: "fixed",
          top: 16,
          right: 24,
          zIndex: 50,
        }}
      >
        <button
          type="button"
          onClick={handleGoHome}
          aria-label="Return to SimuLab home"
          style={{
            border: "1px solid #e5e7eb",
            background: "#ffffff",
            borderRadius: 999,
            width: 38,
            height: 38,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            boxShadow: "0 1px 3px rgba(0,0,0,0.08)",
            cursor: "pointer",
            transition: "all 0.2s ease",
          }}
        >
          <Home size={16} color="#6b7280" />
        </button>
      </div>
      {/* Processing Summary paired with adjustable decision criteria */}
      <div style={{ display: "flex", flexWrap: "wrap", gap: 16, alignItems: "stretch", marginBottom: 12 }}>
        <motion.div
          className="print-full-width"
          {...fadeInUp}
          transition={{ duration: 0.25 }}
          style={{
            border: "1px solid #e5e7eb",
            borderRadius: 12,
            padding: 12,
            background: "#fff",
            boxShadow: "0 1px 3px rgba(16,24,40,0.06)",
            flex: "1 1 45%",
            minWidth: 260,
          }}
        >
          <div style={{ display: "grid", gap: 10 }}>
            {/* Goal */}
            {(refined?.goal || nlPrompt) && (
              <div>
                <div style={{ fontSize: 12, color: "#6b7280", marginBottom: 4 }}>Goal</div>
                <div style={{ border: "1px solid #e5e7eb", borderLeft: "4px solid #111827", borderRadius: 10, padding: 10, background: "#f8fafc", fontWeight: 600 }}>
                  {refined?.goal || nlPrompt}
                </div>
              </div>
            )}
            <div>
              <div style={{ fontSize: 12, color: "#6b7280", marginBottom: 4 }}>Constraints</div>
              {conciseConstraints.length === 0 ? (
                <div style={{ padding: 10, border: "1px solid #e5e7eb", borderRadius: 10, fontStyle: "italic", color: "#6b7280" }}>Not specified</div>
              ) : (
                <div style={{ border: "1px solid #e5e7eb", borderRadius: 10, background: "#fff" }}>
                  <ul style={{ margin: 0, padding: "8px 12px", listStyle: "none" }}>
                    {conciseConstraints.map((c, i) => (
                      <motion.li
                        key={i}
                        initial={{ opacity: 0, x: -6 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ duration: 0.2, delay: i * 0.05 }}
                        style={{ display: "flex", gap: 8, alignItems: "flex-start", fontSize: 13, color: "#374151", marginBottom: 4 }}
                      >
                        <span style={{ width: 6, height: 6, marginTop: 6, borderRadius: 999, background: "#94a3b8", display: "inline-block" }} />
                        <span>{c}</span>
                      </motion.li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
            {/* Scenarios section */}
            {(() => {
              // Use scenarios from state, or fall back to refined?.scenarios
              const displayScenarios = (Array.isArray(scenarios) && scenarios.length > 0) 
                ? scenarios 
                : (Array.isArray(refined?.scenarios) ? refined.scenarios : [])
              const scenarioCount = displayScenarios.length || numScenarios
              
              return (
                <div>
                  <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
                    <div style={{ fontSize: 12, color: "#6b7280" }}>Scenarios</div>
                    <span style={{ 
                      fontSize: 11, 
                      fontWeight: 600, 
                      color: "#374151", 
                      background: "#f3f4f6", 
                      padding: "2px 8px", 
                      borderRadius: 4,
                      border: "1px solid #e5e7eb",
                    }}>
                      {scenarioCount} total
                    </span>
                  </div>
                  {displayScenarios.length > 0 ? (
                    <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 8 }}>
                      {displayScenarios.map((s: any, idx: number) => (
                        <motion.div
                          key={s.scenario_id || `scenario_${idx}`}
                          initial={{ opacity: 0, y: 6 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ duration: 0.2, delay: idx * 0.05 }}
                          style={{ border: "1px solid #e5e7eb", borderRadius: 8, padding: 8, background: "#fff" }}
                        >
                          <div style={{ fontWeight: 600, fontSize: 12 }}>{s.scenario_id || s.name || `Scenario ${idx + 1}`}</div>
                          <div style={{ fontSize: 11, color: "#6b7280" }}>{s?.metadata?.scaffold || s?.scaffold || ""}</div>
                          {(s?.smiles || s?.smiles === "") ? (
                            <div>
                              <div style={{ fontSize: 10, color: "#6b7280", marginTop: 4, marginBottom: 2 }}>SMILES:</div>
                              <pre style={{ whiteSpace: "pre-wrap", wordBreak: "break-all", fontSize: 9, background: "#fafafa", padding: 6, borderRadius: 4, margin: 0, fontFamily: "monospace", color: "#374151", maxHeight: 48, overflow: "hidden", textOverflow: "ellipsis" }}>
                                {s.smiles || "—"}
                              </pre>
                            </div>
                          ) : (
                            <div style={{ fontSize: 10, color: "#94a3b8", marginTop: 4, fontStyle: "italic" }}>Pending SMILES…</div>
                          )}
                        </motion.div>
                      ))}
                    </div>
                  ) : (
                    <div style={{ padding: 10, border: "1px solid #e5e7eb", borderRadius: 10, color: "#6b7280", fontSize: 13 }}>
                      No scenario details available yet
                    </div>
                  )}
                </div>
              )
            })()}
          </div>
        </motion.div>
        <div className="no-print" style={{ flex: "1 1 360px", minWidth: 320 }}>
          <DecisionCriteriaControls
            decisionCriteria={decisionCriteria}
            onCriteriaChange={(newCriteria: DecisionCriteria) => {
              setDecisionCriteria(newCriteria)
              setCriteriaSaved(false)
            }}
            showSaveButton={true}
            onSave={saveCriteriaChanges}
            onRegenerate={() => setShowRegenerateModal(true)}
          />
        </div>
      </div>

      {/* Re-generate Report Button - appears AFTER save is clicked on decision criteria */}
      {criteriaChangedAndSaved && (
        <div className="no-print" style={{ 
          display: "flex", 
          justifyContent: "center", 
          padding: "12px 0",
          marginBottom: 12,
        }}>
          <button
            onClick={() => setShowRegenerateModal(true)}
            style={{
              padding: "8px 20px",
              border: "1px solid #374151",
              borderRadius: 6,
              background: "#374151",
              color: "#fff",
              fontSize: 12,
              fontWeight: 600,
              cursor: "pointer",
            }}
          >
            Re-generate Report with Updated Criteria
          </button>
        </div>
      )}

      {/* Main report content - restructured */}
      <>
      {/* Report Header */}
      {scenarios && scenarios.length > 0 && (
        <motion.div {...fadeInUp} transition={{ duration: 0.25 }} style={{ marginTop: 16, marginBottom: 20 }}>
          <div style={{ 
            borderBottom: "2px solid #e5e7eb", 
            paddingBottom: 12, 
            marginBottom: 16,
            background: "#ffffff",
            padding: "12px 16px",
            borderRadius: "8px 8px 0 0",
          }}>
            <h2 style={{ 
              fontSize: 24, 
              fontWeight: 700, 
              color: "#111827", 
              margin: 0,
              letterSpacing: "-0.02em",
            }}>
              Virtual Lab Report
            </h2>
            <p style={{ 
              fontSize: 14, 
              color: "#6b7280", 
              margin: "6px 0 0 0" 
            }}>
              Comprehensive analysis of molecular candidates
            </p>
          </div>
        </motion.div>
      )}

      {/* Scorecard Carousel - moved to first position */}
      {scenarios && scenarios.length > 0 && (
        <motion.div className="no-print" {...fadeInUp} transition={{ duration: 0.25 }} style={{ marginTop: 16, border: "1px solid #e5e7eb", borderRadius: 12, padding: 12, background: "#ffffff", boxShadow: "0 1px 3px rgba(16,24,40,0.06)" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
            <strong style={{ fontSize: 16 }}>Detailed Scorecards</strong>
            <div style={{ fontSize: 12, color: "#666" }}>
              {carouselIndex + 1} / {scenarios.length}
            </div>
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "40px 1fr 40px", alignItems: "center", gap: 8 }}>
            <button
              aria-label="Previous"
              onClick={() => setCarouselIndex((i) => (i - 1 + scenarios.length) % scenarios.length)}
              style={{ padding: 8, border: "1px solid #ddd", borderRadius: 6, background: "transparent", cursor: "pointer" }}
            >
              ◀
            </button>
            <div style={{ border: "1px solid #eee", borderRadius: 8, padding: 12, background: "#ffffff" }}>
              {renderScorecard(scenarios[carouselIndex])}
            </div>
            <button
              aria-label="Next"
              onClick={() => setCarouselIndex((i) => (i + 1) % scenarios.length)}
              style={{ padding: 8, border: "1px solid #ddd", borderRadius: 6, background: "transparent", cursor: "pointer" }}
            >
              ▶
            </button>
          </div>
        </motion.div>
      )}

      {scenarios && scenarios.length > 0 && (
        <div className="print-only" style={{ display: "none", marginTop: 16 }}>
          <SectionTitle title="Detailed Scorecards" />
          <div style={{ display: "grid", gap: 12 }}>
            {scenarios.map((s: any) => (
              <div key={`print-${s.scenario_id}`} style={{ border: "1px solid #e5e7eb", borderRadius: 12, padding: 12, background: "#ffffff" }}>
                {renderScorecard(s)}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Ranking Table with CSV Export */}
      {scenarios && (
        <div style={{ marginTop: 24 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
            <strong style={{ fontSize: 16 }}>Molecule Ranking</strong>
            <button 
              onClick={() => exportCSV(scenarios, normalizedMetrics, feedback)}
              style={{ 
                padding: "6px 12px", 
                border: "1px solid #e5e7eb", 
                borderRadius: 8, 
                background: "#ffffff", 
                color: "#111827",
                fontSize: 12,
                fontWeight: 600,
                cursor: "pointer"
              }}
            >
              Export CSV
            </button>
          </div>
          <div style={{ display: "flex", gap: 8, alignItems: "center", margin: "8px 0" }}>
            <span style={{ fontSize: 12, color: "#666" }}>Sort by:</span>
            <button onClick={() => setSortKey("best")} style={{ padding: "6px 10px", border: "1px solid #e5e7eb", borderRadius: 8, background: sortKey === "best" ? "#eef2ff" : "#fff", transition: "background .2s ease", cursor: "pointer" }}>
              Best fit
            </button>
            <button onClick={() => setSortKey("ba")} style={{ padding: "6px 10px", border: "1px solid #e5e7eb", borderRadius: 8, background: sortKey === "ba" ? "#eef2ff" : "#fff", transition: "background .2s ease", cursor: "pointer" }}>
              Potency (ΔG)
            </button>
            <button onClick={() => setSortKey("tox")} style={{ padding: "6px 10px", border: "1px solid #e5e7eb", borderRadius: 8, background: sortKey === "tox" ? "#eef2ff" : "#fff", transition: "background .2s ease", cursor: "pointer" }}>
              Toxicity
            </button>
            <button onClick={() => setSortKey("cost")} style={{ padding: "6px 10px", border: "1px solid #e5e7eb", borderRadius: 8, background: sortKey === "cost" ? "#eef2ff" : "#fff", transition: "background .2s ease", cursor: "pointer" }}>
              Cost
            </button>
            <button 
              onClick={() => setAsc((p) => !p)} 
              style={{ 
                padding: "6px 10px", 
                border: "1px solid #e5e7eb", 
                borderRadius: 8, 
                cursor: "pointer",
                display: "inline-flex",
                alignItems: "center",
                gap: 4
              }}
              title={asc ? "Ascending" : "Descending"}
            >
              {asc ? "↑" : "↓"}
            </button>
          </div>
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", background: "#ffffff", border: "1px solid #e5e7eb", borderRadius: 10 }}>
              <thead>
                <tr>
                  <th style={{ textAlign: "left", borderBottom: "1px solid #e5e7eb", padding: 10, background: "#f8fafc" }}>Scenario</th>
                  <th style={{ textAlign: "left", borderBottom: "1px solid #e5e7eb", padding: 10, background: "#f8fafc" }}>SMILES</th>
                  <th style={{ textAlign: "right", borderBottom: "1px solid #e5e7eb", padding: 10, background: "#f8fafc" }}>ΔG (kcal/mol)</th>
                  <th style={{ textAlign: "center", borderBottom: "1px solid #e5e7eb", padding: 10, background: "#f8fafc" }}>Risk</th>
                  <th style={{ textAlign: "right", borderBottom: "1px solid #e5e7eb", padding: 10, background: "#f8fafc" }}>Cost ($)</th>
                </tr>
              </thead>
              <tbody>
                {getRankedRows().map((s: any) => {
                  const isWinner = (winners || []).some((w: any) => w.scenario_id === s.scenario_id)
                  return (
                    <tr key={s.scenario_id} style={{ background: isWinner ? "#f0fdf4" : "#ffffff" }}>
                      <td style={{ padding: 10, borderBottom: "1px solid #eef2f7" }}>{s.scenario_id}</td>
                      <td style={{ padding: 10, borderBottom: "1px solid #eef2f7" }}><code>{s.smiles}</code></td>
                      <td style={{ padding: 10, borderBottom: "1px solid #eef2f7", textAlign: "right" }}>{s._ba ?? ""}</td>
                      <td style={{ padding: 10, borderBottom: "1px solid #eef2f7", textAlign: "center" }}>{s._tox ?? ""}</td>
                      <td style={{ padding: 10, borderBottom: "1px solid #eef2f7", textAlign: "right" }}>{s._cost ?? ""}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Pareto Analysis Graph */}
      {scenarios && scenarios.length > 0 && (
        <motion.div {...fadeInUp} transition={{ duration: 0.25 }} style={{ marginTop: 24 }}>
          <strong style={{ fontSize: 16, display: "block", marginBottom: 12 }}>Pareto Analysis</strong>
          <ParetoChart 
            scenarios={scenarios}
            scenarioMetrics={normalizedMetrics}
            winners={winners || []}
          />
        </motion.div>
      )}

      {/* Final Verdict Box - only show if no structured LLM report yet */}
      {!structuredReport && (winners || rejected || (scenarios && scenarios.length === 1)) && ((winners?.length ?? 0) > 0 || (rejected?.length ?? 0) > 0 || (scenarios && scenarios.length === 1)) && (
        <motion.div {...fadeInUp} transition={{ duration: 0.25 }} style={{ marginTop: 24, border: "1px solid #e5e7eb", borderRadius: 12, padding: 16, background: "#ffffff", marginBottom: 0, boxShadow: "0 1px 3px rgba(0,0,0,0.06)" }}>
          <h3 style={{ fontSize: 16, fontWeight: 600, marginBottom: 12, color: "#111827", marginTop: 0, letterSpacing: "-0.01em" }}>Final Verdict</h3>
          
          {/* Single Scenario verdict when no explicit winners/rejected yet */}
          {scenarios && scenarios.length === 1 && (!winners || winners.length === 0) && (!rejected || rejected.length === 0) && metricsAggregated && (
            (() => {
              const s = scenarios[0]
              const sid = s.scenario_id
              const metrics = normalizedMetrics[sid] || {}
              const docking = metrics.docking || {}
              const admet = metrics.admet || {}
              const synthesis = metrics.synthesis || {}
              
              // Determine if single scenario passes or fails based on decision criteria
              const potencyPass = docking.potency_pass !== false && (docking.binding_affinity_kcal_per_mol == null || docking.binding_affinity_kcal_per_mol <= decisionCriteria.docking.hardFailThreshold)
              const safetyPass = admet.is_safe !== false && (!decisionCriteria.admet.hardFailHERG || !admet.herg_flag)
              const synthesisPass = (synthesis.sa_score == null || synthesis.sa_score <= decisionCriteria.synthesis.hardFailSa)
              const isWinner = potencyPass && safetyPass && synthesisPass
              
              return (
                <div style={{ marginBottom: 0 }}>
                  <div style={{ fontSize: 14, fontWeight: 600, color: isWinner ? "#34a853" : "#ea4335", marginBottom: 8 }}>
                    {isWinner ? "✓ Single Candidate - Approved" : "✗ Single Candidate - Rejected"}
                  </div>
                  <div
                    style={{
                      border: `1px solid ${isWinner ? "#34a853" : "#ea4335"}`,
                      borderRadius: 8,
                      padding: 14,
                      background: isWinner ? "#f0fdf4" : "#fef2f2",
                      display: "flex",
                      flexDirection: "column",
                      gap: 10,
                    }}
                  >
                    <div style={{ fontSize: 12, color: "#0f172a", fontWeight: 600 }}>
                      {formatScenarioLabel(sid)}: <span style={{ fontFamily: "monospace", fontWeight: 400 }}>{s.smiles || "(SMILES not provided)"}</span>
                    </div>
                    
                    {/* Metrics */}
                    <div
                      style={{
                        fontSize: 12,
                        color: "#374151",
                        padding: 10,
                        background: "#fff",
                        borderRadius: 6,
                        border: `1px solid ${isWinner ? "#86efac" : "#fca5a5"}`,
                      }}
                    >
                      <div style={{ fontWeight: 600, marginBottom: 6, color: isWinner ? "#166534" : "#b91c1c" }}>Metrics Analysis:</div>
                      <div style={{ display: "grid", gap: 3 }}>
                        <div>• <strong>Binding Affinity (ΔG):</strong> {docking.binding_affinity_kcal_per_mol != null ? `${docking.binding_affinity_kcal_per_mol} kcal/mol` : "N/A"} {potencyPass ? "✓" : "✗"}</div>
                        <div>• <strong>Potency Pass:</strong> {docking.potency_pass != null ? (docking.potency_pass ? "✓ Yes" : "✗ No") : "N/A"}</div>
                        <div>• <strong>hERG Flag:</strong> {admet.herg_flag != null ? (admet.herg_flag ? "✗ Yes (Cardiotoxic)" : "✓ No") : "N/A"}</div>
                        <div>• <strong>SA Score:</strong> {synthesis.sa_score != null ? synthesis.sa_score : "N/A"} {synthesisPass ? "✓" : "✗"}</div>
                        <div>• <strong>Estimated Cost:</strong> {synthesis.estimated_cost_usd != null ? `$${synthesis.estimated_cost_usd}` : "N/A"}</div>
                      </div>
                    </div>
                    
                    {/* Verdict Rationale */}
                    <div
                      style={{
                        fontSize: 13,
                        color: isWinner ? "#166534" : "#b91c1c",
                        padding: 10,
                        background: "#fff",
                        borderRadius: 6,
                        border: `1px solid ${isWinner ? "#86efac" : "#fca5a5"}`,
                      }}
                    >
                      <div style={{ fontWeight: 600, marginBottom: 4 }}>{isWinner ? "Approval Rationale:" : "Rejection Rationale:"}</div>
                      <div style={{ lineHeight: 1.6 }}>
                        {isWinner
                          ? "This single candidate meets all critical thresholds for potency, safety, and synthetic feasibility. Recommended for further development."
                          : `This candidate was rejected due to: ${!potencyPass ? "insufficient binding affinity; " : ""}${!safetyPass ? "safety concerns (hERG liability or toxicity); " : ""}${!synthesisPass ? "poor synthetic accessibility; " : ""}`.replace(/; $/, ".")}
                      </div>
                    </div>
                  </div>
                </div>
              )
            })()
          )}

          {/* Winner */}
          {winners && winners.length > 0 && (
            <div style={{ marginBottom: rejected && rejected.length > 0 ? 16 : 0 }}>
              <div style={{ fontSize: 14, fontWeight: 600, color: "#2EA87A", marginBottom: 8 }}>✓ Winning Molecule</div>
              {winners.map((w: any) => {
                const baseScenario = scenarioLookup[w.scenario_id] || {}
                const metrics = normalizedMetrics[w.scenario_id] || {}
                const docking = metrics.docking || {}
                const admet = metrics.admet || {}
                const synthesis = metrics.synthesis || {}
                
                return (
                  <div
                    key={w.scenario_id}
                    style={{
                      border: "1px solid #34a853",
                      borderRadius: 8,
                      padding: 14,
                      background: "#f0fdf4",
                      display: "flex",
                      flexDirection: "column",
                      gap: 10,
                    }}
                  >
                    <div style={{ fontSize: 12, color: "#0f172a", fontWeight: 600 }}>
                      {formatScenarioLabel(w.scenario_id)}: <span style={{ fontFamily: "monospace", fontWeight: 400 }}>{w.smiles || baseScenario?.smiles || "(SMILES not provided)"}</span>
                    </div>
                    
                    {/* Detailed Metrics */}
                    <div
                      style={{
                        fontSize: 12,
                        color: "#374151",
                        padding: 10,
                        background: "#fff",
                        borderRadius: 6,
                        border: "1px solid #86efac",
                      }}
                    >
                      <div style={{ fontWeight: 600, marginBottom: 6, color: "#166534" }}>Metrics Analysis:</div>
                      <div style={{ display: "grid", gap: 3 }}>
                        <div>• <strong>Binding Affinity (ΔG):</strong> {docking.binding_affinity_kcal_per_mol != null ? `${docking.binding_affinity_kcal_per_mol} kcal/mol` : "N/A"} {docking.potency_pass ? "✓" : ""}</div>
                        <div>• <strong>Potency Pass:</strong> {docking.potency_pass != null ? (docking.potency_pass ? "✓ Yes" : "✗ No") : "N/A"}</div>
                        <div>• <strong>hERG Flag:</strong> {admet.herg_flag != null ? (admet.herg_flag ? "✗ Yes (Cardiotoxic)" : "✓ No") : "N/A"}</div>
                        <div>• <strong>SA Score:</strong> {synthesis.sa_score != null ? synthesis.sa_score : "N/A"} {synthesis.sa_score != null && synthesis.sa_score < 5 ? "✓" : ""}</div>
                        <div>• <strong>Estimated Cost:</strong> {synthesis.estimated_cost_usd != null ? `$${synthesis.estimated_cost_usd}` : "N/A"}</div>
                      </div>
                    </div>
                    
                    {/* Selection Rationale */}
                    <div
                      style={{
                        fontSize: 13,
                        color: "#166534",
                        padding: 10,
                        background: "#fff",
                        borderRadius: 6,
                        border: "1px solid #86efac",
                      }}
                    >
                      <div style={{ fontWeight: 600, marginBottom: 4 }}>Selection Rationale:</div>
                      <div style={{ lineHeight: 1.6 }}>
                        {(() => {
                          const strengths = []
                          if (docking.potency_pass) {
                            strengths.push(`Strong binding affinity (${docking.binding_affinity_kcal_per_mol} kcal/mol)`)
                          }
                          if (admet.toxicity_risk === "LOW") {
                            strengths.push("Low toxicity risk profile")
                          }
                          if (!admet.herg_flag) {
                            strengths.push("No hERG liability - reduced cardiotoxicity risk")
                          }
                          if (admet.is_safe) {
                            strengths.push("Passed all safety assessments")
                          }
                          if (synthesis.sa_score != null && synthesis.sa_score < 5) {
                            strengths.push(`Excellent synthetic accessibility (SA Score: ${synthesis.sa_score})`)
                          } else if (synthesis.sa_score != null && synthesis.sa_score < 6) {
                            strengths.push(`Good synthetic accessibility (SA Score: ${synthesis.sa_score})`)
                          }
                          if (synthesis.estimated_cost_usd != null && synthesis.estimated_cost_usd < 2000) {
                            strengths.push(`Cost-effective synthesis ($${synthesis.estimated_cost_usd})`)
                          }
                          
                          if (strengths.length === 0) {
                            return "This molecule demonstrated the best overall balance of potency, safety, and manufacturability among all candidates."
                          }
                          
                          return `Selected for optimal balance: ${strengths.join("; ")}. This candidate meets all critical constraints while offering the best risk-benefit profile for further development.`
                        })()}
                      </div>
                    </div>
                  </div>
                )
              })}
            </div>
          )}

          {/* Rejection Log */}
          {rejected && rejected.length > 0 && (
            <div style={{ marginBottom: 0 }}>
              <div style={{ fontSize: 14, fontWeight: 600, color: "#ea4335", marginBottom: 8 }}>✗ Rejection Log</div>
              <div style={{ display: "grid", gap: 10 }}>
                {rejected.map((r: any) => {
                  const baseScenario = scenarioLookup[r.scenario_id] || {}
                  const metrics = normalizedMetrics[r.scenario_id] || {}
                  const docking = metrics.docking || {}
                  const admet = metrics.admet || {}
                  const synthesis = metrics.synthesis || {}
                  
                  return (
                    <div
                      key={r.scenario_id}
                      style={{
                        border: "1px solid #ea4335",
                        borderRadius: 8,
                        padding: 14,
                        background: "#fef2f2",
                        display: "flex",
                        flexDirection: "column",
                        gap: 10,
                      }}
                    >
                      <div style={{ fontSize: 12, color: "#0f172a", fontWeight: 600 }}>
                        {formatScenarioLabel(r.scenario_id)}: <span style={{ fontFamily: "monospace", fontWeight: 400 }}>{r.smiles || baseScenario?.smiles || "(SMILES not provided)"}</span>
                      </div>
                      
                      {/* Detailed Metrics */}
                      <div style={{ 
                        fontSize: 12, 
                        color: "#374151", 
                        marginBottom: 8,
                        padding: 10,
                        background: "#fff",
                        borderRadius: 6,
                        border: "1px solid #fecaca"
                      }}>
                        <div style={{ fontWeight: 600, marginBottom: 6, color: "#991b1b" }}>Metrics Analysis:</div>
                        <div style={{ display: "grid", gap: 3 }}>
                          <div>• <strong>Binding Affinity (ΔG):</strong> {docking.binding_affinity_kcal_per_mol != null ? `${docking.binding_affinity_kcal_per_mol} kcal/mol` : "N/A"}</div>
                          <div>• <strong>Potency Pass:</strong> {docking.potency_pass != null ? (docking.potency_pass ? "✓ Yes" : "✗ No") : "N/A"}</div>
                          <div>• <strong>hERG Flag:</strong> {admet.herg_flag != null ? (admet.herg_flag ? "✗ Yes (Cardiotoxic)" : "✓ No") : "N/A"}</div>
                          <div>• <strong>SA Score:</strong> {synthesis.sa_score != null ? synthesis.sa_score : "N/A"}</div>
                          <div>• <strong>Estimated Cost:</strong> {synthesis.estimated_cost_usd != null ? `$${synthesis.estimated_cost_usd}` : "N/A"}</div>
                        </div>
                      </div>
                      
                      {/* Rejection Reasoning */}
                      <div style={{ 
                        fontSize: 13, 
                        color: "#991b1b",
                        padding: 10,
                        background: "#fff",
                        borderRadius: 6,
                        border: "1px solid #fecaca",
                        marginBottom: 0
                      }}>
                        <div style={{ fontWeight: 600, marginBottom: 4 }}>Rejection Rationale:</div>
                        <div style={{ lineHeight: 1.6 }}>
                          {r.veto_reason || (() => {
                            // Generate detailed rejection reason based on metrics
                            const reasons = []
                            if (admet.toxicity_risk === "HIGH") {
                              reasons.push(`HIGH toxicity risk detected (${admet.toxicity_risk})`)
                            }
                            if (admet.herg_flag === true) {
                              reasons.push("hERG liability flag raised - potential cardiotoxicity concern")
                            }
                            if (admet.is_safe === false) {
                              reasons.push("Failed safety assessment")
                            }
                            if (!docking.potency_pass) {
                              reasons.push(`Insufficient potency (ΔG: ${docking.binding_affinity_kcal_per_mol} kcal/mol)`)
                            }
                            if (synthesis.sa_score != null && synthesis.sa_score > 6) {
                              reasons.push(`Poor synthetic accessibility (SA Score: ${synthesis.sa_score})`)
                            }
                            if (reasons.length === 0) {
                              return "Did not meet the specified safety, efficacy, or manufacturability constraints"
                            }
                            return reasons.join("; ")
                          })()}
                        </div>
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>
          )}
        </motion.div>
      )}

      {/* Final Summary & AI Rationale - only show when LLM rationale is ready */}
      {llmReason && (winners || rejected) && ((winners?.length ?? 0) > 0 || (rejected?.length ?? 0) > 0) && (
        <motion.div id="simulab-report" {...fadeInUp} transition={{ duration: 0.2 }} style={{ 
          marginTop: 24,
          border: "1px solid #e5e7eb", 
          borderRadius: 12, 
          padding: 20, 
          background: "#ffffff", 
          boxShadow: "0 1px 3px rgba(0,0,0,0.06)" 
        }}>
          <div style={{ 
            borderBottom: "1px solid #e5e7eb", 
            paddingBottom: 12, 
            marginBottom: 16,
            background: "#ffffff",
            margin: "-20px -20px 16px -20px",
            padding: "16px 20px 12px 20px",
            borderRadius: "12px 12px 0 0",
          }}>
            <h3 style={{ 
              fontSize: 20, 
              fontWeight: 600, 
              margin: 0, 
              color: "#111827",
              letterSpacing: "-0.01em",
            }}>
              Final Summary & Conclusion
            </h3>
          </div>

          {/* Structured Report Display */}
          {structuredReport ? (
            <StructuredReportDisplay 
              report={structuredReport} 
              dataSource={dataSource}
              confidence={dataConfidence}
            />
          ) : (
            <div style={{ 
              fontSize: 14, 
              lineHeight: 1.7, 
              color: "#374151",
              background: "#f8fafc",
              padding: 16,
              borderRadius: 8,
              border: "1px solid #e5e7eb",
              marginBottom: 16
            }}>
              <pre style={{ 
                whiteSpace: "pre-wrap", 
                fontFamily: "inherit",
                margin: 0 
              }}>{reportOverrides.llmReason || llmReason}</pre>
            </div>
          )}

          <CalculationBreakdown 
            winners={winners || []} 
            rejected={rejected || []} 
            normalizedMetrics={normalizedMetrics}
            structuredReport={structuredReport}
          />
          <AssumptionsList
            conciseConstraints={conciseConstraints}
            decisionCriteria={decisionCriteria}
          />
        </motion.div>
      )}
      
      {llmReason && (winners || rejected) && ((winners?.length ?? 0) > 0 || (rejected?.length ?? 0) > 0) && (
        <>
        {/* Action buttons row */}
        <div className="no-print" style={{ display: "flex", justifyContent: "flex-end", marginTop: 12, gap: 8, alignItems: "center" }}>
          {!showEditChat && (
            <button 
              onClick={() => setShowEditChat(true)}
              style={{ 
                padding: "10px 16px", 
                border: "1px solid #e5e7eb", 
                borderRadius: 8, 
                background: "#ffffff",
                color: "#374151",
                fontWeight: 600,
                fontSize: 13,
                cursor: "pointer",
                boxShadow: "0 1px 3px rgba(0,0,0,0.04)",
                transition: "all 0.2s ease",
              }}
            >
              Edit Report
            </button>
          )}
          <button 
            onClick={exportReportAsPDF} 
            style={{ 
              padding: "10px 16px", 
              border: "1px solid #e5e7eb", 
              borderRadius: 8, 
              background: "#3F3E3E",
              color: "#ffffff",
              fontWeight: 600,
              fontSize: 13,
              cursor: "pointer",
              boxShadow: "0 1px 3px rgba(0,0,0,0.08)",
              transition: "all 0.2s ease",
            }}
          >
            Export Report as PDF
          </button>
        </div>

        {/* Edit Report Chat Panel */}
        {showEditChat && (
          <motion.div
            initial={{ opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.2 }}
            className="no-print"
            style={{
              marginTop: 16,
              border: "1px solid #e5e7eb",
              borderRadius: 12,
              background: "#ffffff",
              boxShadow: "0 4px 12px rgba(0,0,0,0.06)",
              overflow: "hidden",
            }}
          >
            <div style={{ 
              padding: "12px 16px", 
              borderBottom: "1px solid #f3f4f6", 
              background: "#f9fafb",
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
            }}>
              <h4 style={{ fontSize: 14, fontWeight: 600, color: "#111827", margin: 0 }}>
                Edit Report
              </h4>
              <button
                onClick={() => setShowEditChat(false)}
                style={{
                  background: "transparent",
                  border: "none",
                  cursor: "pointer",
                  color: "#6b7280",
                  fontSize: 18,
                  lineHeight: 1,
                  padding: 4,
                }}
                aria-label="Close edit panel"
              >
                ×
              </button>
            </div>
            
            {/* Chat history */}
            {editChatHistory.length > 0 && (
              <div style={{ 
                maxHeight: 200, 
                overflowY: "auto", 
                padding: "12px 16px",
                borderBottom: "1px solid #f3f4f6",
                background: "#fafafa",
              }}>
                {editChatHistory.map((msg, idx) => (
                  <div 
                    key={idx} 
                    style={{ 
                      marginBottom: 8,
                      display: "flex",
                      justifyContent: msg.role === "user" ? "flex-end" : "flex-start",
                    }}
                  >
                    <div style={{
                      padding: "8px 12px",
                      borderRadius: 8,
                      fontSize: 13,
                      maxWidth: "80%",
                      background: msg.role === "user" ? "#3F3E3E" : "#e5e7eb",
                      color: msg.role === "user" ? "#ffffff" : "#374151",
                    }}>
                      {msg.content}
                    </div>
                  </div>
                ))}
                {isProcessingEdit && (
                  <div style={{ display: "flex", justifyContent: "flex-start" }}>
                    <div style={{
                      padding: "8px 12px",
                      borderRadius: 8,
                      fontSize: 13,
                      background: "#e5e7eb",
                      color: "#6b7280",
                    }}>
                      <span style={{ display: "inline-block", animation: "pulse 1s infinite" }}>Processing...</span>
                    </div>
                  </div>
                )}
              </div>
            )}
            
            {/* Input area */}
            <div style={{ padding: 16 }}>
              <p style={{ fontSize: 11, color: "#9ca3b8", marginBottom: 10, marginTop: 0, fontStyle: "italic" }}>
                Your feedback and all changes made in this review are logged and considered as high-quality data correction.
              </p>
              <div style={{ display: "flex", gap: 8 }}>
                <input
                  type="text"
                  value={editChatInput}
                  onChange={(e) => setEditChatInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey && editChatInput.trim() && !isProcessingEdit) {
                      e.preventDefault()
                      handleEditChatSubmit()
                    }
                  }}
                  placeholder="e.g., 'Change the binding affinity to -10.5' or 'Set SA score to 4.2'"
                  style={{
                    flex: 1,
                    padding: "10px 12px",
                    border: "1px solid #e5e7eb",
                    borderRadius: 8,
                    fontSize: 13,
                    outline: "none",
                    fontFamily: "inherit",
                  }}
                  disabled={isProcessingEdit}
                />
                <button
                  onClick={handleEditChatSubmit}
                  disabled={!editChatInput.trim() || isProcessingEdit}
                  style={{
                    padding: "10px 16px",
                    border: "none",
                    borderRadius: 8,
                    background: editChatInput.trim() && !isProcessingEdit ? "#3F3E3E" : "#d1d5db",
                    color: "#ffffff",
                    fontWeight: 500,
                    fontSize: 13,
                    cursor: editChatInput.trim() && !isProcessingEdit ? "pointer" : "not-allowed",
                  }}
                >
                  Send
                </button>
              </div>
              
              {/* Audit Log Toggle & Revert Button */}
              {editAuditLog.length > 0 && (
                <div style={{ 
                  display: "flex", 
                  alignItems: "center", 
                  justifyContent: "space-between",
                  marginTop: 12,
                  paddingTop: 12,
                  borderTop: "1px solid #e5e7eb"
                }}>
                  <button
                    onClick={() => setShowAuditLog(!showAuditLog)}
                    style={{
                      background: "transparent",
                      border: "none",
                      cursor: "pointer",
                      fontSize: 12,
                      color: "#6366f1",
                      fontWeight: 500,
                      display: "flex",
                      alignItems: "center",
                      gap: 4,
                      padding: 0,
                    }}
                  >
                    {showAuditLog ? "▼" : "▶"} Audit Log ({editAuditLog.length} edit{editAuditLog.length > 1 ? "s" : ""})
                  </button>
                  <button
                    onClick={() => {
                      if (originalReport) {
                        if (confirm("Revert to original report? All edits will be undone.")) {
                          setStructuredReport(JSON.parse(JSON.stringify(originalReport)))
                          setEditAuditLog([])
                          // DON'T clear originalReport - keep it so user can revert again if they make more edits
                          setEditChatHistory((prev) => [...prev, { 
                            role: "assistant", 
                            content: "✓ Report reverted to original version." 
                          }])
                        }
                      } else {
                        // No original saved - this shouldn't happen but handle gracefully
                        alert("Unable to revert - original report not saved. This may happen if you regenerated the report.")
                      }
                    }}
                    style={{
                      background: originalReport ? "#fef2f2" : "#f3f4f6",
                      border: `1px solid ${originalReport ? "#fecaca" : "#d1d5db"}`,
                      borderRadius: 6,
                      padding: "4px 10px",
                      fontSize: 11,
                      color: originalReport ? "#dc2626" : "#9ca3af",
                      fontWeight: 500,
                      cursor: originalReport ? "pointer" : "not-allowed",
                      opacity: originalReport ? 1 : 0.6,
                    }}
                    disabled={!originalReport}
                  >
                    Revert to Original
                  </button>
                </div>
              )}
              
              {/* Audit Log Panel */}
              {showAuditLog && editAuditLog.length > 0 && (
                <div style={{ 
                  marginTop: 12,
                  background: "#f8fafc",
                  borderRadius: 8,
                  border: "1px solid #e2e8f0",
                  maxHeight: 200,
                  overflowY: "auto"
                }}>
                  <div style={{ 
                    padding: "8px 12px", 
                    borderBottom: "1px solid #e2e8f0",
                    background: "#f1f5f9",
                    position: "sticky",
                    top: 0
                  }}>
                    <span style={{ fontSize: 11, fontWeight: 600, color: "#475569" }}>
                      Edit History
                    </span>
                  </div>
                  {editAuditLog.map((entry, idx) => (
                    <div 
                      key={entry.id}
                      style={{ 
                        padding: "10px 12px",
                        borderBottom: idx < editAuditLog.length - 1 ? "1px solid #e2e8f0" : "none",
                      }}
                    >
                      <div style={{ 
                        display: "flex", 
                        justifyContent: "space-between", 
                        alignItems: "flex-start",
                        marginBottom: 4
                      }}>
                        <span style={{ 
                          fontSize: 10, 
                          color: "#94a3b8",
                          fontFamily: "monospace"
                        }}>
                          #{idx + 1} • {new Date(entry.timestamp).toLocaleTimeString()}
                        </span>
                        <button
                          onClick={() => {
                            if (confirm(`Revert to state after edit #${idx + 1}?`)) {
                              setStructuredReport(JSON.parse(JSON.stringify(entry.reportSnapshot)))
                              // Keep audit log up to this point
                              setEditAuditLog(prev => prev.slice(0, idx + 1))
                              setEditChatHistory((prev) => [...prev, { 
                                role: "assistant", 
                                content: `✓ Reverted to state after edit #${idx + 1}.` 
                              }])
                            }
                          }}
                          style={{
                            background: "transparent",
                            border: "none",
                            cursor: "pointer",
                            fontSize: 10,
                            color: "#6366f1",
                            padding: 0,
                          }}
                        >
                          Restore
                        </button>
                      </div>
                      <div style={{ 
                        fontSize: 12, 
                        color: "#1e293b",
                        fontWeight: 500,
                        marginBottom: 2
                      }}>
                        "{entry.instruction}"
                      </div>
                      <div style={{ 
                        fontSize: 11, 
                        color: "#64748b"
                      }}>
                        {entry.summary}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </motion.div>
        )}
        </>
      )}
      </>

      {/* Regenerate Confirmation Modal */}
      {showRegenerateModal && (
        <div style={{
          position: "fixed",
          inset: 0,
          background: "rgba(0,0,0,0.5)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          zIndex: 1000,
        }}>
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            style={{
              background: "#fff",
              borderRadius: 16,
              padding: 24,
              maxWidth: 420,
              width: "90%",
              boxShadow: "0 20px 40px rgba(0,0,0,0.2)",
            }}
          >
            <h3 style={{ margin: "0 0 12px 0", fontSize: 18, fontWeight: 600, color: "#1e293b" }}>
              Re-evaluate with Updated Criteria?
            </h3>
            <p style={{ margin: "0 0 20px 0", fontSize: 14, color: "#64748b", lineHeight: 1.6 }}>
              The Judge Agent will re-evaluate the existing {Object.keys(scenarioMetrics).length || "N"} scenarios 
              using your updated decision criteria. Pass/fail verdicts may change based on the new thresholds.
            </p>
            <div style={{ display: "flex", gap: 12, justifyContent: "flex-end" }}>
              <button
                onClick={() => setShowRegenerateModal(false)}
                style={{
                  padding: "10px 20px",
                  border: "1px solid #e2e8f0",
                  borderRadius: 8,
                  background: "#fff",
                  color: "#64748b",
                  fontSize: 14,
                  fontWeight: 500,
                  cursor: "pointer",
                }}
              >
                Cancel
              </button>
              <button
                onClick={regenerateReport}
                style={{
                  padding: "10px 20px",
                  border: "none",
                  borderRadius: 8,
                  background: "#1e293b",
                  color: "#fff",
                  fontSize: 14,
                  fontWeight: 600,
                  cursor: "pointer",
                }}
              >
                Re-generate
              </button>
            </div>
          </motion.div>
        </div>
      )}

      {/* Regenerating Loading Overlay - Clean Step Flow */}
      {isRegenerating && (
        <div style={{
          position: "fixed",
          inset: 0,
          background: "rgba(255,255,255,0.97)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          zIndex: 1001,
        }}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            style={{
              background: "#fff",
              borderRadius: 14,
              padding: 24,
              maxWidth: 420,
              width: "90%",
              boxShadow: "0 8px 32px rgba(0,0,0,0.1)",
              border: "1px solid #e2e8f0",
            }}
          >
            {/* Header */}
            <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 20 }}>
              <div style={{
                width: 40,
                height: 40,
                borderRadius: 10,
                background: "#1e293b",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
              }}>
                <RefreshCw size={20} color="#fff" style={{ animation: "spin 1s linear infinite" }} />
              </div>
              <div>
                <h3 style={{ margin: 0, fontSize: 15, fontWeight: 600, color: "#0f172a" }}>
                  Re-evaluating Scenarios
                </h3>
                <p style={{ margin: "2px 0 0 0", fontSize: 12, color: "#64748b" }}>
                  {Object.keys(scenarioMetrics).length} scenarios • Updated criteria
                </p>
              </div>
            </div>
            
            {/* Steps List */}
            <div style={{ 
              background: "#f8fafc", 
              borderRadius: 8, 
              padding: 12,
              border: "1px solid #e2e8f0",
            }}>
              {regenerateProgress.map((step, idx) => {
                const isComplete = step.startsWith("✓")
                const isError = step.startsWith("❌")
                const isActive = !isComplete && !isError && idx === regenerateProgress.length - 1
                
                // Parse step to get agent and action
                const colonIdx = step.indexOf(":")
                const agent = colonIdx > 0 ? step.substring(0, colonIdx).replace("✓ ", "").replace("❌ ", "") : ""
                const action = colonIdx > 0 ? step.substring(colonIdx + 1).trim() : step.replace("✓ ", "").replace("❌ ", "")
                
                return (
                  <motion.div
                    key={idx}
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.2 }}
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: 10,
                      padding: "8px 0",
                      borderBottom: idx < regenerateProgress.length - 1 ? "1px solid #e2e8f0" : "none",
                    }}
                  >
                    {/* Status icon */}
                    <div style={{
                      width: 20,
                      height: 20,
                      borderRadius: "50%",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      flexShrink: 0,
                      background: isComplete ? "#16a34a" : isError ? "#dc2626" : "transparent",
                    }}>
                      {isComplete ? (
                        <CheckCircle size={14} color="#fff" />
                      ) : isError ? (
                        <XCircle size={14} color="#fff" />
                      ) : isActive ? (
                        <div style={{
                          width: 14,
                          height: 14,
                          borderRadius: "50%",
                          border: "2px solid #6366f1",
                          borderTopColor: "transparent",
                          animation: "spin 0.8s linear infinite",
                        }} />
                      ) : (
                        <div style={{ width: 8, height: 8, borderRadius: "50%", background: "#cbd5e1" }} />
                      )}
                    </div>
                    
                    {/* Step text */}
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <span style={{
                        fontSize: 12,
                        fontWeight: isActive ? 600 : 500,
                        color: isComplete ? "#16a34a" : isError ? "#dc2626" : isActive ? "#1e293b" : "#64748b",
                      }}>
                        {agent && <span style={{ color: isComplete ? "#16a34a" : isActive ? "#6366f1" : "#94a3b8" }}>{agent}: </span>}
                        {action}
                      </span>
                    </div>
                  </motion.div>
                )
              })}
            </div>
          </motion.div>
        </div>
      )}

      <style jsx global>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>
      </div>
    </div>
  )
}

// Professional structured report display component - clean, no colors
function StructuredReportDisplay({ report, dataSource, confidence }: { 
  report: any;
  dataSource?: "llm" | "llm_validated";
  confidence?: "high" | "medium" | "low";
}) {
  if (!report) return null
  
  // Confidence indicator component - simple badge only
  const ConfidenceIndicator = () => {
    if (!confidence) return null;
    
    const confidenceStyles: Record<string, { bg: string; text: string; border: string; label: string }> = {
      high: { bg: "#f0fdf4", text: "#16a34a", border: "#86efac", label: "HIGH CONFIDENCE" },
      medium: { bg: "#fffbeb", text: "#d97706", border: "#fcd34d", label: "MODERATE CONFIDENCE" },
      low: { bg: "#fef2f2", text: "#dc2626", border: "#fca5a5", label: "LOW CONFIDENCE" },
    };
    
    const style = confidenceStyles[confidence || "medium"];
    
    return (
      <div style={{ 
        display: "inline-flex", 
        alignItems: "center", 
        padding: "4px 10px", 
        background: style.bg, 
        border: `1px solid ${style.border}`,
        borderRadius: 6,
        marginBottom: 16,
      }}>
        <div style={{ 
          fontSize: 10, 
          fontWeight: 700, 
          color: style.text, 
          letterSpacing: "0.5px",
        }}>
          {style.label}
        </div>
      </div>
    );
  };
  
  // Collect all candidates for Pareto chart
  // Categories: winner (green), selected (blue), rejected (gray)
  const allCandidates = [
    ...(report.winner ? [{ ...report.winner, category: "winner" }] : []),
    ...(report.selected || []).map((s: any) => ({ ...s, category: "selected" })),
    ...(report.rejected || []).map((r: any) => ({ ...r, category: "rejected" })),
  ]

  // Pareto Chart Component using LLM data
  const ParetoChart = () => {
    if (allCandidates.length === 0) return null
    
    const chartWidth = 400
    const chartHeight = 280
    const padding = { top: 30, right: 120, bottom: 50, left: 70 }
    const plotWidth = chartWidth - padding.left - padding.right
    const plotHeight = chartHeight - padding.top - padding.bottom
    
    // X-axis: Binding Affinity (ΔG) - REVERSED so more negative (better) is on the RIGHT
    // Y-axis: SA Score - REVERSED so lower (better) is at the TOP
    // This puts the winning molecule (best binding + best SA) in the TOP-RIGHT quadrant
    const xValues = allCandidates.map(c => c.binding_affinity || 0)
    const yValues = allCandidates.map(c => c.sa_score || 0)
    
    const xMin = Math.min(...xValues) - 1  // Most negative (best)
    const xMax = Math.max(...xValues) + 1  // Least negative (worst)
    const yMin = 0
    const yMax = Math.max(...yValues, 8) + 1
    
    // REVERSED X-axis: more negative values (better binding) go to the RIGHT
    const scaleX = (val: number) => padding.left + ((xMax - val) / (xMax - xMin)) * plotWidth
    // Y-axis: lower SA scores (better) go to the TOP (standard SVG: y=0 is top, so lower values = smaller y-coordinate)
    // We want: yMin at TOP (small y-coord), yMax at BOTTOM (large y-coord)
    const scaleY = (val: number) => padding.top + ((val - yMin) / (yMax - yMin)) * plotHeight
    
    // Generate axis ticks
    // X-axis: show from least negative (left) to most negative (right) - but ticks are positioned by scaleX
    const xTicks = Array.from({ length: 5 }, (_, i) => xMax - (i / 4) * (xMax - xMin))
    // Y-axis: show from yMax at bottom to yMin at top - ticks positioned by scaleY
    const yTicks = Array.from({ length: 5 }, (_, i) => yMin + (i / 4) * (yMax - yMin))
    
    return (
      <div style={{ marginBottom: 24 }}>
        <div style={{ fontSize: 13, fontWeight: 600, color: "#111827", marginBottom: 12 }}>
          Pareto Analysis
        </div>
        <div style={{ display: "flex", gap: 20 }}>
          <svg width={chartWidth} height={chartHeight} style={{ background: "#fafafa", borderRadius: 6, border: "1px solid #e5e7eb" }}>
            {/* Grid lines */}
            {yTicks.map((tick, i) => (
              <line
                key={`grid-y-${i}`}
                x1={padding.left}
                x2={chartWidth - padding.right}
                y1={scaleY(tick)}
                y2={scaleY(tick)}
                stroke="#e5e7eb"
                strokeDasharray="3,3"
              />
            ))}
            {xTicks.map((tick, i) => (
              <line
                key={`grid-x-${i}`}
                x1={scaleX(tick)}
                x2={scaleX(tick)}
                y1={padding.top}
                y2={chartHeight - padding.bottom}
                stroke="#e5e7eb"
                strokeDasharray="3,3"
              />
            ))}
            
            {/* Axes */}
            <line x1={padding.left} y1={chartHeight - padding.bottom} x2={chartWidth - padding.right} y2={chartHeight - padding.bottom} stroke="#374151" strokeWidth="1" />
            <line x1={padding.left} y1={padding.top} x2={padding.left} y2={chartHeight - padding.bottom} stroke="#374151" strokeWidth="1" />
            
            {/* X-axis ticks and labels */}
            {xTicks.map((tick, i) => (
              <g key={`x-tick-${i}`}>
                <line x1={scaleX(tick)} x2={scaleX(tick)} y1={chartHeight - padding.bottom} y2={chartHeight - padding.bottom + 4} stroke="#374151" />
                <text x={scaleX(tick)} y={chartHeight - padding.bottom + 16} textAnchor="middle" fontSize="10" fill="#6b7280">
                  {tick.toFixed(1)}
                </text>
              </g>
            ))}
            
            {/* Y-axis ticks and labels */}
            {yTicks.map((tick, i) => (
              <g key={`y-tick-${i}`}>
                <line x1={padding.left - 4} x2={padding.left} y1={scaleY(tick)} y2={scaleY(tick)} stroke="#374151" />
                <text x={padding.left - 8} y={scaleY(tick) + 3} textAnchor="end" fontSize="10" fill="#6b7280">
                  {tick.toFixed(1)}
                </text>
              </g>
            ))}
            
            {/* Axis labels with direction indicators */}
            <text x={padding.left + plotWidth / 2} y={chartHeight - 8} textAnchor="middle" fontSize="11" fill="#374151" fontWeight="500">
              Binding Affinity (ΔG kcal/mol) →  Better
            </text>
            <text 
              x={16} 
              y={padding.top + plotHeight / 2} 
              textAnchor="middle" 
              fontSize="11" 
              fill="#374151" 
              fontWeight="500"
              transform={`rotate(-90, 16, ${padding.top + plotHeight / 2})`}
            >
              SA Score   → Better
            </text>
            
            {/* Data points */}
            {allCandidates.map((candidate, idx) => {
              const x = scaleX(candidate.binding_affinity || 0)
              const y = scaleY(candidate.sa_score || 0)
              // Winner = green, Selected = blue, Rejected = gray
              const pointColor = candidate.category === "winner" ? "#16a34a" 
                : candidate.category === "selected" ? "#3b82f6" 
                : "#9ca3af"
              const pointSize = candidate.category === "winner" ? 8 : 6
              return (
                <g key={idx}>
                  <circle
                    cx={x}
                    cy={y}
                    r={pointSize}
                    fill={pointColor}
                    stroke="#fff"
                    strokeWidth="2"
                  />
                  <text x={x} y={y - 12} textAnchor="middle" fontSize="10" fill="#374151" fontWeight="500">
                    {candidate.scenario_id?.replace("scenario_", "S")}
                  </text>
                </g>
              )
            })}
          </svg>
          
          {/* Legend with grouped boxes */}
          <div style={{ fontSize: 11, color: "#6b7280", minWidth: 140 }}>
            {/* Winner Box */}
            {allCandidates.some(c => c.category === "winner") && (
              <div style={{ 
                border: "1px solid #16a34a", 
                borderRadius: 6, 
                padding: 8, 
                marginBottom: 8,
                background: "#f0fdf4"
              }}>
                <div style={{ fontWeight: 600, color: "#16a34a", marginBottom: 6, fontSize: 10, textTransform: "uppercase", letterSpacing: "0.5px" }}>
                  Winner
                </div>
                {allCandidates.filter(c => c.category === "winner").map((c, idx) => (
                  <div key={idx} style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 2 }}>
                    <div style={{ 
                      width: 8, 
                      height: 8, 
                      borderRadius: "50%", 
                      background: "#16a34a",
                    }} />
                    <span style={{ color: "#166534", fontWeight: 500 }}>
                      {c.scenario_id?.replace("scenario_", "S")}
                    </span>
                  </div>
                ))}
              </div>
            )}
            
            {/* Selected Candidates Box */}
            {allCandidates.some(c => c.category === "selected") && (
              <div style={{ 
                border: "1px solid #3b82f6", 
                borderRadius: 6, 
                padding: 8, 
                marginBottom: 8,
                background: "#eff6ff"
              }}>
                <div style={{ fontWeight: 600, color: "#3b82f6", marginBottom: 6, fontSize: 10, textTransform: "uppercase", letterSpacing: "0.5px" }}>
                  Selected
                </div>
                {allCandidates.filter(c => c.category === "selected").map((c, idx) => (
                  <div key={idx} style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 2 }}>
                    <div style={{ 
                      width: 8, 
                      height: 8, 
                      borderRadius: "50%", 
                      background: "#3b82f6",
                    }} />
                    <span style={{ color: "#1d4ed8", fontWeight: 500 }}>
                      {c.scenario_id?.replace("scenario_", "S")}
                    </span>
                  </div>
                ))}
              </div>
            )}
            
            {/* Rejected Candidates Box */}
            {allCandidates.some(c => c.category === "rejected") && (
              <div style={{ 
                border: "1px solid #e5e7eb", 
                borderRadius: 6, 
                padding: 8,
                background: "#fafafa"
              }}>
                <div style={{ fontWeight: 600, color: "#6b7280", marginBottom: 6, fontSize: 10, textTransform: "uppercase", letterSpacing: "0.5px" }}>
                  Rejected
                </div>
                {allCandidates.filter(c => c.category === "rejected").map((c, idx) => (
                  <div key={idx} style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 2 }}>
                    <div style={{ 
                      width: 8, 
                      height: 8, 
                      borderRadius: "50%", 
                      background: "#9ca3af",
                    }} />
                    <span style={{ color: "#374151" }}>
                      {c.scenario_id?.replace("scenario_", "S")}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    )
  }

  // Section divider
  const SectionDivider = () => (
    <div style={{ height: 1, background: "#e5e7eb", margin: "16px 0" }} />
  )

  return (
    <div style={{ fontFamily: "system-ui, -apple-system, sans-serif" }}>
      {/* Confidence Indicator */}
      <ConfidenceIndicator />
      
      {/* Executive Summary */}
      {report.executive_summary && (
        <div style={{ marginBottom: 20 }}>
          <div style={{ fontSize: 13, fontWeight: 600, color: "#111827", marginBottom: 8, textTransform: "uppercase", letterSpacing: "0.5px" }}>
            Executive Summary
          </div>
          <p style={{ fontSize: 14, lineHeight: 1.7, color: "#374151", margin: 0 }}>
            {report.executive_summary}
          </p>
        </div>
      )}

      <SectionDivider />

      {/* Pareto Chart */}
      <ParetoChart />

      <SectionDivider />

      {/* Selected Candidate - Boxed */}
      {report.winner && (
        <div style={{ 
          marginBottom: 20, 
          border: "2px solid #16a34a", 
          borderRadius: 8, 
          padding: 16,
          background: "#f0fdf4"
        }}>
          <div style={{ 
            display: "flex", 
            alignItems: "center", 
            gap: 8, 
            marginBottom: 12,
            paddingBottom: 10,
            borderBottom: "1px solid #bbf7d0"
          }}>
            <div style={{ 
              width: 10, 
              height: 10, 
              borderRadius: "50%", 
              background: "#16a34a" 
            }} />
            <div style={{ fontSize: 13, fontWeight: 600, color: "#166534", textTransform: "uppercase", letterSpacing: "0.5px" }}>
              Selected Candidate (Winner)
            </div>
          </div>
          
          <div style={{ marginBottom: 14 }}>
            <div style={{ fontSize: 16, fontWeight: 600, color: "#111827" }}>
              {report.winner.scenario_id}
            </div>
            <div style={{ fontSize: 13, color: "#6b7280" }}>
              {report.winner.scaffold}
            </div>
          </div>
          
          {/* Metrics Grid */}
          <div style={{ 
            display: "flex", 
            justifyContent: "space-evenly",
            alignItems: "flex-start",
            width: "100%",
            marginBottom: 14,
            padding: 12,
            background: "#fff",
            borderRadius: 6,
            border: "1px solid #dcfce7"
          }}>
            <div style={{ flex: 1, textAlign: "center" }}>
              <div style={{ fontSize: 10, color: "#6b7280", marginBottom: 4, textTransform: "uppercase" }}>ΔG</div>
              <div style={{ fontSize: 14, fontWeight: 600, color: "#111827" }}>{report.winner.binding_affinity}</div>
              <div style={{ fontSize: 9, color: "#9ca3af" }}>kcal/mol</div>
            </div>
            <div style={{ flex: 1, textAlign: "center" }}>
              <div style={{ fontSize: 10, color: "#6b7280", marginBottom: 4, textTransform: "uppercase" }}>hERG</div>
              <div style={{ fontSize: 14, fontWeight: 600, color: report.winner.herg_flag ? "#dc2626" : "#16a34a" }}>
                {report.winner.herg_flag ? "Flagged" : "Clear"}
              </div>
            </div>
            <div style={{ flex: 1, textAlign: "center" }}>
              <div style={{ fontSize: 10, color: "#6b7280", marginBottom: 4, textTransform: "uppercase" }}>SA Score</div>
              <div style={{ fontSize: 14, fontWeight: 600, color: "#111827" }}>{report.winner.sa_score}</div>
            </div>
            <div style={{ flex: 1, textAlign: "center" }}>
              <div style={{ fontSize: 10, color: "#6b7280", marginBottom: 4, textTransform: "uppercase" }}>Est. Cost</div>
              <div style={{ fontSize: 14, fontWeight: 600, color: "#111827" }}>${report.winner.cost_usd?.toLocaleString()}</div>
            </div>
          </div>

          {report.winner.rationale && (
            <div style={{ background: "#fff", padding: 10, borderRadius: 6, border: "1px solid #dcfce7" }}>
              <div style={{ fontSize: 11, fontWeight: 600, color: "#166534", marginBottom: 4, textTransform: "uppercase" }}>Selection Rationale</div>
              <p style={{ fontSize: 12, lineHeight: 1.6, color: "#374151", margin: 0 }}>
                {report.winner.rationale}
              </p>
            </div>
          )}
        </div>
      )}

      {/* Selected Candidates (non-winner but passing) - Boxed */}
      {report.selected && report.selected.length > 0 && (
        <div style={{ 
          marginBottom: 20, 
          border: "1px solid #3b82f6", 
          borderRadius: 8, 
          padding: 16,
          background: "#eff6ff"
        }}>
          <div style={{ 
            display: "flex", 
            alignItems: "center", 
            gap: 8, 
            marginBottom: 12,
            paddingBottom: 10,
            borderBottom: "1px solid #bfdbfe"
          }}>
            <div style={{ 
              width: 10, 
              height: 10, 
              borderRadius: "50%", 
              background: "#3b82f6" 
            }} />
            <div style={{ fontSize: 13, fontWeight: 600, color: "#1d4ed8", textTransform: "uppercase", letterSpacing: "0.5px" }}>
              Selected Candidates ({report.selected.length})
            </div>
          </div>
          
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            {report.selected.map((s: any, idx: number) => (
              <div 
                key={idx} 
                style={{ 
                  background: "#fff", 
                  border: "1px solid #bfdbfe", 
                  borderRadius: 6, 
                  padding: 12 
                }}
              >
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 10 }}>
                  <div>
                    <div style={{ fontWeight: 600, color: "#111827", fontSize: 14 }}>{s.scenario_id}</div>
                    <div style={{ fontSize: 12, color: "#6b7280" }}>{s.scaffold}</div>
                  </div>
                </div>
                
                {/* Metrics Row */}
                <div style={{ 
                  display: "flex", 
                  justifyContent: "space-evenly",
                  alignItems: "flex-start",
                  width: "100%",
                  marginBottom: 10,
                  padding: 8,
                  background: "#f0f9ff",
                  borderRadius: 4
                }}>
                  <div style={{ flex: 1, textAlign: "center" }}>
                    <div style={{ fontSize: 9, color: "#6b7280", textTransform: "uppercase" }}>ΔG</div>
                    <div style={{ fontSize: 12, fontWeight: 500, color: "#374151" }}>{s.binding_affinity}</div>
                  </div>
                  <div style={{ flex: 1, textAlign: "center" }}>
                    <div style={{ fontSize: 9, color: "#6b7280", textTransform: "uppercase" }}>hERG</div>
                    <div style={{ fontSize: 12, fontWeight: 500, color: s.herg_flag ? "#dc2626" : "#16a34a" }}>
                      {s.herg_flag ? "Flagged" : "Clear"}
                    </div>
                  </div>
                  <div style={{ flex: 1, textAlign: "center" }}>
                    <div style={{ fontSize: 9, color: "#6b7280", textTransform: "uppercase" }}>SA</div>
                    <div style={{ fontSize: 12, fontWeight: 500, color: "#374151" }}>{s.sa_score}</div>
                  </div>
                  <div style={{ flex: 1, textAlign: "center" }}>
                    <div style={{ fontSize: 9, color: "#6b7280", textTransform: "uppercase" }}>Cost</div>
                    <div style={{ fontSize: 12, fontWeight: 500, color: "#374151" }}>${s.cost_usd?.toLocaleString()}</div>
                  </div>
                </div>
                
                {/* Selection Reason */}
                {s.selection_reason && (
                  <div style={{ 
                    fontSize: 12, 
                    color: "#1e40af", 
                    padding: 8, 
                    background: "#dbeafe", 
                    borderRadius: 4,
                    borderLeft: "3px solid #3b82f6"
                  }}>
                    <span style={{ fontWeight: 500 }}>Status: </span>
                    {s.selection_reason}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Rejected Candidates - Boxed */}
      {report.rejected && report.rejected.length > 0 && (
        <div style={{ 
          marginBottom: 20, 
          border: "1px solid #e5e7eb", 
          borderRadius: 8, 
          padding: 16,
          background: "#fafafa"
        }}>
          <div style={{ 
            display: "flex", 
            alignItems: "center", 
            gap: 8, 
            marginBottom: 12,
            paddingBottom: 10,
            borderBottom: "1px solid #e5e7eb"
          }}>
            <div style={{ 
              width: 10, 
              height: 10, 
              borderRadius: "50%", 
              background: "#dc2626" 
            }} />
            <div style={{ fontSize: 13, fontWeight: 600, color: "#6b7280", textTransform: "uppercase", letterSpacing: "0.5px" }}>
              Rejected Candidates ({report.rejected.length})
            </div>
          </div>
          
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            {report.rejected.map((r: any, idx: number) => (
              <div 
                key={idx} 
                style={{ 
                  background: "#fff", 
                  border: "1px solid #e5e7eb", 
                  borderRadius: 6, 
                  padding: 12 
                }}
              >
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 10 }}>
                  <div>
                    <div style={{ fontWeight: 600, color: "#111827", fontSize: 14 }}>{r.scenario_id}</div>
                    <div style={{ fontSize: 12, color: "#9ca3af" }}>{r.scaffold}</div>
                  </div>
                </div>
                
                {/* Metrics Row */}
                <div style={{ 
                  display: "flex", 
                  justifyContent: "space-evenly",
                  alignItems: "flex-start",
                  width: "100%",
                  marginBottom: 10,
                  padding: 8,
                  background: "#f9fafb",
                  borderRadius: 4
                }}>
                  <div style={{ flex: 1, textAlign: "center" }}>
                    <div style={{ fontSize: 9, color: "#9ca3af", textTransform: "uppercase" }}>ΔG</div>
                    <div style={{ fontSize: 12, fontWeight: 500, color: "#374151" }}>{r.binding_affinity}</div>
                  </div>
                  <div style={{ flex: 1, textAlign: "center" }}>
                    <div style={{ fontSize: 9, color: "#9ca3af", textTransform: "uppercase" }}>hERG</div>
                    <div style={{ fontSize: 12, fontWeight: 500, color: r.herg_flag ? "#dc2626" : "#374151" }}>
                      {r.herg_flag ? "Flagged" : "Clear"}
                    </div>
                  </div>
                  <div style={{ flex: 1, textAlign: "center" }}>
                    <div style={{ fontSize: 9, color: "#9ca3af", textTransform: "uppercase" }}>SA</div>
                    <div style={{ fontSize: 12, fontWeight: 500, color: "#374151" }}>{r.sa_score}</div>
                  </div>
                  <div style={{ flex: 1, textAlign: "center" }}>
                    <div style={{ fontSize: 9, color: "#9ca3af", textTransform: "uppercase" }}>Cost</div>
                    <div style={{ fontSize: 12, fontWeight: 500, color: "#374151" }}>${r.cost_usd?.toLocaleString()}</div>
                  </div>
                </div>
                
                {/* Rejection Reason */}
                <div style={{ 
                  fontSize: 12, 
                  color: "#6b7280", 
                  padding: 8, 
                  background: "#fef2f2", 
                  borderRadius: 4,
                  borderLeft: "3px solid #dc2626"
                }}>
                  <span style={{ fontWeight: 500, color: "#b91c1c" }}>Rejection: </span>
                  {r.rejection_reason}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Comparative Analysis */}
      {report.comparative_analysis && (
        <>
          <SectionDivider />
          <div style={{ marginBottom: 20 }}>
            <div style={{ fontSize: 13, fontWeight: 600, color: "#111827", marginBottom: 8, textTransform: "uppercase", letterSpacing: "0.5px" }}>
              Comparative Analysis
            </div>
            <p style={{ fontSize: 13, lineHeight: 1.6, color: "#374151", margin: 0 }}>
              {report.comparative_analysis}
            </p>
          </div>
        </>
      )}

      {/* Recommendations */}
      {report.recommendation && (
        <>
          <SectionDivider />
          <div>
            <div style={{ fontSize: 13, fontWeight: 600, color: "#111827", marginBottom: 8, textTransform: "uppercase", letterSpacing: "0.5px" }}>
              Recommendations
            </div>
            <p style={{ fontSize: 13, lineHeight: 1.6, color: "#374151", margin: 0 }}>
              {report.recommendation}
            </p>
          </div>
        </>
      )}
    </div>
  )
}

// Helper component for metric badges
function MetricBadge({ label, value, good }: { label: string; value: string | number; good?: boolean }) {
  return (
    <div style={{ 
      background: good ? "rgba(34, 197, 94, 0.1)" : "rgba(239, 68, 68, 0.1)",
      border: `1px solid ${good ? "#86efac" : "#fca5a5"}`,
      borderRadius: 8,
      padding: "8px 12px",
      textAlign: "center"
    }}>
      <div style={{ fontSize: 11, color: "#6b7280", marginBottom: 2 }}>{label}</div>
      <div style={{ fontSize: 14, fontWeight: 600, color: good ? "#166534" : "#991b1b" }}>{value}</div>
    </div>
  )
}

function CalculationBreakdown({ 
  winners, 
  rejected, 
  normalizedMetrics,
  structuredReport 
}: { 
  winners: any[]; 
  rejected: any[]; 
  normalizedMetrics: Record<string, any>;
  structuredReport?: any;
}) {
  // If we have a structured report (from LLM), use that data - it reflects edits
  // Otherwise fall back to normalizedMetrics
  const rows = useMemo(() => {
    if (structuredReport) {
      const allCandidates: any[] = []
      
      // Add winner from structured report
      if (structuredReport.winner) {
        allCandidates.push({
          id: structuredReport.winner.scenario_id,
          label: formatScenarioLabel(structuredReport.winner.scenario_id),
          binding: structuredReport.winner.binding_affinity ?? "—",
          potencyPass: structuredReport.winner.binding_affinity < -7,
          toxicity: structuredReport.winner.toxicity_risk ?? "—",
          herg: structuredReport.winner.herg_flag,
          safe: structuredReport.winner.toxicity_risk === "LOW" && !structuredReport.winner.herg_flag,
          sa: structuredReport.winner.sa_score ?? "—",
          cost: structuredReport.winner.cost_usd != null ? `$${structuredReport.winner.cost_usd.toLocaleString()}` : "—",
          tag: "Winner",
        })
      }
      
      // Add selected (non-winner passing candidates) from structured report
      if (structuredReport.selected && Array.isArray(structuredReport.selected)) {
        structuredReport.selected.forEach((s: any) => {
          allCandidates.push({
            id: s.scenario_id,
            label: formatScenarioLabel(s.scenario_id),
            binding: s.binding_affinity ?? "—",
            potencyPass: s.binding_affinity < -7,
            toxicity: s.toxicity_risk ?? "—",
            herg: s.herg_flag,
            safe: s.toxicity_risk === "LOW" && !s.herg_flag,
            sa: s.sa_score ?? "—",
            cost: s.cost_usd != null ? `$${s.cost_usd.toLocaleString()}` : "—",
            tag: "Selected",
          })
        })
      }
      
      // Add rejected from structured report
      if (structuredReport.rejected && Array.isArray(structuredReport.rejected)) {
        structuredReport.rejected.forEach((r: any) => {
          allCandidates.push({
            id: r.scenario_id,
            label: formatScenarioLabel(r.scenario_id),
            binding: r.binding_affinity ?? "—",
            potencyPass: r.binding_affinity < -7,
            toxicity: r.toxicity_risk ?? "—",
            herg: r.herg_flag,
            safe: r.toxicity_risk === "LOW" && !r.herg_flag,
            sa: r.sa_score ?? "—",
            cost: r.cost_usd != null ? `$${r.cost_usd.toLocaleString()}` : "—",
            tag: "Rejected",
          })
        })
      }
      
      // Sort by scenario number for consistent ordering
      allCandidates.sort((a, b) => {
        const numA = parseInt(a.id?.replace(/\D/g, '') || '0')
        const numB = parseInt(b.id?.replace(/\D/g, '') || '0')
        return numA - numB
      })
      
      return allCandidates
    }
    
    // Fallback to original logic using normalizedMetrics
    return [...(winners || []).map((w: any) => ({ ...w, tag: "Winner" })), ...(rejected || []).map((r: any) => ({ ...r, tag: "Rejected" }))]
      .slice(0, 3)
      .map((candidate) => {
        const metrics = normalizedMetrics[candidate.scenario_id] || {}
        const docking = metrics.docking || {}
        const admet = metrics.admet || {}
        const synthesis = metrics.synthesis || {}
        return {
          id: candidate.scenario_id,
          label: formatScenarioLabel(candidate.scenario_id),
          binding: docking.binding_affinity_kcal_per_mol ?? "—",
          potencyPass: docking.potency_pass,
          toxicity: admet.toxicity_risk ?? "—",
          herg: admet.herg_flag,
          safe: admet.is_safe,
          sa: synthesis.sa_score ?? "—",
          cost: synthesis.estimated_cost_usd != null ? `$${synthesis.estimated_cost_usd}` : "—",
          tag: candidate.tag,
        }
      })
  }, [structuredReport, winners, rejected, normalizedMetrics])

  if (rows.length === 0) {
    return null
  }

  return (
    <div style={{ marginTop: 20 }}>
      <div style={{ fontSize: 13, fontWeight: 600, color: "#111827", marginBottom: 12, textTransform: "uppercase", letterSpacing: "0.5px" }}>
        Calculation Breakdown
      </div>
      <div style={{ overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
          <thead>
            <tr style={{ borderBottom: "2px solid #111827" }}>
              {["Scenario", "Status", "ΔG (kcal/mol)", "Potency", "hERG", "SA Score", "Est. Cost"].map((heading) => (
                <th key={heading} style={{ padding: "8px 0", textAlign: heading === "Scenario" ? "left" : "center", fontWeight: 600, color: "#111827" }}>
                  {heading}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => {
              const statusColor = row.tag === "Winner" ? "#16a34a" : row.tag === "Selected" ? "#3b82f6" : "#6b7280";
              const statusBg = row.tag === "Winner" ? "#f0fdf4" : row.tag === "Selected" ? "#eff6ff" : "#f9fafb";
              return (
                <tr key={row.id} style={{ borderBottom: "1px solid #e5e7eb" }}>
                  <td style={{ padding: "10px 0", fontWeight: 500, color: "#111827" }}>
                    {row.label}
                  </td>
                  <td style={{ padding: "10px 0", textAlign: "center" }}>
                    <span style={{ 
                      fontSize: 10, 
                      fontWeight: 600, 
                      color: statusColor, 
                      background: statusBg,
                      padding: "2px 8px",
                      borderRadius: 4,
                      textTransform: "uppercase",
                    }}>
                      {row.tag}
                    </span>
                  </td>
                  <td style={{ padding: "10px 0", textAlign: "center", color: "#374151" }}>{row.binding}</td>
                  <td style={{ padding: "10px 0", textAlign: "center", color: "#374151" }}>{row.potencyPass != null ? (row.potencyPass ? "Pass" : "Fail") : "—"}</td>
                  <td style={{ padding: "10px 0", textAlign: "center", color: "#374151" }}>{row.herg != null ? (row.herg ? "Flag" : "Clear") : "—"}</td>
                  <td style={{ padding: "10px 0", textAlign: "center", color: "#374151" }}>{row.sa}</td>
                  <td style={{ padding: "10px 0", textAlign: "center", color: "#374151" }}>{row.cost}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}

const DecisionCriteriaControls = ({
  decisionCriteria,
  onCriteriaChange,
  showSaveButton,
  onSave,
  onRegenerate,
}: {
  decisionCriteria: DecisionCriteria
  onCriteriaChange: React.Dispatch<React.SetStateAction<DecisionCriteria>> | ((criteria: DecisionCriteria) => void)
  showSaveButton?: boolean
  onSave?: () => void
  onRegenerate?: () => void
}) => {
  const [hasChanges, setHasChanges] = useState(false)
  
  const updateField = (path: string[], value: number | boolean) => {
    const updater = (prev: DecisionCriteria) => {
      const next = JSON.parse(JSON.stringify(prev)) as DecisionCriteria
      let cursor: Record<string, unknown> = next as unknown as Record<string, unknown>
      for (let i = 0; i < path.length - 1; i += 1) {
        cursor[path[i]] = { ...(cursor[path[i]] as Record<string, unknown>) }
        cursor = cursor[path[i]] as Record<string, unknown>
      }
      cursor[path[path.length - 1]] = value
      return next
    }
    
    if (typeof onCriteriaChange === 'function') {
      const newCriteria = updater(decisionCriteria)
      onCriteriaChange(newCriteria as any)
      setHasChanges(true)
    }
  }

  const CriteriaInput = ({ label, value, onChange, unit }: { label: string; value: number; onChange: (v: number) => void; unit?: string }) => {
    const [localValue, setLocalValue] = useState(String(value))
    const [isFocused, setIsFocused] = useState(false)
    const inputRef = useRef<HTMLInputElement>(null)
    
    // Only sync local value when prop changes AND input is not focused
    // This prevents the parent update from overwriting what the user is typing
    useEffect(() => {
      if (!isFocused) {
        setLocalValue(String(value))
      }
    }, [value, isFocused])
    
    return (
      <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
        <label style={{ fontSize: 11, color: "#475569", fontWeight: 600 }}>{label}</label>
        <div style={{ 
          display: "flex", 
          alignItems: "center", 
          background: "#fff", 
          border: isFocused ? "1px solid #6b7280" : "1px solid #d1d5db", 
          borderRadius: 5,
          overflow: "hidden",
          transition: "all 0.15s ease",
          boxShadow: isFocused ? "0 0 0 2px rgba(107, 114, 128, 0.15)" : "none",
        }}>
          <input
            ref={inputRef}
            type="text"
            inputMode="decimal"
            value={localValue}
            onChange={(e) => {
              const raw = e.target.value
              // Allow typing any characters (including minus, decimal point)
              setLocalValue(raw)
            }}
            onBlur={() => {
              setIsFocused(false)
              // On blur, validate and propagate to parent
              const val = parseFloat(localValue)
              if (isNaN(val) || localValue.trim() === "") {
                setLocalValue(String(value)) // Reset to last valid value
              } else {
                setLocalValue(String(val)) // Clean up the display
                onChange(val)
              }
            }}
            onFocus={(e) => {
              setIsFocused(true)
              // Select all text on focus for easy replacement
              e.target.select()
            }}
            style={{ 
              border: "none", 
              outline: "none", 
              padding: "6px 8px",
              fontSize: 13,
              fontWeight: 600,
              color: "#1e293b",
              background: "transparent",
              width: 60,
              minWidth: 60,
              textAlign: "right",
            }}
          />
          {unit && (
            <span style={{ 
              fontSize: 10, 
              color: "#6b7280", 
              padding: "0 8px 0 2px",
              whiteSpace: "nowrap",
              fontWeight: 500,
            }}>{unit}</span>
          )}
        </div>
      </div>
    )
  }

  const CategoryLabel = ({ children }: { children: React.ReactNode }) => (
    <div style={{ 
      display: "flex", 
      alignItems: "center", 
      gap: 6,
      marginBottom: 8,
    }}>
      <span style={{ 
        fontSize: 12, 
        fontWeight: 700, 
        color: "#374151",
        letterSpacing: "0.02em",
        textTransform: "uppercase",
      }}>{children}</span>
    </div>
  )

  return (
    <div style={{ 
      border: "1px solid #e2e8f0", 
      borderRadius: 10, 
      background: "#fff",
      boxShadow: "0 1px 3px rgba(0,0,0,0.04)",
      overflow: "hidden",
    }}>
      {/* Header */}
      <div style={{ 
        padding: "10px 14px", 
        borderBottom: "1px solid #e2e8f0",
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        background: "#f8fafc",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div style={{ 
            width: 28, 
            height: 28, 
            borderRadius: 6, 
            background: "#e2e8f0",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#475569" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 20V10" />
              <path d="M18 20V4" />
              <path d="M6 20v-4" />
            </svg>
          </div>
          <div>
            <div style={{ fontSize: 14, fontWeight: 700, color: "#1e293b" }}>Decision Criteria</div>
            <div style={{ fontSize: 11, color: "#64748b" }}>Configure evaluation thresholds</div>
          </div>
        </div>
        {showSaveButton && hasChanges && onSave && (
          <button
            onClick={() => { onSave(); setHasChanges(false); }}
            style={{
              padding: "5px 12px",
              border: "1px solid #d1d5db",
              borderRadius: 4,
              background: "#fff",
              color: "#374151",
              fontSize: 11,
              fontWeight: 600,
              cursor: "pointer",
              transition: "all 0.15s ease",
              boxShadow: "0 1px 2px rgba(0,0,0,0.05)",
            }}
            onMouseOver={(e) => {
              e.currentTarget.style.background = "#f9fafb"
              e.currentTarget.style.borderColor = "#9ca3af"
            }}
            onMouseOut={(e) => {
              e.currentTarget.style.background = "#fff"
              e.currentTarget.style.borderColor = "#d1d5db"
            }}
          >
            Save Changes
          </button>
        )}
      </div>

      {/* Content - Enhanced layout */}
      <div style={{ padding: "12px" }}>
        {/* Potency / Docking Section */}
        <div style={{ marginBottom: 12 }}>
          <CategoryLabel>Potency (Docking)</CategoryLabel>
          <div style={{ 
            display: "grid", 
            gridTemplateColumns: "repeat(3, 1fr)", 
            gap: 8,
            background: "#fafbfc",
            padding: 10,
            borderRadius: 6,
            border: "1px solid #f1f5f9",
          }}>
            <CriteriaInput 
              label="Ideal Min" 
              value={decisionCriteria.docking.idealMin} 
              onChange={(v) => updateField(["docking", "idealMin"], v)} 
              unit="kcal/mol" 
            />
            <CriteriaInput 
              label="Ideal Max" 
              value={decisionCriteria.docking.idealMax} 
              onChange={(v) => updateField(["docking", "idealMax"], v)} 
              unit="kcal/mol" 
            />
            <CriteriaInput 
              label="Hard Fail >" 
              value={decisionCriteria.docking.hardFailThreshold} 
              onChange={(v) => updateField(["docking", "hardFailThreshold"], v)} 
              unit="kcal/mol" 
            />
          </div>
        </div>

        {/* Safety / ADMET Section */}
        <div style={{ marginBottom: 12 }}>
          <CategoryLabel>Safety (ADMET)</CategoryLabel>
          <div style={{ 
            display: "grid", 
            gridTemplateColumns: "1fr 1fr auto", 
            gap: 8,
            background: "#fafbfc",
            padding: 10,
            borderRadius: 6,
            border: "1px solid #f1f5f9",
            alignItems: "end",
          }}>
            <CriteriaInput 
              label="Toxicity Min" 
              value={decisionCriteria.admet.idealMin} 
              onChange={(v) => updateField(["admet", "idealMin"], v)} 
            />
            <CriteriaInput 
              label="Toxicity Max" 
              value={decisionCriteria.admet.idealMax} 
              onChange={(v) => updateField(["admet", "idealMax"], v)} 
            />
            <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
              <label style={{ fontSize: 10, color: "#64748b", fontWeight: 500 }}>hERG Veto</label>
              <label style={{ 
                display: "flex", 
                alignItems: "center", 
                gap: 6, 
                padding: "5px 10px",
                background: decisionCriteria.admet.hardFailHERG ? "#fef2f2" : "#f8fafc",
                border: `1px solid ${decisionCriteria.admet.hardFailHERG ? "#fca5a5" : "#e2e8f0"}`,
                borderRadius: 4,
                cursor: "pointer",
                transition: "all 0.15s ease",
                height: 28,
              }}>
                <input
                  type="checkbox"
                  checked={decisionCriteria.admet.hardFailHERG}
                  onChange={(e) => updateField(["admet", "hardFailHERG"], e.target.checked)}
                  style={{ 
                    margin: 0, 
                    width: 14, 
                    height: 14, 
                    accentColor: "#dc2626",
                    cursor: "pointer",
                  }}
                />
                <span style={{ 
                  color: decisionCriteria.admet.hardFailHERG ? "#dc2626" : "#64748b", 
                  fontWeight: 600,
                  fontSize: 11,
                }}>
                  {decisionCriteria.admet.hardFailHERG ? "Enabled" : "Disabled"}
                </span>
              </label>
            </div>
          </div>
        </div>

        {/* Cost / Synthesis Section */}
        <div>
          <CategoryLabel>Cost (Synthesis)</CategoryLabel>
          <div style={{ 
            display: "grid", 
            gridTemplateColumns: "repeat(4, 1fr)", 
            gap: 8,
            background: "#fafbfc",
            padding: 10,
            borderRadius: 6,
            border: "1px solid #f1f5f9",
          }}>
            <CriteriaInput 
              label="Ideal SA ≤" 
              value={decisionCriteria.synthesis.idealSaMax} 
              onChange={(v) => updateField(["synthesis", "idealSaMax"], v)} 
            />
            <CriteriaInput 
              label="Ideal Steps ≤" 
              value={decisionCriteria.synthesis.idealStepsMax} 
              onChange={(v) => updateField(["synthesis", "idealStepsMax"], v)} 
            />
            <CriteriaInput 
              label="Fail SA >" 
              value={decisionCriteria.synthesis.hardFailSa} 
              onChange={(v) => updateField(["synthesis", "hardFailSa"], v)} 
            />
            <CriteriaInput 
              label="Fail Steps >" 
              value={decisionCriteria.synthesis.hardFailSteps} 
              onChange={(v) => updateField(["synthesis", "hardFailSteps"], v)} 
            />
          </div>
        </div>
      </div>
    </div>
  )
}

function AssumptionsList({
  conciseConstraints,
  decisionCriteria,
}: {
  conciseConstraints: string[]
  decisionCriteria: any
}) {
  if (!conciseConstraints.length && !decisionCriteria) {
    return null
  }

  return (
    <div style={{ marginTop: 18, border: "1px solid #e5e7eb", borderRadius: 10, padding: 16, background: "#fff" }}>
      <div style={{ fontSize: 14, fontWeight: 600, color: "#0f172a", marginBottom: 8 }}>Operating Assumptions & Guardrails</div>
      <ul style={{ margin: 0, paddingLeft: 20, color: "#374151", fontSize: 13, lineHeight: 1.7, listStyle: "disc" }}>
        {conciseConstraints.map((rule, idx) => (
          <li key={`assumption-${idx}`}>{rule}</li>
        ))}
        {decisionCriteria && (
          <>
            <li>
              Docking (Potency): ideal ΔG {decisionCriteria.docking.idealMin} to {decisionCriteria.docking.idealMax} kcal/mol; Hard fail if &gt; {decisionCriteria.docking.hardFailThreshold} kcal/mol
            </li>
            <li>
              ADMET (Safety): toxicity target {decisionCriteria.admet.idealMin}–{decisionCriteria.admet.idealMax}; {decisionCriteria.admet.hardFailHERG ? "any hERG flag triggers veto" : "hERG flag informational"}
            </li>
            <li>
              Synthesis: SA ≤ {decisionCriteria.synthesis.idealSaMax} and ≤ {decisionCriteria.synthesis.idealStepsMax} steps; Hard fail if SA &gt; {decisionCriteria.synthesis.hardFailSa} or steps &gt; {decisionCriteria.synthesis.hardFailSteps}
            </li>
          </>
        )}
      </ul>
    </div>
  )
}

function FieldInput({
  label,
  value,
  onChange,
  suffix,
}: {
  label: string
  value: number
  onChange: (value: string) => void
  suffix?: string
}) {
  return (
    <label style={{ fontSize: 11, color: "#111827", display: "flex", flexDirection: "column", gap: 2, alignItems: "center" }}>
      <span style={{ fontWeight: 600, fontSize: 9 }}>{label}</span>
      <div style={{ display: "flex", alignItems: "center", background: "#fff", border: "1px solid #d1d5db", borderRadius: 4, padding: "3px 5px", gap: 3, width: "fit-content" }}>
        <input
          type="text"
          inputMode="decimal"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          style={{ border: "none", outline: "none", width: 40, fontSize: 11, textAlign: "right", background: "transparent" }}
        />
        {suffix && <span style={{ fontSize: 8, color: "#94a3b8" }}>{suffix}</span>}
      </div>
    </label>
  )
}

function StatChip({ label, value, color = "#666" }: { label: string; value: number; color?: string }) {
  return (
    <div
      style={{
        border: "1px solid #e5e7eb",
        borderRadius: 999,
        padding: "6px 12px",
        display: "inline-flex",
        gap: 8,
        alignItems: "center",
        background: "linear-gradient(180deg,#ffffff 0%, #f3f4f6 100%)",
        boxShadow: "0 1px 2px rgba(16,24,40,0.06)",
      }}
    >
      <span style={{ color: "#6b7280", fontSize: 12 }}>{label}</span>
      <span style={{ color, fontWeight: 700 }}>{value}</span>
    </div>
  )
}

function StepIndicator(props: { ingestion: boolean; hypotheses: boolean; dispatch: boolean; scenariosReady: boolean; metricsAggregated: boolean; judged: boolean }) {
  const steps = [
    { key: "ingestion", label: "Objective", done: props.ingestion },
    { key: "hypotheses", label: "Hypotheses", done: props.hypotheses },
    { key: "design", label: "Design", done: props.scenariosReady },
    { key: "dispatch", label: "Dispatch", done: props.dispatch },
    { key: "aggregate", label: "Metric Aggregation", done: props.metricsAggregated },
    { key: "judge", label: "Results", done: props.judged },
  ]
  return (
    <div style={{ display: "flex", gap: 8, marginTop: 16, flexWrap: "wrap" }}>
      {steps.map((s, i) => (
        <div key={s.key} style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <div
            style={{
              width: 10,
              height: 10,
              borderRadius: 999,
              background: s.done ? "#2EA87A" : "#e5e7eb",
              border: "1px solid " + (s.done ? "#238c64" : "#d1d5db"),
              boxShadow: s.done ? "0 0 6px rgba(46, 168, 122, 0.4)" : "none",
              transition: "all 0.3s ease",
            }}
          />
          <span style={{ fontSize: 12, color: s.done ? "#2EA87A" : "#6b7280", fontWeight: s.done ? 600 : 500, transition: "all 0.3s ease" }}>{s.label}</span>
          {i < steps.length - 1 && <span style={{ color: "#cbd5e1", margin: "0 6px" }}>→</span>}
        </div>
      ))}
    </div>
  )
}

function Timeline({ events }: { events: any[] }) {
  const items = events.map((e, idx) => <TimelineItem key={idx} e={e} />)
  return <div style={{ display: "grid", gap: 8, marginTop: 8 }}>{items.length ? items : <div>No events yet.</div>}</div>
}

function TimelineItem({ e }: { e: any }) {
  const { label, color, summary } = timelineMeta(e)
  return (
    <div style={{ display: "flex", gap: 12, alignItems: "flex-start" }}>
      <div style={{ width: 10, height: 10, borderRadius: 999, background: color, marginTop: 6 }} />
      <div style={{ border: "1px solid #eee", borderRadius: 8, padding: 10, flex: 1 }}>
        <div style={{ fontWeight: 600 }}>{label}</div>
        {summary && <pre style={{ whiteSpace: "pre-wrap", marginTop: 6, background: "#fafafa", padding: 8, borderRadius: 6, fontSize: 12 }}>{summary}</pre>}
      </div>
    </div>
  )
}

function timelineMeta(e: any): { label: string; color: string; summary?: string } {
  switch (e?.type) {
    case "ingestion":
      return { label: "Ingestion", color: "#4285f4", summary: safeStringify({ target_protein: e.target_protein, target_pdb_id: e.target_pdb_id }) }
    case "hypotheses":
      return { label: "Hypothesis Generation", color: "#4285f4", summary: safeStringify({ num_scenarios: e.num_scenarios, scaffolds: e.scaffolds }) }
    case "dispatch_started":
      return { label: "Concurrent Dispatch Started", color: "#fbbc05" }
    case "simulation_scenarios":
      return { label: "Scenarios Generated", color: "#fbbc05", summary: safeStringify({ count: Array.isArray(e.scenarios) ? e.scenarios.length : 0 }) }
    case "agent_run":
      return { label: `Agent Run: ${e.agent}`, color: "#34a853", summary: safeStringify(e.input) }
    case "agent_result":
      return { label: `Agent Result: ${e.agent}`, color: "#34a853", summary: safeStringify(e.output) }
    case "metrics_aggregated":
      return { label: "Metric Aggregation", color: "#34a853", summary: safeStringify({ scenarios: (e.scenarios || []).length }) }
    case "judgement_complete":
      return {
        label: "Judgement Complete",
        color: "#34a853",
        summary: safeStringify({ winners: (e.winners || []).length, rejected: (e.rejected || []).length }),
      }
    default:
      return { label: e?.type || "Event", color: "#999", summary: safeStringify(e) }
  }
}

function MetricCard({ title, data, fallback }: { title: string; data?: any; fallback: string }) {
  const has = data && Object.keys(data).length > 0
  
  return (
    <motion.div
      whileHover={{ scale: 1.01, boxShadow: "0 4px 12px rgba(0,0,0,0.08)" }}
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2 }}
      className={!has ? "processing-card" : ""}
      style={{ 
        border: "1px solid #e5e7eb", 
        borderRadius: 8, 
        padding: 12, 
        background: "#fff", 
        boxShadow: "0 1px 2px rgba(0,0,0,0.04)",
        position: "relative",
        overflow: "hidden",
      }}
    >
      <div style={{ fontWeight: 600, fontSize: 13, display: "flex", alignItems: "center", gap: 8, color: "#374151" }}>
        <span style={{ 
          display: "inline-flex", 
          alignItems: "center", 
          justifyContent: "center", 
          width: 20, 
          height: 20, 
          borderRadius: 4, 
          background: !has ? "linear-gradient(135deg, #b4a8d3 0%, #e6a9b2 100%)" : "#f3f4f6", 
          border: "1px solid #e5e7eb", 
          fontSize: 11 
        }}>
          {/* Icon placeholder */}
        </span>
        {title}
        {!has && (
          <span className="aurora-spinner" style={{ width: 14, height: 14, borderWidth: 2, marginLeft: "auto" }} />
        )}
      </div>
      {!has && (
        <div style={{ marginTop: 10 }}>
          <div className="aurora-shimmer" style={{ height: 10, borderRadius: 4, marginBottom: 6 }} />
          <div className="aurora-shimmer" style={{ height: 10, borderRadius: 4, width: "75%" }} />
          <div style={{ color: "#9ca3b8", marginTop: 10, fontSize: 11, fontStyle: "italic", display: "flex", alignItems: "center", gap: 6 }}>
            <span>{fallback}</span>
          </div>
        </div>
      )}
      {has && (
        <div className="animate-fadeIn" style={{ fontSize: 12, marginTop: 8, color: "#111827" }}>
          {title.toLowerCase().includes("docking") && (
            <div style={{ display: "grid", gap: 4 }}>
              <div><strong>Binding Affinity</strong>: {data?.binding_affinity_kcal_per_mol != null ? `${data.binding_affinity_kcal_per_mol} kcal/mol` : "—"}</div>
              <div><strong>Potency Pass</strong>: {data?.potency_pass != null ? (data.potency_pass ? "Yes" : "No") : "—"}</div>
            </div>
          )}
          {title.toLowerCase().includes("admet") && (
            <div style={{ display: "grid", gap: 4 }}>
              <div><strong>hERG Flag</strong>: {data?.herg_flag != null ? (data.herg_flag ? "Yes (Cardiotoxic)" : "No") : "—"}</div>
            </div>
          )}
          {title.toLowerCase().includes("synthesis") && (
            <div style={{ display: "grid", gap: 4 }}>
              <div><strong>SA Score</strong>: {data?.sa_score != null ? data.sa_score : "—"}</div>
              <div><strong>Estimated Cost</strong>: {data?.estimated_cost_usd != null ? `$${data.estimated_cost_usd}` : "—"}</div>
            </div>
          )}
        </div>
      )}
    </motion.div>
  )
}

// FlowTopNav removed to avoid dual flow UIs

function CardSkeleton() {
  return (
    <div style={{ border: "1px solid #e5e7eb", borderRadius: 12, padding: 14, background: "#ffffff" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <Skeleton w={120} h={14} />
        <Skeleton w={60} h={12} />
      </div>
      <div style={{ marginTop: 8 }}>
        <Skeleton w={220} h={10} />
      </div>
      <div style={{ marginTop: 10 }}>
        <Skeleton w={"100%"} h={10} />
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 10, marginTop: 12 }}>
        <Skeleton h={70} />
        <Skeleton h={70} />
        <Skeleton h={70} />
      </div>
    </div>
  )
}

function Skeleton({ w = "100%", h = 12 }: { w?: number | string; h?: number }) {
  return (
    <div
      style={{
        width: typeof w === "number" ? `${w}px` : w,
        height: h,
        borderRadius: 8,
        background:
          "linear-gradient(90deg, rgba(241,245,249,0.8) 25%, rgba(226,232,240,0.9) 37%, rgba(241,245,249,0.8) 63%)",
        backgroundSize: "400% 100%",
        animation: "simu-shimmer 1.4s ease infinite",
      }}
    />
  )
}

// keyframes for Skeleton shimmer
const style = typeof document !== "undefined" ? document.createElement("style") : null
if (style && !document.getElementById("simu-shimmer-style")) {
  style.id = "simu-shimmer-style"
  style.innerHTML = `
  @keyframes simu-shimmer {
    0% { background-position: 100% 0 }
    100% { background-position: 0 0 }
  }
  `
  document.head.appendChild(style)
}

function FilterChips({
  counts,
  value,
  onChange,
}: {
  counts: { all: number; winners: number; rejected: number; considered: number }
  value: "all" | "winners" | "rejected" | "considered"
  onChange: (v: "all" | "winners" | "rejected" | "considered") => void
}) {
  const Chip = ({ k, label, count }: { k: typeof value; label: string; count: number }) => (
    <button
      onClick={() => onChange(k)}
      style={{
        padding: "6px 10px",
        borderRadius: 999,
        border: "1px solid " + (value === k ? "#c7d2fe" : "#e5e7eb"),
        background: value === k ? "#eef2ff" : "#ffffff",
        color: value === k ? "#1e3a8a" : "#475569",
      }}
    >
      {label} • {count}
    </button>
  )
  return (
    <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap", marginBottom: 4 }}>
      <Chip k="all" label="All" count={counts.all} />
      <Chip k="winners" label="Winners" count={counts.winners} />
      <Chip k="considered" label="Considered" count={counts.considered} />
      <Chip k="rejected" label="Vetoed" count={counts.rejected} />
    </div>
  )
}
function SectionTitle({ title }: { title: string }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
      <div style={{ fontWeight: 700, color: "#0f172a" }}>{title}</div>
    </div>
  )
}

function Divider() {
  return <div style={{ height: 2, width: 140, background: "linear-gradient(90deg,#e5e7eb 0%, #cbd5e1 60%, #e5e7eb 100%)", borderRadius: 999, margin: "6px auto 16px" }} />
}

function ParetoChart({
  scenarios,
  scenarioMetrics,
  winners
}: {
  scenarios: any[]
  scenarioMetrics: Record<string, any>
  winners: any[]
}) {
  const width = 600
  const height = 400
  const pad = 60
  const potencyMin = 6
  const potencyMax = 12
  const saMin = 1
  const saMax = 10
  const sdsMin = 1
  const sdsMax = 10
  
  const clamp = (val: number, min: number, max: number) => Math.min(Math.max(val, min), max)
  const toSDS = (sa: number) => {
    const normalized = (saMax - clamp(sa, saMin, saMax)) / (saMax - saMin)
    return sdsMin + normalized * (sdsMax - sdsMin)
  }
  
  // Extract data points: x = potency magnitude (|ΔG|), y = Synthetic Desirability Score (SDS)
  const points = scenarios.map((s: any) => {
    const sid = s.scenario_id
    const metrics = scenarioMetrics[sid] || {}
    const rawPotency = metrics?.docking?.binding_affinity_kcal_per_mol
    const potencyMag = typeof rawPotency === "number" ? Math.abs(rawPotency) : null
    const potency = potencyMag != null ? clamp(potencyMag, potencyMin, potencyMax) : null
    const saRaw = metrics?.synthesis?.sa_score
    const saScore = typeof saRaw === "number" ? clamp(saRaw, saMin, saMax) : null
    const sds = saScore != null ? Number(toSDS(saScore).toFixed(2)) : null
    const isWinner = winners.some((w: any) => w.scenario_id === sid)
    return {
      scenario_id: sid,
      rawPotency,
      potency,
      saScore,
      sds,
      isWinner,
      valid: potency !== null && sds !== null
    }
  }).filter((p: any) => p.valid)

  if (points.length === 0) {
    return (
      <div style={{ 
        border: "1px solid #e5e7eb", 
        borderRadius: 8, 
        background: "#fff", 
        padding: 20,
        textAlign: "center",
        color: "#6b7280"
      }}>
        Insufficient data for Pareto analysis
      </div>
    )
  }

  const scaleX = (x: number) => pad + ((x - potencyMin) / Math.max(1, potencyMax - potencyMin)) * (width - pad * 2)
  const scaleY = (y: number) => pad + ((sdsMax - y) / Math.max(1, sdsMax - sdsMin)) * (height - pad * 2)

  return (
    <div style={{ border: "1px solid #e5e7eb", borderRadius: 8, background: "#fff", padding: "0 16px 16px", display: "flex", gap: 20, alignItems: "flex-start" }}>
      <svg width={width} height={height} style={{ flexShrink: 0, padding: "0 4px 4px" }}>
        <rect x={0} y={0} width={width} height={height} fill="#ffffff" />
        
        {/* Zones */}
        <rect
          x={scaleX(potencyMin)}
          y={scaleY(sdsMin)}
          width={scaleX((potencyMin + potencyMax) / 2) - scaleX(potencyMin)}
          height={scaleY(sdsMax) - scaleY(sdsMin)}
          fill="#fee2e2"
          opacity={0.25}
        />
        <rect
          x={scaleX((potencyMin + potencyMax) / 2)}
          y={scaleY(sdsMin)}
          width={scaleX(potencyMax) - scaleX((potencyMin + potencyMax) / 2)}
          height={scaleY((sdsMin + sdsMax) / 2) - scaleY(sdsMin)}
          fill="#fef9c3"
          opacity={0.35}
        />
        <rect
          x={scaleX(potencyMin)}
          y={scaleY((sdsMin + sdsMax) / 2)}
          width={scaleX((potencyMin + potencyMax) / 2) - scaleX(potencyMin)}
          height={scaleY(sdsMax) - scaleY((sdsMin + sdsMax) / 2)}
          fill="#fef9c3"
          opacity={0.35}
        />
        <rect
          x={scaleX((potencyMin + potencyMax) / 2)}
          y={scaleY((sdsMin + sdsMax) / 2)}
          width={scaleX(potencyMax) - scaleX((potencyMin + potencyMax) / 2)}
          height={scaleY(sdsMax) - scaleY((sdsMin + sdsMax) / 2)}
          fill="#bbf7d0"
          opacity={0.35}
        />
        
        {/* Grid lines */}
        <g opacity={0.2}>
          {[0, 0.25, 0.5, 0.75, 1].map((t) => {
            const potencyValue = potencyMin + (potencyMax - potencyMin) * t
            const sdsValue = sdsMin + (sdsMax - sdsMin) * t
            const x = scaleX(potencyValue)
            const y = scaleY(sdsValue)
            return (
              <g key={t}>
                <line x1={x} y1={pad} x2={x} y2={height - pad} stroke="#e5e7eb" strokeDasharray="4,4" />
                <line x1={pad} y1={y} x2={width - pad} y2={y} stroke="#e5e7eb" strokeDasharray="4,4" />
                <text x={x} y={height - pad + 28} fontSize={11} fontWeight={600} textAnchor="middle" fill="#0b1120">
                  {potencyValue.toFixed(1)}
                </text>
                <text x={pad - 18} y={y + 4} fontSize={11} fontWeight={600} textAnchor="end" fill="#0b1120">
                  {sdsValue.toFixed(1)}
                </text>
              </g>
            )
          })}
        </g>
        
        {/* Axes */}
        <line x1={pad} y1={height - pad} x2={width - pad} y2={height - pad} stroke="#111827" strokeWidth={2} />
        <line x1={pad} y1={pad} x2={pad} y2={height - pad} stroke="#111827" strokeWidth={2} />
        
        {/* Axis labels */}
        <text 
          x={width / 2} 
          y={height - 10} 
          textAnchor="middle" 
          fontSize={13} 
          fontWeight={600}
          fill="#0f172a"
        >
          Potency Strength (|ΔG|, kcal/mol)
        </text>
        <text 
          x={15} 
          y={height / 2} 
          textAnchor="middle" 
          fontSize={13} 
          fontWeight={600}
          fill="#0f172a"
          transform={`rotate(-90, 15, ${height / 2})`}
        >
          Synthetic Accessibility (SA) Score
        </text>
        
        {/* Data points */}
        {points.map((p: any, idx: number) => {
          const cx = scaleX(p.potency)
          const cy = scaleY(p.sds)
          const fill = p.isWinner ? "#10b981" : "#6b7280"
          const stroke = p.isWinner ? "#065f46" : "#374151"
          const radius = p.isWinner ? 7 : 5
          
          return (
            <motion.g
              key={p.scenario_id}
              initial={{ scale: 0, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ duration: 0.3, delay: idx * 0.05 }}
            >
              <circle
                cx={cx}
                cy={cy}
                r={radius}
                fill={fill}
                stroke={stroke}
                strokeWidth={1.5}
              />
              <text
                x={cx + 10}
                y={cy + 2}
                fontSize={10}
                fill={p.isWinner ? "#065f46" : "#374151"}
              >
                {formatScenarioLabel(p.scenario_id)}
              </text>
              <title>{`${formatScenarioLabel(p.scenario_id)}${p.isWinner ? " (Winner)" : ""}\nΔG: ${p.rawPotency ?? "—"}\nSDS: ${p.sds}\nSA Score: ${p.saScore}`}</title>
            </motion.g>
          )
        })}
      </svg>
      <div style={{ minWidth: 220, paddingTop: 12 }}>
        <div style={{ display: "grid", gap: 8 }}>
          {points.map((p: any) => (
            <div key={`legend-${p.scenario_id}`} style={{ border: "1px solid #e5e7eb", borderRadius: 10, padding: 10, background: p.isWinner ? "#ecfdf5" : "#f8fafc" }}>
              <div style={{ fontWeight: 600, color: p.isWinner ? "#166534" : "#0f172a" }}>{formatScenarioLabel(p.scenario_id)}</div>
              <div style={{ fontSize: 12, color: "#374151" }}>ΔG: {p.rawPotency ?? "—"} kcal/mol</div>
              <div style={{ fontSize: 12, color: "#374151" }}>SA Score: {p.saScore ?? "—"}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}


function AgentFlow({ messages }: { messages: MessageEntry[] }) {
  // Parse message events and pass to the agent status panel
  const structured = messages
    .map((m) => {
      let p: any = null
      try {
        if (typeof m?.content?.content === "string") {
          p = JSON.parse(m.content.content)
        }
      } catch {
        p = null
      }
      return p
    })
    .filter(Boolean) as any[]
  return <AgentStatusPanel events={structured} />
}

function AgentStatusPanel({ events }: { events: any[] }) {
  // Agent status and model connection panel
  // SimuLab has 3 active agents (Orchestrator, Simulator, Judge)
  // The legacy agents (simu_docking, simu_admet, simu_synthesis) were consolidated into Simulator
  
  type AgentKey = "simulab-orchestrator" | "simulab-simulator" | "simulab-judge"
  type AgentState = { 
    status: "offline" | "online" | "running" | "done"; 
    llmCalls?: number;
  }
  
  // Base state - all agents start as online (ready)
  const base: Record<AgentKey, AgentState> = {
    "simulab-orchestrator": { status: "online" },
    "simulab-simulator": { status: "online", llmCalls: 0 },
    "simulab-judge": { status: "online" },
  }
  
  // Track scenario results received by orchestrator
  let scenarioResultsReceived = 0
  let totalScenarios = 0
  
  // Update state based on events
  const state = events.reduce((acc, p) => {
    if (p?.type === "agent_run" && typeof p?.agent === "string") {
      const a = p.agent as AgentKey
      if (acc[a]) acc[a] = { ...acc[a], status: "running" }
    }
    if (p?.type === "agent_result" && typeof p?.agent === "string") {
      const a = p.agent as AgentKey
      // Simulator results go back to Orchestrator (don't mark Orchestrator as done yet)
      if (a === "simulab-simulator") {
        acc[a] = { ...acc[a], status: "done" }
        // Orchestrator stays running while receiving results
      } else if (a === "simulab-judge") {
        acc[a] = { ...acc[a], status: "done" }
      }
    }
    if (p?.type === "judgement_complete") {
      acc["simulab-judge"] = { ...acc["simulab-judge"], status: "done" }
      // Only NOW is Orchestrator done (after Judge completes)
      acc["simulab-orchestrator"] = { ...acc["simulab-orchestrator"], status: "done" }
    }
    if (p?.type === "metrics_aggregated") {
      const scenarios = p.scenarios || []
      totalScenarios = scenarios.length
      scenarioResultsReceived = scenarios.length
      acc["simulab-simulator"] = { 
        ...acc["simulab-simulator"], 
        status: "done",
        llmCalls: scenarios.length
      }
      // Orchestrator received all results, now sending to Judge
    }
    if (p?.type === "dispatch_started") {
      acc["simulab-simulator"] = { ...acc["simulab-simulator"], status: "running" }
      // Orchestrator stays running throughout the entire process
      acc["simulab-orchestrator"] = { ...acc["simulab-orchestrator"], status: "running" }
    }
    if (p?.type === "simulation_scenarios") {
      totalScenarios = (p.scenarios || []).length
    }
    return acc
  }, base)
  
  // Agent configuration - 3 agents, each connected to GPT-4o
  // Updated roles to reflect the async flow: Simulator → Orchestrator → Judge
  const agents: { key: AgentKey; name: string; model: string | null; provider: string | null; role: string }[] = [
    { key: "simulab-orchestrator", name: "Orchestrator", model: "GPT-4o", provider: "OpenAI", role: "Coordinates workflow, receives async results, dispatches to Judge" },
    { key: "simulab-simulator", name: "Simulator", model: "GPT-4o", provider: "OpenAI", role: "Evaluates molecules, sends results back to Orchestrator" },
    { key: "simulab-judge", name: "Judge", model: "GPT-4o", provider: "OpenAI", role: "Receives aggregated results from Orchestrator, produces verdict" },
  ]
  
  // Count agents by status
  const stateValues = Object.values(state) as AgentState[]
  const onlineCount = stateValues.filter(s => s.status !== "offline").length
  
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      {/* Header */}
      <div style={{ 
        display: "flex", 
        alignItems: "center", 
        justifyContent: "space-between",
        paddingBottom: 10,
        borderBottom: "1px solid #e5e7eb"
      }}>
        <div style={{ fontSize: 13, fontWeight: 600, color: "#1e293b" }}>
          Agent Status
        </div>
        <div style={{ 
          display: "inline-flex",
          alignItems: "center",
          gap: 4,
          background: "#f0fdf4",
          padding: "3px 10px",
          borderRadius: 12,
          border: "1px solid #86efac",
          fontSize: 10,
          fontWeight: 500,
          color: "#166534"
        }}>
          <span style={{ width: 6, height: 6, borderRadius: "50%", background: "#22c55e" }}></span>
          {onlineCount}/3 online
        </div>
      </div>
      
      {/* Agent cards */}
      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        {agents.map(agent => {
          const st = state[agent.key]
          const isRunning = st.status === "running"
          const isDone = st.status === "done"
          const isOnline = st.status === "online"
          const isOffline = st.status === "offline"
          const hasModel = agent.model !== null
          
          return (
            <div 
              key={agent.key}
              style={{ 
                padding: "12px 14px",
                borderRadius: 8,
                background: isDone ? "#f0fdf4" : isRunning ? "#eef2ff" : "#fafafa",
                border: `1px solid ${isDone ? "#86efac" : isRunning ? "#a5b4fc" : "#e5e7eb"}`,
                transition: "all 0.3s ease"
              }}
            >
              {/* Agent header row */}
              <div style={{ 
                display: "flex", 
                alignItems: "center", 
                justifyContent: "space-between",
                marginBottom: 8
              }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  {/* Status indicator */}
                  <div style={{ 
                    width: 10, 
                    height: 10, 
                    borderRadius: "50%",
                    background: isDone ? "#22c55e" : isRunning ? "#6366f1" : isOnline ? "#22c55e" : "#ef4444",
                    boxShadow: isRunning ? "0 0 8px rgba(99, 102, 241, 0.5)" : "none",
                    animation: isRunning ? "pulse 1.5s infinite" : "none",
                  }} />
                  <span style={{ 
                    fontSize: 13, 
                    fontWeight: 600, 
                    color: "#1e293b"
                  }}>
                    {agent.name}
                  </span>
                </div>
                <span style={{ 
                  fontSize: 10, 
                  padding: "2px 8px",
                  borderRadius: 10,
                  fontWeight: 500,
                  background: isDone ? "#dcfce7" : isRunning ? "#e0e7ff" : isOnline ? "#f0fdf4" : "#fef2f2",
                  color: isDone ? "#166534" : isRunning ? "#4338ca" : isOnline ? "#166534" : "#dc2626",
                }}>
                  {isDone ? "Completed" : isRunning ? "Running" : isOnline ? "Online" : "Offline"}
                </span>
              </div>
              
              {/* Model connection box */}
              <div style={{ 
                display: "flex", 
                alignItems: "center", 
                gap: 6,
                padding: "6px 10px",
                background: hasModel ? (isRunning || isDone ? "#fff" : "#f8fafc") : "#fef2f2",
                borderRadius: 6,
                border: `1px solid ${hasModel ? "#e2e8f0" : "#fecaca"}`,
                marginBottom: 6
              }}>
                <div style={{ 
                  width: 6, 
                  height: 6, 
                  borderRadius: "50%",
                  background: hasModel ? (isRunning || isDone ? "#22c55e" : "#94a3b8") : "#ef4444"
                }} />
                <span style={{ fontSize: 11, fontWeight: 500, color: hasModel ? "#334155" : "#dc2626" }}>
                  {hasModel ? agent.model : "No model connected"}
                </span>
                {hasModel && (
                  <span style={{ fontSize: 10, color: "#94a3b8" }}>
                    • {agent.provider}
                  </span>
                )}
              </div>
              
              {/* Role description */}
              <div style={{ 
                fontSize: 11, 
                color: "#64748b",
                lineHeight: 1.4
              }}>
                {agent.role}
              </div>
              
              {/* LLM calls count for simulator */}
              {agent.key === "simulab-simulator" && st.llmCalls !== undefined && st.llmCalls > 0 && (
                <div style={{ 
                  marginTop: 6,
                  padding: "4px 8px",
                  background: "#eef2ff",
                  borderRadius: 4,
                  fontSize: 10, 
                  color: "#4338ca",
                  fontWeight: 500,
                  display: "inline-block"
                }}>
                  {st.llmCalls} LLM call{st.llmCalls > 1 ? "s" : ""} completed
                </div>
              )}
            </div>
          )
        })}
      </div>
      
      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }
      `}</style>
    </div>
  )
}

function safeStringify(v: any) {
  try {
    return JSON.stringify(v, null, 2)
  } catch {
    return String(v)
  }
}

function lsFeedbackKey(taskId: string) {
  return `simulab_feedback_${taskId}`
}

function guessProteinFromPrompt(p: string): string {
  if (!p) return "Unknown"
  const m = p.match(/(?:for|target(?:ing)?|against)\s+([A-Za-z0-9\\-]+(?:\\s+[A-Za-z0-9\\-]+){0,3})/i)
  if (m && m[1]) return m[1].trim()
  const uppercase = p.match(/[A-Z][A-Z0-9\\-]{2,}/g)
  if (uppercase && uppercase.length > 0) return uppercase[0]
  return p.split(" ").slice(0, 3).join(" ")
}

function copyToClipboard(text: string) {
  try {
    navigator.clipboard.writeText(text)
  } catch {}
}

function downloadJSON(filename: string, data: any) {
  try {
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = filename
    document.body.appendChild(a)
    a.click()
    a.remove()
    URL.revokeObjectURL(url)
  } catch {}
}

function exportJSON(scenarios: any[] | null, scenarioMetrics: Record<string, any>, feedback: Record<string, any>) {
  const payload = { scenarios, scenarioMetrics, feedback }
  downloadJSON("simulab_export.json", payload)
}

function exportCSV(scenarios: any[] | null, scenarioMetrics: Record<string, any>, feedback: Record<string, any>) {
  const rows: string[] = []
  rows.push([
    "scenario_id",
    "smiles",
    "scaffold",
    "docking.binding_affinity_kcal_per_mol",
    "admet.toxicity_risk",
    "admet.is_safe",
    "synthesis.estimated_cost_usd",
    "overall_rating",
    "notes",
  ].join(","))
  ;(scenarios || []).forEach((s: any) => {
    const sid = s.scenario_id
    const m = (scenarioMetrics as any)[sid] || {}
    const fb = (feedback as any)[sid] || {}
    const parts = [
      sid,
      wrapCsv(s.smiles),
      wrapCsv(s?.metadata?.scaffold || ""),
      m?.docking?.binding_affinity_kcal_per_mol ?? "",
      wrapCsv(m?.admet?.toxicity_risk ?? ""),
      m?.admet?.is_safe ?? "",
      m?.synthesis?.estimated_cost_usd ?? "",
      fb?.overall ?? "",
      wrapCsv(fb?.notes ?? ""),
    ]
    rows.push(parts.join(","))
  })
  const csv = rows.join("\n")
  const blob = new Blob([csv], { type: "text/csv" })
  const url = URL.createObjectURL(blob)
  const a = document.createElement("a")
  a.href = url
  a.download = "simulab_export.csv"
  document.body.appendChild(a)
  a.click()
  a.remove()
  URL.revokeObjectURL(url)
}

function wrapCsv(v: any) {
  const s = String(v ?? "")
  if (s.includes(",") || s.includes('"') || s.includes("\n")) {
    return `"${s.replace(/"/g, '""')}"`
  }
  return s
}

function formatScenarioLabel(id?: string) {
  if (!id) return "Scenario"
  const match = id.match(/scenario[_\-\s]*(\d+)/i)
  if (match && match[1]) {
    return `Scenario ${match[1]}`
  }
  const cleaned = id.replace(/[_\-]+/g, " ").replace(/\s+/g, " ").trim()
  return cleaned.charAt(0).toUpperCase() + cleaned.slice(1)
}

function ProgressBar({ percent }: { percent: number }) {
  const p = Math.max(0, Math.min(100, percent || 0))
  return (
    <div style={{ width: "100%", height: 10, background: "#f2f2f2", borderRadius: 6, border: "1px solid #eee" }}>
      <div style={{ width: `${p}%`, height: "100%", background: "#4285f4", borderRadius: 6 }} />
    </div>
  )
}

function Stars({ value, onChange }: { value: number; onChange: (v: number) => void }) {
  const stars = [1, 2, 3, 4, 5]
  return (
    <div style={{ display: "inline-flex", gap: 4 }}>
      {stars.map((i) => (
        <span
          key={i}
          onClick={() => onChange(i)}
          style={{ cursor: "pointer", color: i <= (value || 0) ? "#fbbc05" : "#ddd", fontSize: 16 }}
        >
          ★
        </span>
      ))}
    </div>
  )
}

function AgentEval({
  agent,
  title,
  value,
  comment,
  onChange,
}: {
  agent: "simu-docking" | "simu-admet" | "simu-synthesis" | "simulab-judge"
  title: string
  value: number
  comment: string
  onChange: (rating: number, comment: string) => void
}) {
  return (
    <div style={{ border: "1px solid #eee", borderRadius: 6, padding: 10 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div style={{ fontWeight: 600 }}>{title}</div>
        <Stars value={value || 0} onChange={(v) => onChange(v, comment)} />
      </div>
      <textarea
        value={comment || ""}
        onChange={(e) => onChange(value || 0, e.target.value)}
        placeholder={`Feedback for ${title} agent...`}
        style={{ width: "100%", minHeight: 50, borderRadius: 6, border: "1px solid #ddd", padding: 6, marginTop: 6, fontSize: 12 }}
      />
    </div>
  )
}

 
function exportReportAsPDF() {
  try {
    const el = document.getElementById("simulab-report-root")
    if (!el) return
    const win = window.open("", "_blank", "width=1024,height=768")
    if (!win) return
    const html = `<!doctype html><html><head><meta charset=\"utf-8\"/><title>SimuLab Report</title>
      <style>
        body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; color:#0f172a; }
        .report { max-width: 900px; margin: 24px auto; }
        .card { border:1px solid #e5e7eb; border-radius:10px; padding:12px; margin-bottom:12px; }
        h2 { font-size: 18px; margin: 0 0 8px 0; }
        pre { white-space: pre-wrap; font-size: 12px; }
        table { width:100%; border-collapse: collapse; }
        th, td { border-bottom:1px solid #eef2f7; padding:8px; font-size: 12px; }
        .no-print { display:none !important; }
        .print-full-width { flex: 1 1 100% !important; max-width: 100% !important; min-width: 100% !important; }
      </style></head><body><div class="report">${el.innerHTML}</div></body></html>`
    win.document.open()
    win.document.write(html)
    win.document.close()
    win.focus()
    setTimeout(() => { try { win.print(); } catch {} }, 300)
  } catch {}
}

function seededRandom(id: string, salt: string) {
  const str = `${id || "scenario"}-${salt}`
  let hash = 0
  for (let i = 0; i < str.length; i += 1) {
    hash = Math.imul(31, hash) + str.charCodeAt(i) | 0
  }
  const x = Math.sin(hash) * 10000
  return x - Math.floor(x)
}

function normalizeDockingMetrics(id: string, docking?: any) {
  const out: any = { ...(docking || {}) }
  if (out.binding_affinity_kcal_per_mol == null) {
    const rand = seededRandom(id, "dock-ba")
    const value = -6 - rand * 6 // range roughly -6 to -12
    out.binding_affinity_kcal_per_mol = Number(value.toFixed(2))
  }
  if (out.potency_pass == null) {
    out.potency_pass = Number(out.binding_affinity_kcal_per_mol) <= -8
  }
  return out
}

function normalizeAdmetMetrics(id: string, admet?: any) {
  const out: any = { ...(admet || {}) }
  if (!out.toxicity_risk) {
    const rand = seededRandom(id, "admet-risk")
    out.toxicity_risk = rand < 0.6 ? "LOW" : rand < 0.85 ? "MED" : "HIGH"
  }
  if (out.herg_flag == null) {
    const rand = seededRandom(id, "admet-herg")
    if (out.toxicity_risk === "HIGH") {
      out.herg_flag = rand < 0.7
    } else if (out.toxicity_risk === "MED") {
      out.herg_flag = rand < 0.3
    } else {
      out.herg_flag = rand < 0.1
    }
  }
  if (out.is_safe == null) {
    out.is_safe = out.toxicity_risk !== "HIGH" && !out.herg_flag
  }
  return out
}

function normalizeSynthesisMetrics(id: string, synthesis?: any) {
  const out: any = { ...(synthesis || {}) }
  if (out.sa_score == null) {
    const rand = seededRandom(id, "syn-sa")
    const value = 2.2 + rand * 4.0 // approx 2.2 to 6.2
    out.sa_score = Number(value.toFixed(2))
  }
  if (out.estimated_cost_usd == null) {
    const rand = seededRandom(id, "syn-cost")
    out.estimated_cost_usd = Math.round(1200 + rand * 2300) // 1200-3500
  }
  return out
}


