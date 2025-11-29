"use client"

import { useMemo, useState, useEffect } from "react"
import { motion, AnimatePresence } from "framer-motion"

type AgentKey = "Orchestrator" | "Simulator" | "Docking" | "ADMET" | "Synthesis" | "Judge"
type AgentState = { name: AgentKey; status: "idle" | "running" | "done"; summary?: string }

type MoleculeCard = {
  id: string
  smiles: string
  bindingAffinity: number
  toxicityRisk: "LOW" | "MED" | "HIGH"
  estimatedCost: number
  status: "Winner" | "Vetoed" | "Considered"
}

const sidebarAgentsInitial: AgentState[] = [
  { name: "Orchestrator", status: "running", summary: "Defining scenarios" },
  { name: "Simulator", status: "idle", summary: "Generating SMILES" },
  { name: "Docking", status: "idle", summary: "Potency checks" },
  { name: "ADMET", status: "idle", summary: "Safety checks" },
  { name: "Synthesis", status: "idle", summary: "Cost checks" },
  { name: "Judge", status: "idle", summary: "Pareto decision" },
]

const mockCards: MoleculeCard[] = [
  { id: "scenario_1", smiles: "C1=NC2=CC(N)=NC(N)=C2N1", bindingAffinity: -11.2, toxicityRisk: "HIGH", estimatedCost: 3200, status: "Vetoed" },
  { id: "scenario_2", smiles: "C1=NN2C=CC(N)=NC2=C1", bindingAffinity: -9.1, toxicityRisk: "LOW", estimatedCost: 1400, status: "Winner" },
  { id: "scenario_3", smiles: "CC1=CC=CC=C1NCC", bindingAffinity: -8.7, toxicityRisk: "MED", estimatedCost: 2200, status: "Considered" },
]

const chartData = [
  { x: 0, y: -7.5 },
  { x: 1, y: -8.2 },
  { x: 2, y: -9.1 },
  { x: 3, y: -8.6 },
  { x: 4, y: -9.3 },
  { x: 5, y: -10.1 },
]

export default function DashboardPage() {
  const [agents, setAgents] = useState<AgentState[]>(sidebarAgentsInitial)
  const [cards, setCards] = useState<MoleculeCard[]>(mockCards)
  const [activity, setActivity] = useState<string[]>([
    "Simulator generated 3 SMILES",
    "Docking completed potency checks",
    "ADMET flagged hERG risk for scenario_1",
    "Synthesis estimated costs",
    "Judge selected scenario_2",
  ])
  const [reason, setReason] = useState<string>("")
  const [loadingReason, setLoadingReason] = useState(false)
  const [showConstraints, setShowConstraints] = useState(false)
  const [showNewScenario, setShowNewScenario] = useState(false)
  const [showAgentTune, setShowAgentTune] = useState<null | AgentKey>(null)
  const [newSmiles, setNewSmiles] = useState("")
  const [constraints, setConstraints] = useState({ hergMax: 5, minAffinity: -9.0, maxCost: 3000 })
  const [selectedScenario, setSelectedScenario] = useState<MoleculeCard | null>(null)

  // Simulate agent progress
  useEffect(() => {
    const timeouts: NodeJS.Timeout[] = []
    const seq: Array<Partial<AgentState> & { idx: number }> = [
      { idx: 1, status: "running", summary: "Generating SMILES" },
      { idx: 2, status: "running", summary: "Scoring potency (ΔG)" },
      { idx: 3, status: "running", summary: "Screening toxicity (hERG)" },
      { idx: 4, status: "running", summary: "Estimating cost/SA" },
      { idx: 5, status: "running", summary: "Multi-objective selection" },
      { idx: 1, status: "done", summary: "3 SMILES generated" },
      { idx: 2, status: "done", summary: "Potency scored" },
      { idx: 3, status: "done", summary: "Safety screened" },
      { idx: 4, status: "done", summary: "Costs estimated" },
      { idx: 5, status: "done", summary: "Winner chosen" },
      { idx: 0, status: "done", summary: "Workflow complete" },
    ]
    seq.forEach((s, i) => {
      timeouts.push(
        setTimeout(() => {
          setAgents((prev) => {
            const next = [...prev]
            const idx = s.idx
            next[idx] = { ...next[idx], status: s.status as any, summary: s.summary }
            return next
          })
        }, 700 * (i + 1))
      )
    })
    return () => timeouts.forEach(clearTimeout)
  }, [])

  async function generateReason() {
    try {
      setLoadingReason(true)
      const winners = cards.filter((c) => c.status === "Winner").map((c) => ({ scenario_id: c.id, smiles: c.smiles }))
      const rejected = cards.filter((c) => c.status === "Vetoed").map((c) => ({ scenario_id: c.id, smiles: c.smiles }))
      const resp = await fetch("/api/simulab/reason", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ winners, rejected, scenarios: cards, context: { protein_target: "BCR-ABL Kinase" } }),
      })
      const data = await resp.json()
      setReason(data?.reason || "No rationale available.")
    } catch {
      setReason("No rationale available.")
    } finally {
      setLoadingReason(false)
    }
  }

  const container = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: { staggerChildren: 0.1, delayChildren: 0.1 },
    },
  }
  const item = {
    hidden: { opacity: 0, y: 8 },
    show: { opacity: 1, y: 0, transition: { duration: 0.3 } },
  }
  const feedItem = {
    hidden: { opacity: 0, x: 16 },
    show: { opacity: 1, x: 0, transition: { duration: 0.3 } },
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-slate-100">
      {/* Fixed top header */}
      <header className="fixed top-0 left-0 right-0 h-14 border-b bg-gradient-to-r from-indigo-50 to-blue-50 backdrop-blur z-30 flex items-center justify-between px-4">
        <div className="font-semibold text-slate-800">SimuLab Dashboard</div>
        <div className="flex gap-2">
          <button
            onClick={() => setShowConstraints(true)}
            className="px-3 py-1.5 rounded-md border bg-white hover:bg-gray-50 text-sm"
          >
            Adjust Constraints
          </button>
          <button
            onClick={() => setShowNewScenario(true)}
            className="px-3 py-1.5 rounded-md border bg-white hover:bg-gray-50 text-sm"
          >
            Add Scenario
          </button>
          <button
            onClick={generateReason}
            className="px-3 py-1.5 rounded-md bg-indigo-600 text-white text-sm shadow-sm hover:bg-indigo-700 transition-colors"
          >
            {loadingReason ? "Generating..." : "Generate Summary"}
          </button>
        </div>
      </header>

      {/* Fixed left sidebar */}
      <aside className="fixed top-14 left-0 bottom-0 w-64 border-r bg-white/90 backdrop-blur z-20">
        <div className="px-4 py-3 font-semibold text-slate-800 bg-gradient-to-r from-slate-50 to-white border-b">Agentic Flow</div>
        <div className="px-3 pb-6 overflow-auto h-[calc(100%-3rem)]">
          <div className="space-y-2">
            {agents.map((a, idx) => (
              <motion.div
                key={a.name}
                className="border rounded-md p-3 bg-white shadow-sm"
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                whileHover={{ scale: 1.01 }}
                transition={{ duration: 0.25, delay: 0.05 * idx }}
              >
                <div className="flex items-center justify-between">
                  <div className="text-sm font-medium text-slate-800">{a.name}</div>
                  <span
                    className={`text-[10px] px-2 py-0.5 rounded-full ${
                      a.status === "done"
                        ? "bg-green-100 text-green-700"
                        : a.status === "running"
                        ? "bg-blue-100 text-blue-700"
                        : "bg-gray-100 text-gray-600"
                    }`}
                  >
                    {a.status.toUpperCase()}
                  </span>
                </div>
                {a.summary && <div className="text-xs text-gray-500 mt-1">{a.summary}</div>}
                <div className="flex gap-2 mt-2">
                  <button onClick={() => setShowAgentTune(a.name)} className="text-xs px-2 py-1 rounded border hover:bg-gray-50">
                    Tune
                  </button>
                  <button onClick={() => setActivity((prev) => [`Paused ${a.name}`, ...prev])} className="text-xs px-2 py-1 rounded border hover:bg-gray-50">
                    Pause
                  </button>
                  <button onClick={() => setActivity((prev) => [`Resumed ${a.name}`, ...prev])} className="text-xs px-2 py-1 rounded border hover:bg-gray-50">
                    Resume
                  </button>
                </div>
                {idx < agents.length - 1 && (
                  <div className="flex justify-center pt-1">
                    <div className="w-0.5 h-3 bg-gray-200 rounded-full" />
                  </div>
                )}
              </motion.div>
            ))}
          </div>
        </div>
      </aside>

      {/* Main content */}
      <main className="pt-14 ml-64 px-6">
        {/* Metric Cards */}
        <section className="mt-6">
          <motion.div variants={container} initial="hidden" animate="show" className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
            {cards.map((c) => (
              <motion.div
                key={c.id}
                variants={item}
                className="border rounded-xl bg-white p-4 shadow-sm"
                whileHover={{ y: -3, boxShadow: "0 10px 20px rgba(0,0,0,0.08)" }}
                transition={{ type: "spring", stiffness: 200, damping: 18 }}
              >
                <div className="flex items-center justify-between mb-2">
                  <div className="font-semibold text-slate-800">{c.id}</div>
                  <span
                    className={`text-xs px-2 py-0.5 rounded-full ${
                      c.status === "Winner"
                        ? "bg-green-100 text-green-700"
                        : c.status === "Vetoed"
                        ? "bg-red-100 text-red-700"
                        : "bg-gray-100 text-gray-700"
                    }`}
                  >
                    {c.status}
                  </span>
                </div>
                <div className="text-xs text-gray-500 mb-2">SMILES</div>
                <code className="block text-sm break-words mb-3">{c.smiles}</code>
                <div className="grid grid-cols-3 gap-3 text-sm">
                  <div>
                    <div className="text-xs text-gray-500">Binding Affinity</div>
                    <div className="font-medium text-slate-800">{c.bindingAffinity} kcal/mol</div>
                  </div>
                  <div>
                    <div className="text-xs text-gray-500">Toxicity</div>
                    <div
                      className={`font-medium ${
                        c.toxicityRisk === "LOW" ? "text-green-600" : c.toxicityRisk === "MED" ? "text-yellow-600" : "text-red-600"
                      }`}
                    >
                      {c.toxicityRisk}
                    </div>
                  </div>
                  <div>
                    <div className="text-xs text-gray-500">Est. Cost</div>
                    <div className="font-medium text-slate-800">${c.estimatedCost}</div>
                  </div>
                </div>
                <div className="flex gap-2 mt-3">
                  <button
                    className="text-xs px-2 py-1 rounded border hover:bg-gray-50"
                    onClick={() => {
                      setSelectedScenario(c)
                      setActivity((prev) => [`What-if: re-dock ${c.id}`, ...prev])
                    }}
                  >
                    What-if: Re-dock
                  </button>
                  <button
                    className="text-xs px-2 py-1 rounded border hover:bg-gray-50"
                    onClick={() => setActivity((prev) => [`Marked unsafe ${c.id}`, ...prev])}
                  >
                    Mark Unsafe
                  </button>
                  <button
                    className="text-xs px-2 py-1 rounded border hover:bg-gray-50"
                    onClick={() => setActivity((prev) => [`Promoted winner ${c.id}`, ...prev])}
                  >
                    Promote Winner
                  </button>
                </div>
              </motion.div>
            ))}
          </motion.div>
        </section>

        {/* Line Chart */}
        <section className="mt-8">
          <div className="border rounded-xl bg-white p-4 shadow-sm">
            <div className="font-semibold mb-2 text-slate-800">Potency Trend (ΔG)</div>
            <Chart />
          </div>
        </section>

        {/* Recent Activity */}
        <section className="mt-8 mb-10">
          <div className="border rounded-xl bg-white p-4 shadow-sm">
            <div className="font-semibold mb-3 text-slate-800">Recent Activity</div>
            <motion.ul
              initial="hidden"
              animate="show"
              variants={{ hidden: {}, show: { transition: { staggerChildren: 0.1 } } }}
              className="space-y-2"
            >
              {activity.map((a, i) => (
                <motion.li
                  key={i}
                  variants={feedItem}
                  className="border rounded-md px-3 py-2 bg-gray-50 hover:bg-gray-100 transition-colors"
                >
                  {a}
                </motion.li>
              ))}
            </motion.ul>
          </div>
        </section>

        {/* LLM Summary */}
        {reason && (
          <section className="mt-8 mb-12">
            <div className="border rounded-xl bg-white p-4 shadow-sm">
              <div className="font-semibold mb-2 text-slate-800">LLM Summary</div>
              <pre className="whitespace-pre-wrap text-sm">{reason}</pre>
            </div>
          </section>
        )}
      </main>

      {/* Modals */}
      <AnimatePresence>
        {showConstraints && (
          <Modal onClose={() => setShowConstraints(false)} title="Adjust Constraints">
            <div className="grid grid-cols-3 gap-3">
              <Field label="hERG max (%)">
                <input
                  className="w-full border rounded px-2 py-1"
                  type="number"
                  value={constraints.hergMax}
                  onChange={(e) => setConstraints((p) => ({ ...p, hergMax: parseInt(e.target.value || "0", 10) }))}
                />
              </Field>
              <Field label="Min ΔG (kcal/mol)">
                <input
                  className="w-full border rounded px-2 py-1"
                  type="number"
                  step="0.1"
                  value={constraints.minAffinity}
                  onChange={(e) => setConstraints((p) => ({ ...p, minAffinity: parseFloat(e.target.value || "0") }))}
                />
              </Field>
              <Field label="Max Cost ($)">
                <input
                  className="w-full border rounded px-2 py-1"
                  type="number"
                  value={constraints.maxCost}
                  onChange={(e) => setConstraints((p) => ({ ...p, maxCost: parseInt(e.target.value || "0", 10) }))}
                />
              </Field>
            </div>
            <div className="mt-4 flex justify-end">
              <button className="px-3 py-1.5 rounded bg-indigo-600 text-white" onClick={() => setShowConstraints(false)}>
                Save
              </button>
            </div>
          </Modal>
        )}
        {showNewScenario && (
          <Modal onClose={() => setShowNewScenario(false)} title="Add Scenario">
            <Field label="SMILES">
              <textarea className="w-full border rounded px-2 py-1" value={newSmiles} onChange={(e) => setNewSmiles(e.target.value)} />
            </Field>
            <div className="mt-4 flex justify-end">
              <button
                className="px-3 py-1.5 rounded bg-indigo-600 text-white"
                onClick={() => {
                  if (!newSmiles.trim()) return
                  const id = `scenario_${cards.length + 1}`
                  setCards((prev) => [
                    ...prev,
                    { id, smiles: newSmiles.trim(), bindingAffinity: -8.0, toxicityRisk: "MED", estimatedCost: 2100, status: "Considered" },
                  ])
                  setActivity((prev) => [`Added scenario ${id}`, ...prev])
                  setNewSmiles("")
                  setShowNewScenario(false)
                }}
              >
                Add
              </button>
            </div>
          </Modal>
        )}
        {showAgentTune && (
          <Modal onClose={() => setShowAgentTune(null)} title={`Tune ${showAgentTune}`}>
            <Field label="Parameter A">
              <input className="w-full border rounded px-2 py-1" placeholder="e.g., search_depth=3" />
            </Field>
            <Field label="Parameter B">
              <input className="w-full border rounded px-2 py-1" placeholder="e.g., docking_grid=auto" />
            </Field>
            <div className="mt-4 flex justify-end">
              <button
                className="px-3 py-1.5 rounded bg-indigo-600 text-white"
                onClick={() => {
                  setActivity((prev) => [`Tuned ${showAgentTune}`, ...prev])
                  setShowAgentTune(null)
                }}
              >
                Apply
              </button>
            </div>
          </Modal>
        )}
      </AnimatePresence>
    </div>
  )
}

function Chart() {
  const width = 640
  const height = 200
  const padding = 24
  const xScale = (x: number) => padding + (x / (chartData.length - 1)) * (width - padding * 2)
  const ys = chartData.map((d) => d.y)
  const yMin = Math.min(...ys)
  const yMax = Math.max(...ys)
  const yScale = (y: number) => {
    const t = (y - yMin) / (yMax - yMin || 1)
    return height - padding - t * (height - padding * 2)
  }
  const path = chartData.map((d, i) => `${i === 0 ? "M" : "L"} ${xScale(d.x)},${yScale(d.y)}`).join(" ")

  return (
    <motion.svg
      viewBox={`0 0 ${width} ${height}`}
      className="w-full h-48"
      whileHover={{ scale: 1.01 }}
      transition={{ type: "spring", stiffness: 120, damping: 14 }}
    >
      <defs>
        <linearGradient id="grad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#60a5fa" stopOpacity={0.4} />
          <stop offset="100%" stopColor="#60a5fa" stopOpacity={0} />
        </linearGradient>
      </defs>
      <path d={path} stroke="#3b82f6" strokeWidth={2} fill="none" />
      {/* Area fill */}
      <path d={`${path} L ${xScale(chartData.at(-1)!.x)},${height - padding} L ${xScale(chartData[0].x)},${height - padding} Z`} fill="url(#grad)" />
      {/* Points */}
      {chartData.map((d, i) => (
        <motion.circle
          key={i}
          cx={xScale(d.x)}
          cy={yScale(d.y)}
          r={4}
          fill="#3b82f6"
          whileHover={{ scale: 1.3 }}
          transition={{ type: "spring", stiffness: 260, damping: 14 }}
        />
      ))}
    </motion.svg>
  )
}

function Modal({ onClose, title, children }: { onClose: () => void; title: string; children: React.ReactNode }) {
  return (
    <motion.div
      className="fixed inset-0 z-50 flex items-center justify-center"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
    >
      <div className="absolute inset-0 bg-black/30" onClick={onClose} />
      <motion.div
        className="relative bg-white rounded-xl shadow-xl w-full max-w-lg mx-4 p-4"
        initial={{ y: 20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        exit={{ y: 20, opacity: 0 }}
        transition={{ type: "spring", stiffness: 220, damping: 20 }}
      >
        <div className="text-lg font-semibold text-slate-800">{title}</div>
        <div className="mt-3">{children}</div>
      </motion.div>
    </motion.div>
  )
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="text-sm">
      <div className="text-xs text-gray-500 mb-1">{label}</div>
      {children}
    </label>
  )
}


