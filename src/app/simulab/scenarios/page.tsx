'use client';

import React, { useEffect, useMemo, useState, Suspense } from 'react';
import { useSearchParams } from 'next/navigation';
import { motion } from 'framer-motion';

type MessageEntry = {
  id: string;
  content: { author: string; content: string; format?: string };
  created_at?: string;
};

const fadeInUp = { initial: { opacity: 0, y: 8 }, animate: { opacity: 1, y: 0 } };
const fadeInRight = { initial: { opacity: 0, x: 10 }, animate: { opacity: 1, x: 0 } };
const stagger = { animate: { transition: { staggerChildren: 0.07 } } };

function ScenariosPageContent() {
  const search = useSearchParams();
  const qTask = search.get('task_id') || (typeof window !== 'undefined' ? localStorage.getItem('simulab_current_task_id') || '' : '');
  const [taskId, setTaskId] = useState<string>(qTask || '');
  const [messages, setMessages] = useState<MessageEntry[]>([]);
  const [events, setEvents] = useState<any[]>([]);
  const [extraEvents, setExtraEvents] = useState<any[]>([]);
  const [scenarios, setScenarios] = useState<any[] | null>(null);
  const [winners, setWinners] = useState<any[] | null>(null);
  const [rejected, setRejected] = useState<any[] | null>(null);
  const [scenarioMetrics, setScenarioMetrics] = useState<Record<string, { docking?: any; admet?: any; synthesis?: any }>>({});
  const [metricsAggregated, setMetricsAggregated] = useState<boolean>(false);

  useEffect(() => {
    if (!taskId) return;
    let mounted = true;
    const iv = setInterval(async () => {
      try {
        const res = await fetch(`/api/simulab/messages?task_id=${encodeURIComponent(taskId)}`);
        const data = await res.json();
        if (mounted && Array.isArray(data)) {
          setMessages(data);
          const evts = parseMessagesToEvents(data);
          const combined = [...evts, ...extraEvents];
          setEvents(evts);
          deriveStateFromEvents(combined);
        }
      } catch {}
    }, 2000);
    return () => {
      mounted = false;
      clearInterval(iv);
    };
  }, [taskId, extraEvents]);

  // publish events for left sidebar
  useEffect(() => {
    try {
      (window as any).__simulabAgentEvents = [...events, ...extraEvents];
    } catch {}
  }, [events, extraEvents]);

  function parseMessagesToEvents(list: MessageEntry[]): any[] {
    const out: any[] = [];
    for (const m of list) {
      const raw = m?.content?.content;
      if (typeof raw !== 'string') continue;
      try {
        const j = JSON.parse(raw);
        if (j && typeof j === 'object' && j.type) out.push(j);
      } catch {}
    }
    return out;
  }

  function deriveStateFromEvents(combined: any[]) {
    let _scenarios: any[] | null = null;
    let _winners: any[] | null = null;
    let _rejected: any[] | null = null;
    let _metricsAgg = false;
    const _scenarioMetrics: Record<string, { docking?: any; admet?: any; synthesis?: any }> = {};
    for (const e of combined) {
      if (!e || typeof e !== 'object') continue;
      if (e.type === 'simulation_scenarios' && Array.isArray(e.scenarios)) _scenarios = e.scenarios;
      if (e.type === 'metrics_aggregated') {
        _metricsAgg = true;
        if (Array.isArray(e.scenarios) && !_scenarios) _scenarios = e.scenarios;
      }
      if (e.type === 'agent_result' && e.scenario_id && e.agent && e.output) {
        const sid = String(e.scenario_id);
        _scenarioMetrics[sid] = _scenarioMetrics[sid] || {};
        if (e.agent === 'simu-docking') _scenarioMetrics[sid].docking = e.output;
        if (e.agent === 'simu-admet') _scenarioMetrics[sid].admet = e.output;
        if (e.agent === 'simu-synthesis') _scenarioMetrics[sid].synthesis = e.output;
      }
      if (e.type === 'judgement_complete') {
        _winners = e.winners || [];
        _rejected = e.rejected || [];
      }
    }
    setScenarios(_scenarios);
    setScenarioMetrics(_scenarioMetrics);
    setWinners(_winners);
    setRejected(_rejected);
    setMetricsAggregated(_metricsAgg);
  }

  function toxScore(risk?: string) {
    const r = String(risk || '').toUpperCase();
    if (r === 'LOW') return 0;
    if (r === 'MED') return 1;
    if (r === 'HIGH') return 2;
    return 3;
  }
  type SortKey = 'best' | 'ba' | 'tox' | 'cost';
  const [sortKey, setSortKey] = useState<SortKey>('best');
  const [asc, setAsc] = useState<boolean>(true);
  const rows = useMemo(() => {
    const arr = (scenarios || []).map((s: any) => {
      const ba = s?.docking?.binding_affinity_kcal_per_mol ?? scenarioMetrics[s.scenario_id]?.docking?.binding_affinity_kcal_per_mol;
      const tox = s?.admet?.toxicity_risk ?? scenarioMetrics[s.scenario_id]?.admet?.toxicity_risk;
      const cost = s?.synthesis?.estimated_cost_usd ?? scenarioMetrics[s.scenario_id]?.synthesis?.estimated_cost_usd;
      return { ...s, _ba: ba, _tox: tox, _toxNum: toxScore(tox), _cost: cost };
    });
    const cmp = (a: any, b: any) => {
      let res = 0;
      if (sortKey === 'best') {
        res = (a._ba ?? 1e9) - (b._ba ?? 1e9);
        if (res === 0) res = (a._toxNum ?? 9) - (b._toxNum ?? 9);
        if (res === 0) res = (a._cost ?? 1e12) - (b._cost ?? 1e12);
      } else if (sortKey === 'ba') {
        res = (a._ba ?? 1e9) - (b._ba ?? 1e9);
      } else if (sortKey === 'tox') {
        res = (a._toxNum ?? 9) - (b._toxNum ?? 9);
      } else if (sortKey === 'cost') {
        res = (a._cost ?? 1e12) - (b._cost ?? 1e12);
      }
      return asc ? res : -res;
    };
    return arr.sort(cmp);
  }, [scenarios, scenarioMetrics, sortKey, asc]);

  return (
    <div className="w-full">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold">Scenarios</h1>
          <p className="text-sm text-slate-600">Parallel simulation per scenario with live agent outputs.</p>
        </div>
        <div className="text-sm text-slate-500">{taskId ? `Task: ${taskId}` : 'No active task'}</div>
      </div>

      <div className="mt-6">
        <h2 className="text-lg font-semibold">Scenario Scorecards</h2>
        {!scenarios && <div className="text-slate-600 mt-2">Waiting for simulation scenarios...</div>}
        {scenarios && (
          <motion.div className="grid gap-3" {...stagger}>
            {scenarios.map((s: any, idx: number) => {
              const sid = s.scenario_id;
              const metrics = (scenarioMetrics as any)[sid] || {};
              const isWinner = (winners || []).some((w: any) => w.scenario_id === sid);
              const isRejected = (rejected || []).some((r: any) => r.scenario_id === sid);
              const status = isWinner ? 'Winner' : isRejected ? 'Vetoed' : 'Considered';
              const statusColor = isWinner ? '#34a853' : isRejected ? '#ea4335' : '#999';
              const agentsDone = ['docking', 'admet', 'synthesis'].reduce((acc, k) => (metrics[k] ? acc + 1 : acc), 0);
              const progressPct = Math.round((agentsDone / 3) * 100);
              return (
                <motion.div key={sid} {...fadeInUp} transition={{ duration: 0.2, delay: idx * 0.04 }} className="border border-gray-200 rounded-lg p-3 bg-white">
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="font-semibold">{sid}</div>
                      <div className="text-xs text-slate-600">{s.metadata?.scaffold || ''}</div>
                    </div>
                    <div className="font-semibold" style={{ color: statusColor }}>
                      {status}
                    </div>
                  </div>
                  <div className="mt-2">
                    <div className="text-xs text-slate-600">SMILES</div>
                    <code className="text-sm">{s.smiles}</code>
                  </div>
                  <div className="mt-3">
                    <div className="text-xs text-slate-600 mb-1">Agent Progress</div>
                    <div className="w-full h-[10px] bg-gray-100 rounded-md border border-gray-200">
                      <div className="h-full bg-blue-500 rounded-md" style={{ width: `${progressPct}%` }} />
                    </div>
                  </div>
                  <motion.div className="grid grid-cols-1 sm:grid-cols-3 gap-2 mt-3" {...stagger}>
                    <MetricCard title="Docking (Potency)" data={metrics.docking} fallback="Awaiting result..." />
                    <MetricCard title="ADMET (Safety)" data={metrics.admet} fallback="Awaiting result..." />
                    <MetricCard title="Synthesis (Cost)" data={metrics.synthesis} fallback="Awaiting result..." />
                  </motion.div>
                </motion.div>
              );
            })}
          </motion.div>
        )}
      </div>

      {scenarios && (
        <div className="mt-6">
          <h2 className="text-lg font-semibold">Ranking</h2>
          <div className="flex items-center gap-2 my-2">
            <span className="text-xs text-slate-600">Sort by:</span>
            <button onClick={() => setSortKey('best')} className={`px-2 py-1 border rounded ${sortKey === 'best' ? 'bg-indigo-50 border-indigo-200' : 'bg-white border-gray-200'}`}>
              Best fit
            </button>
            <button onClick={() => setSortKey('ba')} className={`px-2 py-1 border rounded ${sortKey === 'ba' ? 'bg-indigo-50 border-indigo-200' : 'bg-white border-gray-200'}`}>
              Potency (ΔG)
            </button>
            <button onClick={() => setSortKey('tox')} className={`px-2 py-1 border rounded ${sortKey === 'tox' ? 'bg-indigo-50 border-indigo-200' : 'bg-white border-gray-200'}`}>
              Toxicity
            </button>
            <button onClick={() => setSortKey('cost')} className={`px-2 py-1 border rounded ${sortKey === 'cost' ? 'bg-indigo-50 border-indigo-200' : 'bg-white border-gray-200'}`}>
              Cost
            </button>
            <button onClick={() => setAsc((p) => !p)} className="px-2 py-1 border rounded bg-white border-gray-200">
              {asc ? 'Asc' : 'Desc'}
            </button>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full border-collapse">
              <thead>
                <tr>
                  <th className="text-left border-b border-gray-300 p-2">Scenario</th>
                  <th className="text-left border-b border-gray-300 p-2">SMILES</th>
                  <th className="text-right border-b border-gray-300 p-2">ΔG (kcal/mol)</th>
                  <th className="text-center border-b border-gray-300 p-2">Risk</th>
                  <th className="text-right border-b border-gray-300 p-2">Cost ($)</th>
                  <th className="text-center border-b border-gray-300 p-2">Winner</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((s: any) => {
                  const isWinner = (winners || []).some((w: any) => w.scenario_id === s.scenario_id);
                  return (
                    <tr key={s.scenario_id} className={isWinner ? 'bg-green-50' : 'bg-white'}>
                      <td className="p-2 border-b border-gray-100">{s.scenario_id}</td>
                      <td className="p-2 border-b border-gray-100">
                        <code>{s.smiles}</code>
                      </td>
                      <td className="p-2 border-b border-gray-100 text-right">{s._ba ?? ''}</td>
                      <td className="p-2 border-b border-gray-100 text-center">{s._tox ?? ''}</td>
                      <td className="p-2 border-b border-gray-100 text-right">{s._cost ?? ''}</td>
                      <td className="p-2 border-b border-gray-100 text-center">{isWinner ? '🏆' : ''}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

export default function ScenariosPage() {
  return (
    <Suspense fallback={<div className="p-4 text-slate-500">Loading scenarios...</div>}>
      <ScenariosPageContent />
    </Suspense>
  );
}

function MetricCard({ title, data, fallback }: { title: string; data?: any; fallback: string }) {
  const has = data && Object.keys(data).length > 0;
  return (
    <motion.div
      whileHover={{ scale: 1.01 }}
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2 }}
      className="border border-gray-200 rounded-md p-3 bg-white"
    >
      <div className="font-semibold">{title}</div>
      {!has && <div className="text-slate-500 mt-2 text-sm">{fallback}</div>}
      {has && <pre className="whitespace-pre-wrap text-xs mt-2 bg-gray-50 p-2 rounded">{safeStringify(data)}</pre>}
    </motion.div>
  );
}

function safeStringify(v: any) {
  try {
    return JSON.stringify(v, null, 2);
  } catch {
    return String(v);
  }
}


