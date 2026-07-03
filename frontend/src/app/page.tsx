"use client";

import React, { useState, useEffect, useCallback } from "react";
import {
  LayoutDashboard, Database, TrendingUp, Sliders, Cpu,
  Layers, ShieldAlert, MessageSquare, FileText,
  RefreshCw, CheckCircle2, AlertTriangle, ArrowUpRight,
  ArrowDownRight, Sparkles, Zap, Download, DollarSign, BarChart2, PieChart as PieIcon, Send,
  Maximize2, X
} from "lucide-react";
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  BarChart, Bar, Legend, Cell, PieChart as RePieChart, Pie
} from "recharts";

// ─── API base URL ────────────────────────────────────────────────────────────
const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

// ─── Types (mirrors backend/src/main.py response shapes exactly) ────────────

interface StatusState {
  status: string;
  timestamp: string;
  active_llm_provider: string;
  data_quality_score: number;
  risk_classification: string;
  has_critical_warnings: boolean;
}

interface TrajectoryPoint {
  day: string;
  p10: number;
  p50: number;
  p90: number;
}

interface ChannelShare {
  name: string;
  value: number;
  color: string;
}

interface CriticalWarning {
  message: string;
  penalty: number;
}

interface OverviewState {
  total_historical_spend: number;
  total_historical_revenue: number;
  overall_historical_roas: number;
  data_quality_score: number;
  risk_classification: string;
  risk_badge_color: string;
  forecast_90d_p50_revenue: number;
  forecast_90d_p50_roas: number;
  executive_summary: string;
  daily_trajectory: TrajectoryPoint[];
  channel_shares: ChannelShare[];
  critical_warnings: CriticalWarning[];
  trajectory_method: string;
}

interface ValidationSummary {
  total_records: number;
  channels_ingested: string[];
  min_date: string;
  max_date: string;
  total_spend: number;
  total_revenue: number;
  overall_roas: number;
  data_quality_score: number;
  audit_logs: string[];
  critical_warnings: CriticalWarning[];
  has_critical_warnings: boolean;
}

interface ForecastMetricValue {
  P10: number;
  P50: number;
  P90: number;
}
type ForecastPeriodData = Record<string, ForecastMetricValue>;
type ForecastsResponse = Record<string, ForecastPeriodData>;

interface DimensionsResponse {
  channels: string[];
  campaign_types: string[];
}

interface ScenarioForecastWindow {
  Revenue_P10: number;
  Revenue_P50: number;
  Revenue_P90: number;
  ROAS_P10: number;
  ROAS_P50: number;
  ROAS_P90: number;
  Spend_Expected: number;
}

interface ScenarioItem {
  id: string;
  name: string;
  description: string;
  cpc_change: string;
  conv_rate_change: string;
  revenue_multiplier: number;
  roas_multiplier: number;
  tag: string;
  forecasts: Record<string, ScenarioForecastWindow>;
}

interface RiskFactor {
  name: string;
  score: number;
  status: string;
  impact: string;
  mitigation: string;
}

interface RiskProfile {
  risk_score: number;
  risk_classification: string;
  badge_color: string;
  executive_risk_summary: string;
  risk_factors: RiskFactor[];
}

interface GrowthOpportunity {
  title: string;
  tag: string;
  insight: string;
}
interface RiskAssessmentItem {
  title: string;
  severity: string;
  insight: string;
}
interface BudgetRecommendation {
  channel: string;
  action: string;
  rationale: string;
}
interface InsightsResponse {
  executive_summary: string;
  growth_opportunities: GrowthOpportunity[];
  risk_assessment: RiskAssessmentItem[];
  budget_recommendations: BudgetRecommendation[];
  forecast_explanation: string;
}

interface ShapDriver {
  feature: string;
  shap_impact: number;
  description: string;
}
interface ChannelImportance {
  channel: string;
  contribution_share: number;
  roas_stability: string;
  importance_score: number;
}
interface CampaignImportance {
  campaign_name: string;
  channel: string;
  total_historical_revenue: number;
  average_roas: number;
  driver_status: string;
}
interface ExplainabilityResponse {
  top_revenue_drivers: ShapDriver[];
  top_roas_drivers: ShapDriver[];
  channel_importance: ChannelImportance[];
  campaign_importance: CampaignImportance[];
}

interface MonteCarloChannelDist {
  worst_case: number;
  expected_case: number;
  best_case: number;
}
interface HistogramBin {
  bin_center: number;
  bin_min: number;
  bin_max: number;
  frequency: number;
}
interface MonteCarloResponse {
  n_simulations: number;
  worst_case_revenue: number;
  expected_revenue: number;
  best_case_revenue: number;
  worst_case_roas: number;
  expected_roas: number;
  best_case_roas: number;
  channel_distributions: Record<string, MonteCarloChannelDist>;
  revenue_histogram: HistogramBin[];
  roas_histogram: HistogramBin[];
}

interface BudgetSimContribution {
  spend: number;
  revenue: number;
  roas: number;
  revenue_change_pct: number;
}
interface BudgetSimResponse {
  total_spend: number;
  total_revenue: number;
  total_roas: number;
  channel_contributions: Record<string, BudgetSimContribution>;
}

interface BudgetOptChannelRec {
  allocated_spend: number;
  expected_revenue: number;
  expected_roas: number;
  budget_share: number;
}
interface ConfidenceRange {
  revenue_p10: number;
  revenue_p50: number;
  revenue_p90: number;
  roas_p10: number;
  roas_p50: number;
  roas_p90: number;
}
interface BudgetOptResponse {
  max_budget: number;
  target_roas: number;
  recommended_total_spend: number;
  expected_total_revenue: number;
  expected_total_roas: number;
  channel_recommendations: Record<string, BudgetOptChannelRec>;
  confidence_range: ConfidenceRange;
}

interface ChatMessage {
  role: "user" | "bot";
  text: string;
}

type TabId =
  | "overview" | "validation" | "forecasts" | "scenarios"
  | "budget" | "montecarlo" | "explainability" | "risk" | "chat";

const TABS: { id: TabId; label: string; icon: React.ElementType }[] = [
  { id: "overview", label: "Overview", icon: LayoutDashboard },
  { id: "validation", label: "Data Validation", icon: Database },
  { id: "forecasts", label: "Forecasts", icon: TrendingUp },
  { id: "scenarios", label: "Scenarios", icon: Sliders },
  { id: "budget", label: "Budget Optimizer", icon: DollarSign },
  { id: "montecarlo", label: "Monte Carlo", icon: Layers },
  { id: "explainability", label: "Explainability", icon: Cpu },
  { id: "risk", label: "Risk & Insights", icon: ShieldAlert },
  { id: "chat", label: "Ask ForecastIQ", icon: MessageSquare },
];

// ─── Formatters ───────────────────────────────────────────────────────────
const fmtCompactCurrency = (n: number | undefined | null): string => {
  if (n === undefined || n === null || Number.isNaN(n)) return "$0";
  const abs = Math.abs(n);
  if (abs >= 1_000_000) return `${(n / 1_000_000).toFixed(2)}M`;
  if (abs >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return `${n.toFixed(0)}`;
};
const fmtCurrency = (n: number | undefined | null): string => {
  if (n === undefined || n === null || Number.isNaN(n)) return "$0.00";
  return `${n.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
};
const fmtRoas = (n: number | undefined | null): string => {
  if (n === undefined || n === null || Number.isNaN(n)) return "0.00x";
  return `${n.toFixed(2)}x`;
};
const fmtPct = (n: number | undefined | null, digits = 1): string => {
  if (n === undefined || n === null || Number.isNaN(n)) return "0%";
  return `${n.toFixed(digits)}%`;
};

// ─── Small shared UI primitives ──────────────────────────────────────────
function Card({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return (
    <div className={`rounded-xl border border-slate-800 bg-slate-900/60 backdrop-blur-sm p-5 ${className}`}>
      {children}
    </div>
  );
}

function KpiCard({ label, value, sub, icon: Icon, trend }: {
  label: string; value: string; sub?: string; icon: React.ElementType; trend?: "up" | "down";
}) {
  return (
    <Card className="flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <span className="text-xs uppercase tracking-wide text-slate-400">{label}</span>
        <Icon size={16} className="text-sky-400" />
      </div>
      <span className="text-2xl font-semibold text-slate-50">{value}</span>
      {sub && (
        <span className={`flex items-center gap-1 text-xs ${trend === "up" ? "text-emerald-400" : trend === "down" ? "text-rose-400" : "text-slate-400"}`}>
          {trend === "up" && <ArrowUpRight size={12} />}
          {trend === "down" && <ArrowDownRight size={12} />}
          {sub}
        </span>
      )}
    </Card>
  );
}

function LoadingBlock({ label }: { label: string }) {
  return (
    <div className="flex items-center gap-2 text-slate-400 text-sm py-10 justify-center">
      <RefreshCw size={16} className="animate-spin" />
      Loading {label}...
    </div>
  );
}

function ErrorBlock({ label, message }: { label: string; message: string }) {
  return (
    <div className="flex flex-col items-center gap-2 text-rose-400 text-sm py-10 justify-center text-center">
      <AlertTriangle size={20} />
      <span>Failed to load {label}.</span>
      <span className="text-xs text-slate-500">{message}</span>
      <span className="text-xs text-slate-500">Check that the FastAPI backend is running at {API_BASE}.</span>
    </div>
  );
}

// ─── Tab: Overview ─────────────────────────────────────────────────────────
function TabOverview({ data, loading, error }: { data: OverviewState | null; loading: boolean; error?: string }) {
  if (loading) return <LoadingBlock label="Overview" />;
  if (error) return <ErrorBlock label="Overview" message={error} />;
  if (!data) return <LoadingBlock label="Overview" />;
  return (
    <div className="flex flex-col gap-6">
      {data.critical_warnings.length > 0 && (
        <div className="rounded-xl border border-amber-500/30 bg-amber-500/10 p-4 flex items-start gap-3">
          <AlertTriangle size={18} className="text-amber-400 shrink-0 mt-0.5" />
          <div className="flex flex-col gap-1">
            <span className="text-sm font-semibold text-amber-300">{data.critical_warnings.length} data quality warning(s) detected</span>
            {data.critical_warnings.slice(0, 3).map((w, i) => (
              <span key={i} className="text-xs text-amber-200/80">{w.message}</span>
            ))}
          </div>
        </div>
      )}
      <div className="grid grid-cols-4 gap-4">
        <KpiCard label="Historical Spend" value={fmtCompactCurrency(data.total_historical_spend)} icon={DollarSign} />
        <KpiCard label="Historical Revenue" value={fmtCompactCurrency(data.total_historical_revenue)} icon={TrendingUp} />
        <KpiCard label="Overall ROAS" value={fmtRoas(data.overall_historical_roas)} icon={BarChart2} />
        <KpiCard label="Data Quality" value={fmtPct(data.data_quality_score, 1)} icon={CheckCircle2} sub={data.risk_classification} />
      </div>
      <div className="grid grid-cols-3 gap-6">
        <Card className="col-span-2">
          <div className="flex items-center justify-between mb-1">
            <h3 className="text-sm font-semibold text-slate-300">90-Day Revenue Trajectory</h3>
            <span className="text-[10px] text-slate-500 uppercase tracking-wide">{data.trajectory_method}</span>
          </div>
          <p className="text-xs text-slate-500 mb-4">P10 / P50 / P90 ensemble-weighted daily projection.</p>
          <ResponsiveContainer width="100%" height={260}>
            <AreaChart data={data.daily_trajectory} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
              <defs>
                <linearGradient id="p50Fill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#0EA5E9" stopOpacity={0.35} />
                  <stop offset="95%" stopColor="#0EA5E9" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
              <XAxis dataKey="day" tick={{ fontSize: 9, fill: "#64748B" }} interval={9} />
              <YAxis tick={{ fontSize: 10, fill: "#64748B" }} tickFormatter={v => fmtCompactCurrency(v)} />
              <Tooltip contentStyle={{ backgroundColor: "#0F172A", borderColor: "#334155", borderRadius: 8, color: "#fff" }} itemStyle={{ color: "#fff" }} labelStyle={{ color: "#94A3B8" }} formatter={(v: any) => fmtCurrency(v as number)} />
              <Area type="monotone" dataKey="p90" stroke="#334155" fill="transparent" strokeDasharray="4 4" strokeWidth={1} name="P90" />
              <Area type="monotone" dataKey="p50" stroke="#0EA5E9" fill="url(#p50Fill)" strokeWidth={2} name="P50" />
              <Area type="monotone" dataKey="p10" stroke="#334155" fill="transparent" strokeDasharray="4 4" strokeWidth={1} name="P10" />
            </AreaChart>
          </ResponsiveContainer>
        </Card>
        <Card>
          <h3 className="text-sm font-semibold text-slate-300 mb-1">Channel Attribution</h3>
          <p className="text-xs text-slate-500 mb-4">Historical revenue share by channel.</p>
          <ResponsiveContainer width="100%" height={200}>
            <RePieChart>
              <Pie data={data.channel_shares} dataKey="value" nameKey="name" cx="50%" cy="50%" innerRadius={45} outerRadius={75} paddingAngle={2}>
                {data.channel_shares.map((c, i) => <Cell key={i} fill={c.color} />)}
              </Pie>
              <Tooltip contentStyle={{ backgroundColor: "#0F172A", borderColor: "#334155", borderRadius: 8, color: "#fff" }} itemStyle={{ color: "#fff" }} labelStyle={{ color: "#94A3B8" }} formatter={(v: any) => `${v}%`} />
            </RePieChart>
          </ResponsiveContainer>
          <div className="flex flex-col gap-1.5 mt-2">
            {data.channel_shares.map((c, i) => (
              <div key={i} className="flex items-center justify-between text-xs">
                <span className="flex items-center gap-2 text-slate-300"><span className="w-2 h-2 rounded-full" style={{ backgroundColor: c.color }} />{c.name}</span>
                <span className="text-slate-400">{c.value}%</span>
              </div>
            ))}
          </div>
        </Card>
      </div>
      <Card>
        <h3 className="text-sm font-semibold text-slate-300 mb-2 flex items-center gap-2"><Sparkles size={14} className="text-sky-400" /> Executive Summary</h3>
        <p className="text-sm text-slate-400 leading-relaxed">{data.executive_summary}</p>
        <div className="grid grid-cols-2 gap-4 mt-4 pt-4 border-t border-slate-800">
          <div>
            <span className="text-xs text-slate-500">90-Day P50 Revenue Forecast</span>
            <div className="text-lg font-semibold text-slate-100">{fmtCompactCurrency(data.forecast_90d_p50_revenue)}</div>
          </div>
          <div>
            <span className="text-xs text-slate-500">90-Day P50 ROAS Forecast</span>
            <div className="text-lg font-semibold text-slate-100">{fmtRoas(data.forecast_90d_p50_roas)}</div>
          </div>
        </div>
      </Card>
    </div>
  );
}

// ─── Tab: Data Validation ───────────────────────────────────────────────────
function TabValidation({ data, loading, error }: { data: ValidationSummary | null; loading: boolean; error?: string }) {
  if (loading) return <LoadingBlock label="Data Validation" />;
  if (error) return <ErrorBlock label="Data Validation" message={error} />;
  if (!data) return <LoadingBlock label="Data Validation" />;
  return (
    <div className="flex flex-col gap-6">
      <div className="grid grid-cols-4 gap-4">
        <KpiCard label="Total Records" value={data.total_records.toLocaleString()} icon={Database} />
        <KpiCard label="Data Quality Score" value={fmtPct(data.data_quality_score, 1)} icon={CheckCircle2} />
        <KpiCard label="Date Range" value={`${data.min_date} → ${data.max_date}`} icon={TrendingUp} />
        <KpiCard label="Channels Ingested" value={String(data.channels_ingested.length)} icon={Database} sub={data.channels_ingested.join(", ")} />
      </div>
      {data.has_critical_warnings && (
        <Card className="border-amber-500/30">
          <h3 className="text-sm font-semibold text-amber-300 mb-3 flex items-center gap-2"><AlertTriangle size={16} /> Critical Warnings ({data.critical_warnings.length})</h3>
          <div className="flex flex-col gap-2">
            {data.critical_warnings.map((w, i) => (
              <div key={i} className="flex items-center justify-between text-xs bg-amber-500/5 rounded-lg px-3 py-2">
                <span className="text-amber-200/90">{w.message}</span>
                <span className="text-amber-400 font-bold shrink-0 ml-4">-{w.penalty} pts</span>
              </div>
            ))}
          </div>
        </Card>
      )}
      <Card>
        <h3 className="text-sm font-semibold text-slate-300 mb-3">Full Audit Log</h3>
        <div className="flex flex-col gap-1.5 max-h-96 overflow-y-auto font-mono">
          {data.audit_logs.map((log, i) => (
            <div key={i} className="text-[11px] text-slate-400 border-b border-slate-800/60 pb-1.5">{log}</div>
          ))}
        </div>
      </Card>
    </div>
  );
}

// ─── Tab: Forecasts ─────────────────────────────────────────────────────
const PERIOD_LABELS: Record<string, string> = { "30_days": "30 Days", "60_days": "60 Days", "90_days": "90 Days" };

function TabForecasts({
  forecasts, dimensions, selectedDimension, onSelectDimension, loading, error,
}: {
  forecasts: ForecastsResponse | null;
  dimensions: DimensionsResponse | null;
  selectedDimension: string;
  onSelectDimension: (d: string) => void;
  loading: boolean;
  error?: string;
}) {
  const dimensionOptions = ["Overall", ...(dimensions?.channels ?? []), ...(dimensions?.campaign_types ?? [])];
  return (
    <div className="flex flex-col gap-6">
      <Card>
        <h3 className="text-sm font-semibold text-slate-300 mb-3">Dimension</h3>
        <div className="flex flex-wrap gap-2">
          {dimensionOptions.map(d => (
            <button key={d} onClick={() => onSelectDimension(d)}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium border transition-colors ${selectedDimension === d ? "bg-sky-500 border-sky-500 text-white" : "bg-slate-800/60 border-slate-700 text-slate-300 hover:border-sky-500/50"}`}>
              {d}
            </button>
          ))}
        </div>
      </Card>
      {loading ? <LoadingBlock label="Forecasts" /> : error ? <ErrorBlock label="Forecasts" message={error} /> : !forecasts ? <LoadingBlock label="Forecasts" /> : (
        <div className="grid grid-cols-3 gap-6">
          {Object.entries(forecasts).map(([period, metrics]) => (
            <Card key={period}>
              <h3 className="text-sm font-semibold text-slate-300 mb-4">{PERIOD_LABELS[period] ?? period}</h3>
              <div className="flex flex-col gap-3">
                {Object.entries(metrics).map(([metric, vals]) => (
                  <div key={metric} className="bg-slate-800/40 rounded-lg p-3">
                    <span className="text-xs font-semibold text-slate-200">{metric}</span>
                    <div className="grid grid-cols-3 gap-2 mt-2 text-center">
                      <div>
                        <div className="text-[10px] text-slate-500">P10</div>
                        <div className="text-xs font-medium text-slate-300">{metric.toLowerCase().includes("roas") ? fmtRoas(vals.P10) : fmtCompactCurrency(vals.P10)}</div>
                      </div>
                      <div>
                        <div className="text-[10px] text-sky-500">P50</div>
                        <div className="text-xs font-bold text-sky-300">{metric.toLowerCase().includes("roas") ? fmtRoas(vals.P50) : fmtCompactCurrency(vals.P50)}</div>
                      </div>
                      <div>
                        <div className="text-[10px] text-slate-500">P90</div>
                        <div className="text-xs font-medium text-slate-300">{metric.toLowerCase().includes("roas") ? fmtRoas(vals.P90) : fmtCompactCurrency(vals.P90)}</div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}

// ─── Tab: Scenarios ────────────────────────────────────────────────────
function TabScenarios({ scenarios, loading, error }: { scenarios: ScenarioItem[] | null; loading: boolean; error?: string }) {
  const [activeWindow, setActiveWindow] = useState<string>("30_days");
  if (loading) return <LoadingBlock label="Scenarios" />;
  if (error) return <ErrorBlock label="Scenarios" message={error} />;
  if (!scenarios) return <LoadingBlock label="Scenarios" />;
  return (
    <div className="flex flex-col gap-6">
      <Card>
        <div className="flex items-center gap-2">
          {Object.keys(PERIOD_LABELS).map(w => (
            <button key={w} onClick={() => setActiveWindow(w)}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium border transition-colors ${activeWindow === w ? "bg-sky-500 border-sky-500 text-white" : "bg-slate-800/60 border-slate-700 text-slate-300 hover:border-sky-500/50"}`}>
              {PERIOD_LABELS[w]}
            </button>
          ))}
        </div>
      </Card>
      <div className="grid grid-cols-2 gap-6">
        {scenarios.map(s => {
          const w = s.forecasts[activeWindow];
          return (
            <Card key={s.id}>
              <div className="flex items-center justify-between mb-2">
                <h3 className="text-sm font-bold text-slate-100">{s.name}</h3>
                <span className="text-[10px] px-2 py-0.5 rounded bg-sky-500/10 text-sky-400 border border-sky-500/20 font-semibold">{s.tag}</span>
              </div>
              <p className="text-xs text-slate-400 mb-4">{s.description}</p>
              {w && (
                <div className="grid grid-cols-2 gap-4 mb-4">
                  <div className="bg-slate-800/40 rounded-lg p-3">
                    <span className="text-[10px] text-slate-500 uppercase">Revenue P50</span>
                    <div className="text-lg font-bold text-slate-100">{fmtCompactCurrency(w.Revenue_P50)}</div>
                    <div className="text-[10px] text-slate-500">{fmtCompactCurrency(w.Revenue_P10)} – {fmtCompactCurrency(w.Revenue_P90)}</div>
                  </div>
                  <div className="bg-slate-800/40 rounded-lg p-3">
                    <span className="text-[10px] text-slate-500 uppercase">ROAS P50</span>
                    <div className="text-lg font-bold text-slate-100">{fmtRoas(w.ROAS_P50)}</div>
                    <div className="text-[10px] text-slate-500">{fmtRoas(w.ROAS_P10)} – {fmtRoas(w.ROAS_P90)}</div>
                  </div>
                </div>
              )}
              <div className="flex gap-4 text-[11px] text-slate-500 border-t border-slate-800 pt-3">
                <span>CPC {s.cpc_change}</span>
                <span>Conv. Rate {s.conv_rate_change}</span>
              </div>
            </Card>
          );
        })}
      </div>
    </div>
  );
}

// ─── Tab: Budget Optimizer ────────────────────────────────────────────
function TabBudget({
  simInputs, onSimChange, simResult, simLoading,
  optInputs, onOptChange, optResult, optLoading, onRunOptimize,
}: {
  simInputs: { google_pct: number; meta_pct: number; bing_pct: number };
  onSimChange: (k: "google_pct" | "meta_pct" | "bing_pct", v: number) => void;
  simResult: BudgetSimResponse | null;
  simLoading: boolean;
  optInputs: { max_budget: number; target_roas: number };
  onOptChange: (k: "max_budget" | "target_roas", v: number) => void;
  optResult: BudgetOptResponse | null;
  optLoading: boolean;
  onRunOptimize: () => void;
}) {
  const sliderRows: { key: "google_pct" | "meta_pct" | "bing_pct"; label: string }[] = [
    { key: "google_pct", label: "Google Ads" },
    { key: "meta_pct", label: "Meta Ads" },
    { key: "bing_pct", label: "Bing Ads" },
  ];
  return (
    <div className="flex flex-col gap-6">
      <div className="grid grid-cols-2 gap-6">
        <Card>
          <h3 className="text-sm font-semibold text-slate-300 mb-1">Live Budget Simulator</h3>
          <p className="text-xs text-slate-500 mb-4">Adjust spend % per channel against your current 30-day run-rate.</p>
          <div className="flex flex-col gap-4">
            {sliderRows.map(row => (
              <div key={row.key}>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-slate-300 font-medium">{row.label}</span>
                  <span className={simInputs[row.key] > 0 ? "text-emerald-400" : simInputs[row.key] < 0 ? "text-rose-400" : "text-slate-400"}>{simInputs[row.key] > 0 ? "+" : ""}{simInputs[row.key]}%</span>
                </div>
                <input type="range" min={-50} max={100} step={5} value={simInputs[row.key]}
                  onChange={e => onSimChange(row.key, Number(e.target.value))}
                  className="w-full accent-sky-500" />
              </div>
            ))}
          </div>
          {simLoading ? <LoadingBlock label="simulation" /> : simResult && (
            <div className="grid grid-cols-3 gap-3 mt-5 pt-4 border-t border-slate-800">
              <div>
                <span className="text-[10px] text-slate-500 uppercase">Total Spend</span>
                <div className="text-sm font-bold text-slate-100">{fmtCompactCurrency(simResult.total_spend)}</div>
              </div>
              <div>
                <span className="text-[10px] text-slate-500 uppercase">Total Revenue</span>
                <div className="text-sm font-bold text-slate-100">{fmtCompactCurrency(simResult.total_revenue)}</div>
              </div>
              <div>
                <span className="text-[10px] text-slate-500 uppercase">Total ROAS</span>
                <div className="text-sm font-bold text-slate-100">{fmtRoas(simResult.total_roas)}</div>
              </div>
            </div>
          )}
        </Card>
        <Card>
          <h3 className="text-sm font-semibold text-slate-300 mb-1">Optuna Global Optimizer</h3>
          <p className="text-xs text-slate-500 mb-4">Find the allocation that maximizes revenue at your target ROAS.</p>
          <div className="flex flex-col gap-4 mb-4">
            <div>
              <label className="text-xs text-slate-400">Max Budget</label>
              <input type="number" value={optInputs.max_budget} onChange={e => onOptChange("max_budget", Number(e.target.value))}
                className="w-full mt-1 bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white" />
            </div>
            <div>
              <label className="text-xs text-slate-400">Target ROAS</label>
              <input type="number" step={0.1} value={optInputs.target_roas} onChange={e => onOptChange("target_roas", Number(e.target.value))}
                className="w-full mt-1 bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white" />
            </div>
            <button onClick={onRunOptimize} disabled={optLoading}
              className="flex items-center justify-center gap-2 bg-sky-500 hover:bg-sky-400 text-white font-bold rounded-lg py-2.5 text-sm transition-colors disabled:opacity-50">
              <Zap size={14} /> {optLoading ? "Optimizing…" : "Run Optimization"}
            </button>
          </div>
          {optResult && (
            <div className="flex flex-col gap-2 pt-4 border-t border-slate-800">
              {Object.entries(optResult.channel_recommendations).map(([ch, rec]) => (
                <div key={ch} className="flex items-center justify-between text-xs bg-slate-800/40 rounded-lg px-3 py-2">
                  <span className="font-medium text-slate-200 w-24">{ch}</span>
                  <span className="text-slate-400">{fmtCompactCurrency(rec.allocated_spend)}</span>
                  <span className="text-slate-400">{fmtRoas(rec.expected_roas)}</span>
                  <span className="text-sky-400 font-semibold">{fmtPct(rec.budget_share)}</span>
                </div>
              ))}
              <div className="flex justify-between text-xs mt-2 pt-2 border-t border-slate-800">
                <span className="text-slate-400">Expected Revenue (P10–P90)</span>
                <span className="text-slate-200 font-semibold">{fmtCompactCurrency(optResult.confidence_range.revenue_p10)} – {fmtCompactCurrency(optResult.confidence_range.revenue_p90)}</span>
              </div>
            </div>
          )}
        </Card>
      </div>
    </div>
  );
}

// ─── Tab: Monte Carlo ────────────────────────────────────────────
function TabMontecarlo({ mc, loading, error }: { mc: MonteCarloResponse | null; loading: boolean; error?: string }) {
  if (loading) return <LoadingBlock label="Monte Carlo" />;
  if (error) return <ErrorBlock label="Monte Carlo" message={error} />;
  if (!mc) return <LoadingBlock label="Monte Carlo" />;
  return (
    <div className="flex flex-col gap-6">
      <div className="grid grid-cols-3 gap-4">
        <KpiCard label="Worst Case Revenue" value={fmtCompactCurrency(mc.worst_case_revenue)} icon={ArrowDownRight} trend="down" sub={`${fmtRoas(mc.worst_case_roas)} ROAS`} />
        <KpiCard label="Expected Revenue" value={fmtCompactCurrency(mc.expected_revenue)} icon={TrendingUp} sub={`${fmtRoas(mc.expected_roas)} ROAS`} />
        <KpiCard label="Best Case Revenue" value={fmtCompactCurrency(mc.best_case_revenue)} icon={ArrowUpRight} trend="up" sub={`${fmtRoas(mc.best_case_roas)} ROAS`} />
      </div>
      <Card>
        <h3 className="text-sm font-semibold text-slate-300 mb-1">Revenue Distribution ({mc.n_simulations.toLocaleString()} simulations)</h3>
        <p className="text-xs text-slate-500 mb-4">Histogram of simulated 30-day portfolio revenue outcomes.</p>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={mc.revenue_histogram} margin={{ top: 4, right: 16, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
            <XAxis dataKey="bin_center" tickFormatter={v => fmtCompactCurrency(v)} tick={{ fontSize: 10, fill: "#64748B" }} />
            <YAxis tick={{ fontSize: 10, fill: "#64748B" }} />
            <Tooltip contentStyle={{ backgroundColor: "#0F172A", borderColor: "#334155", borderRadius: 8, color: "#fff" }} itemStyle={{ color: "#fff" }} labelStyle={{ color: "#94A3B8" }} formatter={(v: any) => [v, "Frequency"]} labelFormatter={(v: any) => fmtCurrency(v)} />
            <Bar dataKey="frequency" fill="#0EA5E9" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>
      <Card>
        <h3 className="text-sm font-semibold text-slate-300 mb-4">Channel Distributions</h3>
        <div className="grid grid-cols-3 gap-4">
          {Object.entries(mc.channel_distributions).map(([ch, d]) => (
            <div key={ch} className="bg-slate-800/40 rounded-xl p-4">
              <p className="text-xs font-bold text-slate-300 mb-3">{ch}</p>
              {[["Worst Case", d.worst_case], ["Expected", d.expected_case], ["Best Case", d.best_case]].map(([label, val]) => (
                <div key={label} className="flex justify-between text-xs mb-1">
                  <span className="text-slate-400">{label}</span>
                  <span className="text-slate-200 font-medium">{fmtCompactCurrency(val as number)}</span>
                </div>
              ))}
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}

// ─── Tab: Explainability ─────────────────────────────────────────
function ShapBarChart({ data, mode, height }: { data: ShapDriver[]; mode: "revenue" | "roas"; height: number }) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} layout="vertical" margin={{ top: 4, right: 16, left: 100, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
        <XAxis type="number" tick={{ fontSize: 10, fill: "#64748B" }} tickFormatter={v => mode === "revenue" ? `${(v / 1000).toFixed(0)}k` : `${v}x`} />
        <YAxis dataKey="feature" type="category" tick={{ fontSize: 10, fill: "#94A3B8" }} width={140} />
        <Tooltip contentStyle={{ backgroundColor: "#0F172A", borderColor: "#334155", borderRadius: 8, color: "#fff" }} itemStyle={{ color: "#fff" }} labelStyle={{ color: "#94A3B8" }} formatter={(v: any) => mode === "revenue" ? [`${Number(v).toLocaleString("en-US", { maximumFractionDigits: 0 })}`, "Causal Impact"] : [`${Number(v).toFixed(2)}x`, "ROAS Impact"]} />
        <Bar dataKey="shap_impact" fill={mode === "revenue" ? "#0EA5E9" : "#14B8A6"} radius={[0, 8, 8, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}

function TabExplainability({ data, loading, error }: { data: ExplainabilityResponse | null; loading: boolean; error?: string }) {
  const [showAllCampaigns, setShowAllCampaigns] = useState(false);
  const [campaignSort, setCampaignSort] = useState<"revenue" | "best" | "worst">("revenue");
  const [expandedChart, setExpandedChart] = useState<"revenue" | "roas" | null>(null);
  if (loading) return <LoadingBlock label="Explainability" />;
  if (error) return <ErrorBlock label="Explainability" message={error} />;
  if (!data) return <LoadingBlock label="Explainability" />;

  const sortedCampaigns = [...data.campaign_importance].sort((a, b) => {
    if (campaignSort === "best") return b.average_roas - a.average_roas;
    if (campaignSort === "worst") return a.average_roas - b.average_roas;
    return b.total_historical_revenue - a.total_historical_revenue;
  });
  const visibleCampaigns = showAllCampaigns ? sortedCampaigns : sortedCampaigns.slice(0, 3);

  return (
    <div className="flex flex-col gap-6">
      <div className="grid grid-cols-2 gap-6">
        <Card>
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-semibold text-slate-300">Top Revenue Drivers (SHAP Value Impact)</h3>
            <button onClick={() => setExpandedChart("revenue")} className="text-slate-500 hover:text-sky-400 transition-colors" title="Expand chart">
              <Maximize2 size={14} />
            </button>
          </div>
          <ShapBarChart data={data.top_revenue_drivers} mode="revenue" height={280} />
        </Card>
        <Card>
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-semibold text-slate-300">Top ROAS Drivers (SHAP Multiplier Impact)</h3>
            <button onClick={() => setExpandedChart("roas")} className="text-slate-500 hover:text-sky-400 transition-colors" title="Expand chart">
              <Maximize2 size={14} />
            </button>
          </div>
          <ShapBarChart data={data.top_roas_drivers} mode="roas" height={280} />
        </Card>
      </div>
      <Card>
        <h3 className="text-sm font-semibold text-slate-300 mb-4">Channel Importance Rankings</h3>
        <div className="flex flex-col gap-2">
          {data.channel_importance.map((c, i) => (
            <div key={i} className="flex items-center justify-between bg-slate-800/40 rounded-lg px-4 py-2.5 text-sm">
              <span className="font-medium text-slate-200 w-32">{c.channel}</span>
              <span className="text-slate-400">{fmtPct(c.contribution_share)} contribution</span>
              <span className={`text-xs px-2 py-0.5 rounded font-semibold ${c.roas_stability.startsWith("High") ? "bg-emerald-500/10 text-emerald-400" : c.roas_stability.startsWith("Medium") ? "bg-amber-500/10 text-amber-400" : "bg-rose-500/10 text-rose-400"}`}>{c.roas_stability}</span>
              <span className="text-sky-400 font-bold">{c.importance_score} score</span>
            </div>
          ))}
        </div>
      </Card>
      <Card>
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-sm font-semibold text-slate-300">Top Campaigns by Historical Revenue</h3>
          <div className="flex gap-2">
            {([["revenue", "Revenue"], ["best", "Best ROAS"], ["worst", "Worst ROAS"]] as const).map(([key, label]) => (
              <button key={key} onClick={() => setCampaignSort(key)}
                className={`px-2.5 py-1 rounded-lg text-[11px] font-medium border transition-colors ${campaignSort === key ? "bg-sky-500 border-sky-500 text-white" : "bg-slate-800/60 border-slate-700 text-slate-300 hover:border-sky-500/50"}`}>
                {label}
              </button>
            ))}
          </div>
        </div>
        <div className="flex flex-col gap-2">
          {visibleCampaigns.map((c, i) => (
            <div key={i} className="flex items-center justify-between bg-slate-800/40 rounded-lg px-4 py-2.5 text-sm">
              <div className="flex flex-col">
                <span className="font-medium text-slate-200">{c.campaign_name}</span>
                <span className="text-[10px] text-slate-500">{c.channel}</span>
              </div>
              <span className="text-slate-400">{fmtCompactCurrency(c.total_historical_revenue)}</span>
              <span className="text-slate-400">{fmtRoas(c.average_roas)}</span>
              <span className={`text-[10px] px-2 py-0.5 rounded font-semibold ${c.driver_status === "Primary Bedrock" ? "bg-emerald-500/10 text-emerald-400" : "bg-sky-500/10 text-sky-400"}`}>{c.driver_status}</span>
            </div>
          ))}
        </div>
        {sortedCampaigns.length > 3 && (
          <button onClick={() => setShowAllCampaigns(v => !v)}
            className="w-full mt-3 py-2 rounded-lg text-xs font-medium text-sky-400 border border-slate-800 hover:border-sky-500/40 hover:bg-slate-800/40 transition-colors">
            {showAllCampaigns ? "Show less" : `Show ${sortedCampaigns.length - 3} more (${sortedCampaigns.length} total)`}
          </button>
        )}
      </Card>
      {expandedChart && (
        <div className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-center justify-center p-8" onClick={() => setExpandedChart(null)}>
          <div className="bg-slate-900 border border-slate-700 rounded-xl p-6 w-full max-w-4xl" onClick={e => e.stopPropagation()}>
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-base font-semibold text-slate-200">
                {expandedChart === "revenue" ? "Top Revenue Drivers (SHAP Value Impact)" : "Top ROAS Drivers (SHAP Multiplier Impact)"}
              </h3>
              <button onClick={() => setExpandedChart(null)} className="text-slate-400 hover:text-white transition-colors">
                <X size={18} />
              </button>
            </div>
            <ShapBarChart
              data={expandedChart === "revenue" ? data.top_revenue_drivers : data.top_roas_drivers}
              mode={expandedChart}
              height={520}
            />
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Tab: Risk & Insights ────────────────────────────────────────────
const POSITIVE_STATUSES = new Set(["Healthy", "Diversified", "Stable", "Excellent", "Low"]);
const SEVERITY_STYLES: Record<string, string> = {
  High: "text-rose-400 bg-rose-500/10 border-rose-500/20",
  Medium: "text-amber-400 bg-amber-500/10 border-amber-500/20",
  Low: "text-emerald-400 bg-emerald-500/10 border-emerald-500/20",
};

function RiskGauge({ score }: { score: number }) {
  const clamped = Math.max(0, Math.min(100, score));
  const color = clamped >= 70 ? "#F43F5E" : clamped >= 40 ? "#F59E0B" : "#10B981";
  const circumference = 2 * Math.PI * 42;
  const offset = circumference * (1 - clamped / 100);
  return (
    <div className="relative w-28 h-28 shrink-0">
      <svg viewBox="0 0 100 100" className="w-28 h-28 -rotate-90">
        <circle cx="50" cy="50" r="42" fill="none" stroke="#1E293B" strokeWidth="10" />
        <circle cx="50" cy="50" r="42" fill="none" stroke={color} strokeWidth="10" strokeLinecap="round"
          strokeDasharray={circumference} strokeDashoffset={offset} style={{ transition: "stroke-dashoffset 0.6s ease" }} />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-2xl font-black text-white">{clamped}</span>
        <span className="text-[9px] text-slate-500 uppercase tracking-wide">/ 100</span>
      </div>
    </div>
  );
}

function TabRisk({ risk, insights, loading, error }: { risk: RiskProfile | null; insights: InsightsResponse | null; loading: boolean; error?: string }) {
  if (loading) return <LoadingBlock label="Risk & Insights" />;
  if (error) return <ErrorBlock label="Risk & Insights" message={error} />;
  if (!risk) return <LoadingBlock label="Risk & Insights" />;
  return (
    <div className="flex flex-col gap-6">
      <Card>
        <div className="flex items-start gap-6">
          <RiskGauge score={risk.risk_score} />
          <div className="flex-1">
            <div className="flex items-center gap-3 mb-2">
              <h3 className="text-sm font-semibold text-slate-300">Risk Intelligence Profile</h3>
              <span className={`px-2 py-0.5 rounded text-xs font-bold border ${risk.badge_color}`}>{risk.risk_classification}</span>
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">{risk.executive_risk_summary}</p>
          </div>
        </div>
      </Card>

      <div className="grid grid-cols-2 gap-4">
        {risk.risk_factors.map((f, i) => {
          const positive = POSITIVE_STATUSES.has(f.status);
          return (
            <Card key={i} className="!p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-semibold text-slate-200">{f.name}</span>
                <span className={`text-xs font-bold ${positive ? "text-emerald-400" : "text-amber-400"}`}>{f.status}</span>
              </div>
              <div className="h-1.5 rounded-full bg-slate-800 overflow-hidden mb-2">
                <div className={`h-full rounded-full ${positive ? "bg-emerald-500" : "bg-amber-500"}`} style={{ width: `${Math.max(4, Math.min(100, f.score))}%` }} />
              </div>
              <p className="text-[11px] text-slate-400 leading-relaxed">{f.mitigation}</p>
            </Card>
          );
        })}
      </div>
      {insights && (
        <div className="grid grid-cols-3 gap-4">
          <Card>
            <h3 className="text-sm font-semibold text-slate-300 mb-3 flex items-center gap-2"><Sparkles size={14} className="text-emerald-400" /> Growth Opportunities</h3>
            <div className="flex flex-col gap-3">
              {insights.growth_opportunities.map((g, i) => (
                <div key={i} className="bg-slate-800/40 rounded-lg p-3">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-xs font-bold text-emerald-400">{g.title}</span>
                    <span className="text-[9px] px-1.5 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">{g.tag}</span>
                  </div>
                  <p className="text-[11px] text-slate-400 leading-relaxed">{g.insight}</p>
                </div>
              ))}
            </div>
          </Card>
          <Card>
            <h3 className="text-sm font-semibold text-slate-300 mb-3 flex items-center gap-2"><AlertTriangle size={14} className="text-amber-400" /> Risk Assessment</h3>
            <div className="flex flex-col gap-3">
              {insights.risk_assessment.map((r, i) => (
                <div key={i} className="bg-slate-800/40 rounded-lg p-3">
                  <span className={`inline-block text-[10px] px-1.5 py-0.5 rounded font-semibold border mb-1 ${SEVERITY_STYLES[r.severity] ?? "text-slate-300 bg-slate-700/30 border-slate-700"}`}>{r.severity}</span>
                  <p className="text-xs font-bold text-slate-200 mb-1">{r.title}</p>
                  <p className="text-[11px] text-slate-400 leading-relaxed">{r.insight}</p>
                </div>
              ))}
            </div>
          </Card>
          <Card>
            <h3 className="text-sm font-semibold text-slate-300 mb-3 flex items-center gap-2"><DollarSign size={14} className="text-sky-400" /> Budget Recommendations</h3>
            <div className="flex flex-col gap-2">
              {insights.budget_recommendations.map((b, i) => (
                <div key={i} className="bg-slate-800/40 rounded-lg p-3">
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-bold text-sky-400 text-xs">{b.channel}</span>
                    <span className={`text-[11px] font-semibold ${b.action.includes("Increase") ? "text-emerald-400" : b.action.includes("Maintain") ? "text-amber-400" : "text-sky-400"}`}>{b.action}</span>
                  </div>
                  <p className="text-[11px] text-slate-400 leading-relaxed">{b.rationale}</p>
                </div>
              ))}
            </div>
          </Card>
        </div>
      )}
      {insights && (
        <Card>
          <h3 className="text-sm font-semibold text-slate-300 mb-2">How This Forecast Was Built</h3>
          <p className="text-xs text-slate-400 leading-relaxed">{insights.forecast_explanation}</p>
        </Card>
      )}
    </div>
  );
}

// ─── Tab: Chat ───────────────────────────────────────────────────────────
function TabChat({ messages, input, onInput, onSend, loading }: {
  messages: ChatMessage[]; input: string; onInput: (v: string) => void; onSend: () => void; loading: boolean;
}) {
  const bottomRef = React.useRef<HTMLDivElement>(null);
  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages]);
  return (
    <div className="flex flex-col h-[calc(100vh-9rem)]">
      <Card className="flex-1 overflow-y-auto flex flex-col gap-3 mb-4">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full gap-3 text-slate-500">
            <MessageSquare size={32} />
            <p className="text-sm">Ask ForecastIQ anything about your campaign data and forecasts.</p>
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
            <div className={`max-w-[75%] rounded-2xl px-4 py-2.5 text-sm leading-relaxed ${m.role === "user" ? "bg-sky-500 text-white" : "bg-slate-800 text-slate-200 border border-slate-700"}`}>
              {m.text}
            </div>
          </div>
        ))}
        {loading && (
          <div className="flex justify-start">
            <div className="bg-slate-800 rounded-2xl px-4 py-2.5 border border-slate-700 flex items-center gap-2 text-slate-400 text-sm">
              <RefreshCw size={13} className="animate-spin" /> ForecastIQ is thinking…
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </Card>
      <div className="flex gap-3">
        <input value={input} onChange={e => onInput(e.target.value)}
          onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); onSend(); }}}
          placeholder="Ask about your revenue, ROAS, campaigns…"
          className="flex-1 bg-slate-800 border border-slate-700 rounded-xl px-4 py-2.5 text-sm text-white placeholder:text-slate-500 focus:outline-none focus:border-sky-500" />
        <button onClick={onSend} disabled={loading || !input.trim()}
          className="px-5 py-2.5 rounded-xl bg-sky-500 hover:bg-sky-400 text-white font-bold transition-colors disabled:opacity-50 flex items-center gap-2 text-sm">
          <Send size={14} /> Send
        </button>
      </div>
    </div>
  );
}

// ─── Main Page ───────────────────────────────────────────────────────────────
// Endpoint map, verified directly against backend/src/main.py (not guessed):
//   GET  /api/status            -> StatusState
//   GET  /api/overview          -> OverviewState
//   GET  /api/validation        -> ValidationSummary
//   GET  /api/forecasts?dimension=X -> ForecastsResponse
//   GET  /api/dimensions        -> DimensionsResponse
//   GET  /api/scenarios         -> ScenarioItem[]
//   GET  /api/simulations       -> MonteCarloResponse   (NOT /api/montecarlo)
//   GET  /api/explainability    -> ExplainabilityResponse
//   GET  /api/risk              -> RiskProfile
//   GET  /api/insights          -> InsightsResponse
//   POST /api/simulate-budget   -> BudgetSimResponse
//   POST /api/optimize-budget   -> BudgetOptResponse
//   POST /api/chat              -> { question, answer }
//   GET  /api/report/pdf        -> PDF file stream

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, init);
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`${res.status} ${res.statusText}${body ? " — " + body.slice(0, 200) : ""}`);
  }
  return res.json() as Promise<T>;
}

interface LoadState<T> {
  data: T | null;
  loading: boolean;
  error?: string;
}

export default function Page() {
  const [activeTab, setActiveTab] = useState<TabId>("overview");

  const [status, setStatus] = useState<LoadState<StatusState>>({ data: null, loading: true });
  const [overview, setOverview] = useState<LoadState<OverviewState>>({ data: null, loading: true });
  const [validation, setValidation] = useState<LoadState<ValidationSummary>>({ data: null, loading: true });
  const [dimensions, setDimensions] = useState<LoadState<DimensionsResponse>>({ data: null, loading: true });
  const [scenarios, setScenarios] = useState<LoadState<ScenarioItem[]>>({ data: null, loading: true });
  const [montecarlo, setMontecarlo] = useState<LoadState<MonteCarloResponse>>({ data: null, loading: true });
  const [explainability, setExplainability] = useState<LoadState<ExplainabilityResponse>>({ data: null, loading: true });
  const [risk, setRisk] = useState<LoadState<RiskProfile>>({ data: null, loading: true });
  const [insights, setInsights] = useState<LoadState<InsightsResponse>>({ data: null, loading: true });

  const [selectedDimension, setSelectedDimension] = useState<string>("Overall");
  const [forecasts, setForecasts] = useState<LoadState<ForecastsResponse>>({ data: null, loading: true });

  const [simInputs, setSimInputs] = useState({ google_pct: 0, meta_pct: 0, bing_pct: 0 });
  const [simResult, setSimResult] = useState<BudgetSimResponse | null>(null);
  const [simLoading, setSimLoading] = useState(false);

  const [optInputs, setOptInputs] = useState({ max_budget: 100000, target_roas: 4.5 });
  const [optResult, setOptResult] = useState<BudgetOptResponse | null>(null);
  const [optLoading, setOptLoading] = useState(false);

  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState("");
  const [chatLoading, setChatLoading] = useState(false);

  // ── Initial fetch-all on mount ──────────────────────────────────────────
  useEffect(() => {
    fetchJson<StatusState>("/api/status")
      .then(data => setStatus({ data, loading: false }))
      .catch(e => setStatus({ data: null, loading: false, error: String(e.message ?? e) }));

    fetchJson<OverviewState>("/api/overview")
      .then(data => setOverview({ data, loading: false }))
      .catch(e => setOverview({ data: null, loading: false, error: String(e.message ?? e) }));

    fetchJson<ValidationSummary>("/api/validation")
      .then(data => setValidation({ data, loading: false }))
      .catch(e => setValidation({ data: null, loading: false, error: String(e.message ?? e) }));

    fetchJson<DimensionsResponse>("/api/dimensions")
      .then(data => setDimensions({ data, loading: false }))
      .catch(e => setDimensions({ data: null, loading: false, error: String(e.message ?? e) }));

    fetchJson<ScenarioItem[]>("/api/scenarios")
      .then(data => setScenarios({ data, loading: false }))
      .catch(e => setScenarios({ data: null, loading: false, error: String(e.message ?? e) }));

    // NOTE: this is /api/simulations, not /api/montecarlo — verified against main.py.
    fetchJson<MonteCarloResponse>("/api/simulations")
      .then(data => setMontecarlo({ data, loading: false }))
      .catch(e => setMontecarlo({ data: null, loading: false, error: String(e.message ?? e) }));

    fetchJson<ExplainabilityResponse>("/api/explainability")
      .then(data => setExplainability({ data, loading: false }))
      .catch(e => setExplainability({ data: null, loading: false, error: String(e.message ?? e) }));

    fetchJson<RiskProfile>("/api/risk")
      .then(data => setRisk({ data, loading: false }))
      .catch(e => setRisk({ data: null, loading: false, error: String(e.message ?? e) }));

    fetchJson<InsightsResponse>("/api/insights")
      .then(data => setInsights({ data, loading: false }))
      .catch(e => setInsights({ data: null, loading: false, error: String(e.message ?? e) }));
  }, []);

  // ── Forecasts: re-fetch whenever the selected dimension changes ────────
  useEffect(() => {
    setForecasts(prev => ({ ...prev, loading: true }));
    fetchJson<ForecastsResponse>(`/api/forecasts?dimension=${encodeURIComponent(selectedDimension)}`)
      .then(data => setForecasts({ data, loading: false }))
      .catch(e => setForecasts({ data: null, loading: false, error: String(e.message ?? e) }));
  }, [selectedDimension]);

  // ── Budget simulator: debounce-free, re-run on every slider change ─────
  useEffect(() => {
    setSimLoading(true);
    fetchJson<BudgetSimResponse>("/api/simulate-budget", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(simInputs),
    })
      .then(data => { setSimResult(data); setSimLoading(false); })
      .catch(() => setSimLoading(false));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [simInputs]);

  const handleSimChange = useCallback((k: "google_pct" | "meta_pct" | "bing_pct", v: number) => {
    setSimInputs(prev => ({ ...prev, [k]: v }));
  }, []);

  const handleOptChange = useCallback((k: "max_budget" | "target_roas", v: number) => {
    setOptInputs(prev => ({ ...prev, [k]: v }));
  }, []);

  const handleRunOptimize = useCallback(() => {
    setOptLoading(true);
    fetchJson<BudgetOptResponse>("/api/optimize-budget", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(optInputs),
    })
      .then(data => { setOptResult(data); setOptLoading(false); })
      .catch(() => setOptLoading(false));
  }, [optInputs]);

  const handleSendChat = useCallback(() => {
    const question = chatInput.trim();
    if (!question) return;
    setChatMessages(prev => [...prev, { role: "user", text: question }]);
    setChatInput("");
    setChatLoading(true);
    fetchJson<{ question: string; answer: string }>("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    })
      .then(res => {
        setChatMessages(prev => [...prev, { role: "bot", text: res.answer }]);
        setChatLoading(false);
      })
      .catch(e => {
        setChatMessages(prev => [...prev, { role: "bot", text: `Sorry, something went wrong: ${String(e.message ?? e)}` }]);
        setChatLoading(false);
      });
  }, [chatInput]);

  const handleDownloadPdf = useCallback(() => {
    window.open(`${API_BASE}/api/report/pdf`, "_blank");
  }, []);

  return (
    <div className="flex min-h-screen bg-slate-950">
      {/* Sidebar */}
      <aside className="w-60 shrink-0 border-r border-slate-800 bg-slate-900/40 flex flex-col">
        <div className="px-5 py-6 border-b border-slate-800">
          <h1 className="text-lg font-black text-white tracking-tight">ForecastIQ</h1>
          <p className="text-[10px] text-slate-500 uppercase tracking-wide mt-0.5">AI Revenue Intelligence</p>
        </div>
        <nav className="flex-1 py-4 flex flex-col gap-1 px-3">
          {TABS.map(tab => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button key={tab.id} onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors text-left ${isActive ? "bg-sky-500/10 text-sky-400 border border-sky-500/20" : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/60 border border-transparent"}`}>
                <Icon size={16} />
                {tab.label}
              </button>
            );
          })}
        </nav>
        <div className="px-5 py-4 border-t border-slate-800">
          <button onClick={handleDownloadPdf}
            className="w-full flex items-center justify-center gap-2 bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold rounded-lg py-2.5 transition-colors">
            <Download size={13} /> Export PDF Report
          </button>
        </div>
      </aside>

      {/* Main content */}
      <div className="flex-1 flex flex-col">
        {/* Top bar */}
        <header className="h-14 border-b border-slate-800 flex items-center justify-between px-6 shrink-0">
          <h2 className="text-sm font-semibold text-slate-200">{TABS.find(t => t.id === activeTab)?.label}</h2>
          <div className="flex items-center gap-4 text-xs">
            {status.data && (
              <>
                <span className="flex items-center gap-1.5 text-slate-400">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
                  {status.data.status === "online" ? "System Online" : status.data.status}
                </span>
                <span className="text-slate-600">|</span>
                <span className="text-slate-400">LLM: {status.data.active_llm_provider}</span>
                <span className="text-slate-600">|</span>
                <span className="text-slate-400">Data Quality: {fmtPct(status.data.data_quality_score, 1)}</span>
              </>
            )}
          </div>
        </header>

        {/* Tab body */}
        <main key={activeTab} className="flex-1 overflow-y-auto p-6 animate-fadeIn">
          {activeTab === "overview" && <TabOverview data={overview.data} loading={overview.loading} error={overview.error} />}
          {activeTab === "validation" && <TabValidation data={validation.data} loading={validation.loading} error={validation.error} />}
          {activeTab === "forecasts" && (
            <TabForecasts
              forecasts={forecasts.data} dimensions={dimensions.data}
              selectedDimension={selectedDimension} onSelectDimension={setSelectedDimension}
              loading={forecasts.loading} error={forecasts.error}
            />
          )}
          {activeTab === "scenarios" && <TabScenarios scenarios={scenarios.data} loading={scenarios.loading} error={scenarios.error} />}
          {activeTab === "budget" && (
            <TabBudget
              simInputs={simInputs} onSimChange={handleSimChange} simResult={simResult} simLoading={simLoading}
              optInputs={optInputs} onOptChange={handleOptChange} optResult={optResult} optLoading={optLoading}
              onRunOptimize={handleRunOptimize}
            />
          )}
          {activeTab === "montecarlo" && <TabMontecarlo mc={montecarlo.data} loading={montecarlo.loading} error={montecarlo.error} />}
          {activeTab === "explainability" && <TabExplainability data={explainability.data} loading={explainability.loading} error={explainability.error} />}
          {activeTab === "risk" && <TabRisk risk={risk.data} insights={insights.data} loading={risk.loading} error={risk.error} />}
          {activeTab === "chat" && (
            <TabChat messages={chatMessages} input={chatInput} onInput={setChatInput} onSend={handleSendChat} loading={chatLoading} />
          )}
        </main>
      </div>
    </div>
  );
}
