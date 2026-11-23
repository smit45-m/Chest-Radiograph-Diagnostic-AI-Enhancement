import React from "react";
import { motion } from "framer-motion";
import {
  Activity,
  ChevronLeft,
  ChevronRight,
  Cpu,
  Database,
  FileText,
  Layers,
  LayoutGrid,
  Scan,
  Sparkles,
  UserRound,
} from "lucide-react";
import { NAV_PAGES, type PageId } from "../data/clinicalCatalog";
import type { AIConfigStatus, HealthStatus, UserRecord } from "../types/clinical";

interface SidebarNavProps {
  activePage: PageId;
  collapsed: boolean;
  onToggleCollapse: () => void;
  onSelectPage: (page: PageId) => void;
  health: HealthStatus | null;
  aiConfig: AIConfigStatus | null;
  users: UserRecord[];
  activeUserId: string | null;
  onSelectUser: (userId: string) => void;
  positiveCount: number;
  decisionActive: boolean;
  hasReport: boolean;
  batchStatCount: number | null;
  totalUserStudies: number;
  onOpenGemini: () => void;
}

const PAGE_ICONS: Record<
  PageId,
  React.ComponentType<{ className?: string; style?: React.CSSProperties }>
> = {
  overview: LayoutGrid,
  studio: Scan,
  consensus: Activity,
  report: FileText,
  triage: Layers,
  records: Database,
};

export const SidebarNav: React.FC<SidebarNavProps> = ({
  activePage,
  collapsed,
  onToggleCollapse,
  onSelectPage,
  health,
  aiConfig,
  users,
  activeUserId,
  onSelectUser,
  positiveCount,
  decisionActive,
  hasReport,
  batchStatCount,
  totalUserStudies,
  onOpenGemini,
}) => {
  const isCuda = health?.device?.includes("cuda");
  const geminiReady = Boolean(aiConfig?.gemini_configured || health?.gemini_configured);
  const activeUser = users.find((u) => u.id === activeUserId) || users[0] || null;

  return (
    <>
      {/* Desktop Architectural Clinical Navigation Rail */}
      <motion.aside
        aria-label="Primary clinical workspace navigation"
        animate={{ width: collapsed ? 76 : 244 }}
        transition={{ duration: 0.32, ease: [0.16, 1, 0.3, 1] }}
        className="no-print fixed top-0 bottom-0 left-0 z-40 hidden flex-col border-r lg:flex"
        style={{
          backgroundColor: "var(--bg-glass)",
          borderColor: "var(--border-subtle)",
          backdropFilter: "blur(24px) saturate(160%)",
          WebkitBackdropFilter: "blur(24px) saturate(160%)",
        }}
      >
        {/* Brand Header */}
        <div
          className="flex h-16 shrink-0 items-center justify-between border-b px-4"
          style={{ borderColor: "var(--border-subtle)" }}
        >
          <button
            type="button"
            onClick={() => onSelectPage("overview")}
            className="group flex min-w-0 cursor-pointer items-center gap-3 text-left"
          >
            <div
              className="relative flex h-9 w-9 shrink-0 items-center justify-center rounded-xl border"
              style={{
                borderColor: "var(--border-strong)",
                backgroundColor: "var(--bg-elevated)",
              }}
            >
              <svg
                viewBox="0 0 28 28"
                className="h-5 w-5 transition-transform duration-300 group-hover:scale-105"
                fill="none"
                aria-hidden="true"
              >
                <circle
                  cx="14"
                  cy="14"
                  r="10.5"
                  stroke="var(--accent-clinical)"
                  strokeOpacity="0.32"
                  strokeWidth="1.2"
                />
                <path
                  d="M4.5 14.5h4.2l2.3-5.2 3.2 10 2.3-4.8h7"
                  stroke="var(--accent-clinical)"
                  strokeWidth="1.9"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
              <span
                className="clinical-led absolute -top-0.5 -right-0.5 h-2 w-2 rounded-full ring-2"
                style={{
                  backgroundColor: "var(--accent-clinical)",
                  // @ts-expect-error CSS ring color token
                  "--tw-ring-color": "var(--bg-primary)",
                }}
              />
            </div>

            {!collapsed && (
              <div className="min-w-0">
                <div className="truncate font-display text-[13.5px] font-bold tracking-[0.04em]">
                  CXR·DIAGNOSTIC
                </div>
                <div
                  className="font-mono-tech text-[9.5px] tracking-[0.16em] uppercase"
                  style={{ color: "var(--accent-clinical)" }}
                >
                  THORACIC AI SUITE
                </div>
              </div>
            )}
          </button>

          {!collapsed && (
            <button
              type="button"
              onClick={onToggleCollapse}
              title="Collapse navigation rail"
              aria-label="Collapse navigation rail"
              className="flex h-7 w-7 cursor-pointer items-center justify-center rounded-lg border transition-opacity hover:opacity-80"
              style={{ borderColor: "var(--border-subtle)", color: "var(--text-muted)" }}
            >
              <ChevronLeft className="h-3.5 w-3.5" />
            </button>
          )}
        </div>

        {/* Active Persisted Patient / User Record Selector */}
        {!collapsed && activeUser && (
          <div
            className="border-b px-3 py-3"
            style={{ borderColor: "var(--border-subtle)" }}
          >
            <div className="mb-1.5 flex items-center justify-between px-1">
              <span
                className="font-mono-tech text-[9px] tracking-[0.18em] uppercase"
                style={{ color: "var(--text-muted)" }}
              >
                ACTIVE RECORD
              </span>
              <span
                className="flex items-center gap-1 font-mono-tech text-[9px]"
                style={{ color: "var(--accent-clinical)" }}
              >
                <span
                  className="h-1.5 w-1.5 rounded-full"
                  style={{ backgroundColor: "var(--accent-clinical)" }}
                />
                SQLite
              </span>
            </div>
            <div
              className="flex items-center gap-2 rounded-xl border px-2.5 py-2"
              style={{
                borderColor: "var(--border-strong)",
                backgroundColor: "var(--bg-elevated)",
              }}
            >
              <UserRound
                className="h-3.5 w-3.5 shrink-0"
                style={{ color: "var(--accent-clinical)" }}
              />
              <select
                aria-label="Active Patient or User Record"
                value={activeUser.id}
                onChange={(e) => onSelectUser(e.target.value)}
                className="w-full cursor-pointer truncate bg-transparent text-xs font-semibold focus:outline-none"
                style={{ color: "var(--text-primary)" }}
              >
                {users.map((u) => (
                  <option
                    key={u.id}
                    value={u.id}
                    style={{
                      backgroundColor: "var(--bg-elevated)",
                      color: "var(--text-primary)",
                    }}
                  >
                    {u.full_name} ({u.mrn})
                  </option>
                ))}
              </select>
            </div>
          </div>
        )}

        {/* Workspace Pages List */}
        <nav aria-label="Workspace pages" className="flex-1 space-y-1 overflow-y-auto px-3 py-4">
          {!collapsed && (
            <div
              className="px-2.5 pb-1.5 font-mono-tech text-[9.5px] tracking-[0.2em] uppercase"
              style={{ color: "var(--text-muted)" }}
            >
              DIAGNOSTIC MODULES
            </div>
          )}

          {NAV_PAGES.map((page) => {
            const Icon = PAGE_ICONS[page.id];
            const isActive = activePage === page.id;

            let badge: { text: string; alert?: boolean } | null = null;
            if (page.id === "studio" && positiveCount > 0) {
              badge = { text: `${positiveCount}`, alert: true };
            } else if (page.id === "consensus" && decisionActive) {
              badge = { text: "4/4" };
            } else if (page.id === "report" && hasReport) {
              badge = { text: "ACR" };
            } else if (page.id === "triage" && batchStatCount != null && batchStatCount > 0) {
              badge = { text: `${batchStatCount}`, alert: true };
            } else if (page.id === "records" && totalUserStudies > 0) {
              badge = { text: `${totalUserStudies}` };
            }

            return (
              <button
                key={page.id}
                type="button"
                onClick={() => onSelectPage(page.id)}
                title={collapsed ? `${page.label} (${page.keyHint})` : undefined}
                className={`group relative flex w-full cursor-pointer items-center rounded-2xl transition-colors ${
                  collapsed ? "justify-center px-0 py-3" : "gap-3 px-3 py-2 text-left"
                }`}
                style={{
                  color: isActive ? "var(--text-primary)" : "var(--text-secondary)",
                }}
              >
                {isActive && (
                  <motion.span
                    layoutId="sidebar-active-surface"
                    transition={{ type: "spring", stiffness: 380, damping: 32 }}
                    className="absolute inset-0 -z-10 rounded-2xl border"
                    style={{
                      backgroundColor: "var(--accent-clinical-dim)",
                      borderColor: "var(--border-strong)",
                    }}
                  />
                )}

                {isActive && (
                  <motion.span
                    layoutId="sidebar-active-bar"
                    transition={{ type: "spring", stiffness: 420, damping: 34 }}
                    className="absolute top-2 bottom-2 left-0 w-[3px] rounded-r-full"
                    style={{
                      backgroundColor: "var(--accent-clinical)",
                      boxShadow: "0 0 12px var(--accent-clinical-glow)",
                    }}
                  />
                )}

                <Icon
                  className="h-[17px] w-[17px] shrink-0 transition-transform duration-200 group-hover:scale-105"
                  style={{
                    color: isActive ? "var(--accent-clinical)" : "var(--text-muted)",
                  }}
                />

                {!collapsed && (
                  <>
                    <div className="min-w-0 flex-1">
                      <div className="truncate text-[13px] font-semibold leading-tight">
                        {page.label}
                      </div>
                      <div
                        className="mt-0.5 truncate text-[10.5px]"
                        style={{ color: "var(--text-muted)" }}
                      >
                        {page.shortDesc}
                      </div>
                    </div>

                    {badge ? (
                      <span
                        className="rounded-full border px-1.5 py-0.5 font-mono-tech text-[9px] font-medium"
                        style={{
                          borderColor: badge.alert
                            ? "var(--accent-alert)"
                            : "var(--accent-clinical)",
                          backgroundColor: badge.alert
                            ? "var(--accent-alert-dim)"
                            : "var(--accent-clinical-dim)",
                          color: badge.alert
                            ? "var(--accent-alert)"
                            : "var(--accent-clinical)",
                        }}
                      >
                        {badge.text}
                      </span>
                    ) : (
                      <span
                        className="rounded-md border px-1.5 py-0.5 font-mono-tech text-[9px] opacity-55 transition-opacity group-hover:opacity-100"
                        style={{
                          borderColor: "var(--border-subtle)",
                          color: "var(--text-muted)",
                        }}
                      >
                        {page.keyHint}
                      </span>
                    )}
                  </>
                )}
              </button>
            );
          })}
        </nav>

        {/* Bottom Hardware & Gemini Telemetry Dock */}
        <div
          className="shrink-0 space-y-2 border-t p-3"
          style={{ borderColor: "var(--border-subtle)" }}
        >
          {collapsed ? (
            <button
              type="button"
              onClick={onToggleCollapse}
              title="Expand navigation rail"
              aria-label="Expand navigation rail"
              className="mx-auto flex h-9 w-9 cursor-pointer items-center justify-center rounded-xl border transition-opacity hover:opacity-80"
              style={{ borderColor: "var(--border-subtle)", color: "var(--text-secondary)" }}
            >
              <ChevronRight className="h-4 w-4" />
            </button>
          ) : (
            <>
              <button
                type="button"
                onClick={onOpenGemini}
                className="flex w-full cursor-pointer items-center justify-between rounded-2xl border px-3 py-2 text-left transition-colors"
                style={{
                  borderColor: geminiReady ? "var(--accent-clinical)" : "var(--border-subtle)",
                  backgroundColor: geminiReady ? "var(--accent-clinical-dim)" : "var(--bg-elevated)",
                }}
              >
                <div className="min-w-0">
                  <div className="flex items-center gap-1.5 font-mono-tech text-[10px] font-medium uppercase">
                    <Sparkles
                      className="h-3 w-3 shrink-0"
                      style={{ color: "var(--accent-clinical)" }}
                    />
                    <span className="truncate">
                      {geminiReady ? "Gemini Connected" : "Connect Gemini"}
                    </span>
                  </div>
                  <div
                    className="mt-0.5 truncate font-mono-tech text-[9.5px]"
                    style={{ color: "var(--text-muted)" }}
                  >
                    {aiConfig?.gemini_flash_model || "gemini-3.5-flash-lite"}
                  </div>
                </div>
                <span
                  className="h-2 w-2 shrink-0 rounded-full clinical-led"
                  style={{
                    backgroundColor: geminiReady ? "var(--accent-clinical)" : "var(--text-muted)",
                  }}
                />
              </button>

              <div
                className="flex items-center justify-between rounded-xl border px-3 py-1.5 font-mono-tech text-[10px]"
                style={{
                  borderColor: "var(--border-subtle)",
                  backgroundColor: "var(--bg-secondary)",
                  color: "var(--text-muted)",
                }}
              >
                <span className="flex items-center gap-1.5 truncate">
                  <Cpu className="h-3 w-3 shrink-0" style={{ color: "var(--accent-optical)" }} />
                  <span className="truncate">{isCuda ? "RTX CUDA FP16" : "CLINICAL CPU"}</span>
                </span>
                <span style={{ color: "var(--accent-clinical)" }}>LIVE</span>
              </div>
            </>
          )}
        </div>
      </motion.aside>

      {/* Mobile Bottom Dock */}
      <nav
        aria-label="Mobile workspace navigation"
        className="no-print fixed right-3 bottom-3 left-3 z-40 flex items-center justify-around rounded-2xl border px-1.5 py-2 lg:hidden"
        style={{
          backgroundColor: "var(--bg-glass)",
          borderColor: "var(--border-strong)",
          backdropFilter: "blur(20px)",
          boxShadow: "var(--shadow-float)",
        }}
      >
        {NAV_PAGES.map((page) => {
          const Icon = PAGE_ICONS[page.id];
          const isActive = activePage === page.id;
          return (
            <button
              key={page.id}
              type="button"
              onClick={() => onSelectPage(page.id)}
              className="flex cursor-pointer flex-col items-center gap-1 rounded-xl px-2 py-1.5 text-[9.5px] font-medium"
              style={{
                backgroundColor: isActive ? "var(--accent-clinical-dim)" : "transparent",
                color: isActive ? "var(--accent-clinical)" : "var(--text-secondary)",
              }}
            >
              <Icon className="h-4 w-4" />
              <span className="truncate">{page.label.split(" ")[0]}</span>
            </button>
          );
        })}
      </nav>
    </>
  );
};
